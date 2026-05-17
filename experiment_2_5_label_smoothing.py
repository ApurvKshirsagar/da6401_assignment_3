"""
experiment_2_5_label_smoothing.py
DA6401 Assignment 3 - W&B Report: Question 2.5
Decoder Sensitivity: Label Smoothing

Trains two Transformers:
  1. eps=0.1  (label smoothing, paper default)
  2. eps=0.0  (standard cross-entropy)

Logs: train/val loss, prediction confidence, perplexity,
      confidence histograms, reliability diagrams, test BLEU.

Usage:
    python experiment_2_5_label_smoothing.py
"""

import math
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
from tqdm import tqdm
import spacy

from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']


# ======================================================================
#  LOCAL JSONL DATASET / BLEU UTILITIES
#  Avoids importing Hugging Face datasets/pyarrow, which crashes in this
#  Windows venv before the script can print anything.
# ======================================================================

def find_multi30k_snapshot():
    root = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--bentrevett--multi30k" / "snapshots"
    if not root.exists():
        raise FileNotFoundError(f"Could not find cached Multi30k snapshot under {root}")
    for snapshot in root.iterdir():
        if all((snapshot / name).exists() for name in ["train.jsonl", "val.jsonl", "test.jsonl"]):
            return snapshot
    raise FileNotFoundError(f"No train.jsonl/val.jsonl/test.jsonl found under {root}")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class Multi30kJsonlDataset:
    def __init__(self, split="train", snapshot_dir=None):
        self.split = split
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else find_multi30k_snapshot()
        file_name = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}[split]
        self.raw = read_jsonl(self.snapshot_dir / file_name)
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, min_freq=2):
        train_rows = read_jsonl(self.snapshot_dir / "train.jsonl")
        de_counter, en_counter = Counter(), Counter()
        for item in train_rows:
            de_counter.update(self.tokenize_de(item["de"]))
            en_counter.update(self.tokenize_en(item["en"]))
        self.src_itos = SPECIALS + [w for w, c in de_counter.items() if c >= min_freq]
        self.tgt_itos = SPECIALS + [w for w, c in en_counter.items() if c >= min_freq]
        self.src_stoi = {w: i for i, w in enumerate(self.src_itos)}
        self.tgt_stoi = {w: i for i, w in enumerate(self.tgt_itos)}

    def process_data(self):
        self.data = []
        for item in self.raw:
            src_ids = [SOS_IDX] + [self.src_stoi.get(t, UNK_IDX) for t in self.tokenize_de(item["de"])] + [EOS_IDX]
            tgt_ids = [SOS_IDX] + [self.tgt_stoi.get(t, UNK_IDX) for t in self.tokenize_en(item["en"])] + [EOS_IDX]
            self.data.append((src_ids, tgt_ids))
        return self.data


class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def get_dataloader(split, dataset_obj, batch_size=128, shuffle=True):
    return DataLoader(
        TranslationDataset(dataset_obj.data),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.config,
    }, path)
    print(f"  Checkpoint saved -> {path} (epoch {epoch})", flush=True)


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break
    return ys


def evaluate_bleu(model, dataloader, tgt_vocab, device="cpu", max_len=100):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    model.eval()
    hypotheses, references = [], []
    smoother = SmoothingFunction().method1
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i, pad_idx=PAD_IDX).to(device)
                pred = greedy_decode(
                    model, src_i, src_mask, max_len=max_len,
                    start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device,
                )
                hyp_tokens = []
                for idx in pred.squeeze(0).tolist():
                    if idx == EOS_IDX:
                        break
                    if idx != SOS_IDX:
                        hyp_tokens.append(tgt_vocab.itos[idx])
                ref_tokens = [
                    tgt_vocab.itos[idx] for idx in tgt[i].tolist()
                    if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
                ]
                hypotheses.append(hyp_tokens)
                references.append([ref_tokens])
    return corpus_bleu(references, hypotheses, smoothing_function=smoother) * 100


# ======================================================================
#  LABEL SMOOTHING LOSS  (eps=0.0 - standard CE)
# ======================================================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.criterion  = nn.KLDivLoss(reduction='sum')

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            if self.smoothing == 0.0:
                dist = torch.zeros_like(log_probs)
                dist.scatter_(1, target.unsqueeze(1), 1.0)
            else:
                dist = torch.full_like(log_probs,
                                       self.smoothing / (self.vocab_size - 2))
                dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            dist[:, self.pad_idx] = 0.0
            dist[target == self.pad_idx] = 0.0
        loss     = self.criterion(log_probs, dist)
        n_tokens = (target != self.pad_idx).sum().clamp(min=1)
        return loss / n_tokens


# ======================================================================
#  CONFIDENCE METRICS
# ======================================================================

@torch.no_grad()
def confidence_stats(model, loader, device, max_batches=25):
    model.eval()
    correct_probs, top1_probs, correct_flags = [], [], []
    total_nll, total_tok = 0.0, 0

    for i, (src, tgt) in enumerate(loader):
        if i >= max_batches: break
        src=src.to(device); tgt=tgt.to(device)
        ti=tgt[:,:-1]; tt=tgt[:,1:]
        sm=make_src_mask(src,pad_idx=PAD_IDX).to(device)
        tm=make_tgt_mask(ti, pad_idx=PAD_IDX).to(device)
        logits=model(src,ti,sm,tm)
        B,T,V=logits.shape
        probs=F.softmax(logits,dim=-1)
        tf=tt.reshape(-1); pf=probs.reshape(-1,V); lf=F.log_softmax(logits,dim=-1).reshape(-1,V)

        non_pad = (tf != PAD_IDX)
        cp = pf[torch.arange(B*T), tf.clamp(min=0)]
        correct_probs.extend(cp[non_pad].cpu().tolist())
        t1p, t1pred = pf.max(-1)
        top1_probs.extend(t1p[non_pad].cpu().tolist())
        correct_flags.extend((t1pred[non_pad]==tf[non_pad]).cpu().tolist())
        nll = -lf[torch.arange(B*T), tf.clamp(min=0)]
        total_nll += nll[non_pad].sum().item(); total_tok += non_pad.sum().item()

    ece = expected_calibration_error(top1_probs, correct_flags)

    return dict(
        mean_confidence = float(np.mean(correct_probs)) if correct_probs else 0.0,
        mean_top1_prob  = float(np.mean(top1_probs))    if top1_probs    else 0.0,
        perplexity      = math.exp(total_nll / max(total_tok, 1)),
        ece             = ece,
        confidences     = correct_probs,
        top1_probs      = top1_probs,
        correct_flags   = correct_flags,
    )


def expected_calibration_error(top1_probs, correct_flags, n_bins=10):
    if not top1_probs:
        return 0.0
    top1_probs = np.asarray(top1_probs)
    correct_flags = np.asarray(correct_flags, dtype=np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (top1_probs >= lo) & (top1_probs <= hi)
        else:
            mask = (top1_probs >= lo) & (top1_probs < hi)
        if not mask.any():
            continue
        acc = correct_flags[mask].mean()
        conf = top1_probs[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


# ======================================================================
#  PLOTS
# ======================================================================

def hist_fig(confs, title, color='steelblue'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confs, bins=50, range=(0,1), color=color, alpha=0.8,
            edgecolor='white', linewidth=0.3)
    mu = np.mean(confs)
    ax.axvline(mu, color='red', ls='--', lw=2, label=f'Mean={mu:.3f}')
    ax.set_xlabel('P(correct token)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); plt.tight_layout(); return fig


def reliability_fig(top1_probs, correct_flags, title, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot([0,1],[0,1],'k--',lw=1.5,label='Perfect')
    for i in range(n_bins):
        lo,hi = bins[i],bins[i+1]
        mask  = [lo<=p<hi for p in top1_probs]
        if not any(mask): continue
        acc  = np.mean([correct_flags[j] for j,m in enumerate(mask) if m])
        conf = np.mean([top1_probs[j]    for j,m in enumerate(mask) if m])
        color = 'tomato' if conf-acc>0.1 else 'steelblue'
        ax.bar(conf, acc, width=1/n_bins*0.8, color=color, alpha=0.7)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Confidence',fontsize=11); ax.set_ylabel('Accuracy',fontsize=11)
    ax.set_title(title,fontsize=12,fontweight='bold'); ax.legend(fontsize=10)
    plt.tight_layout(); return fig


# ======================================================================
#  CONFIG
# ======================================================================

CFG = dict(d_model=256, N=3, num_heads=8, d_ff=512, dropout=0.1,
           warmup_steps=2000, batch_size=128, num_epochs=30, min_freq=2)

WANDB_PROJECT = "da6401-a3"
WANDB_GROUP   = "2.5_label_smoothing"

SNAPSHOT_EPOCHS = {0, 14, 29}


# ======================================================================
#  DATA
# ======================================================================

def build_data(cfg):
    print("Building datasets...", flush=True)
    snapshot_dir = find_multi30k_snapshot()
    print(f"Using cached Multi30k JSONL snapshot: {snapshot_dir}", flush=True)
    tr = Multi30kJsonlDataset(split='train', snapshot_dir=snapshot_dir)
    tr.build_vocab(min_freq=cfg['min_freq'])
    va = Multi30kJsonlDataset(split='validation', snapshot_dir=snapshot_dir)
    te = Multi30kJsonlDataset(split='test', snapshot_dir=snapshot_dir)
    for ds in [va,te]:
        ds.src_stoi=tr.src_stoi; ds.src_itos=tr.src_itos
        ds.tgt_stoi=tr.tgt_stoi; ds.tgt_itos=tr.tgt_itos
    for ds in [tr,va,te]: ds.process_data()
    trl = get_dataloader('train',      tr, batch_size=cfg['batch_size'], shuffle=True)
    val = get_dataloader('validation', va, batch_size=cfg['batch_size'], shuffle=False)
    val_bleu = get_dataloader('validation', va, batch_size=1, shuffle=False)
    tel = get_dataloader('test',       te, batch_size=1, shuffle=False)
    length_stats = {
        "train/max_src_len": max(len(src) for src, _ in tr.data),
        "train/max_tgt_len": max(len(tgt) for _, tgt in tr.data),
        "val/max_src_len":   max(len(src) for src, _ in va.data),
        "val/max_tgt_len":   max(len(tgt) for _, tgt in va.data),
        "test/max_src_len":  max(len(src) for src, _ in te.data),
        "test/max_tgt_len":  max(len(tgt) for _, tgt in te.data),
    }
    return tr, trl, val, val_bleu, tel, length_stats


# ======================================================================
#  TRAINING
# ======================================================================

def run(smoothing, cfg, train_ds, trl, val, val_bleu_loader, tel, device, length_stats):
    name  = f"eps_{str(smoothing).replace('.','_')}"
    color = 'royalblue' if smoothing > 0 else 'tomato'

    wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP,
               name=name, config={**cfg, **length_stats, "smoothing":smoothing}, reinit=True)

    sv = len(train_ds.src_itos); tv = len(train_ds.tgt_itos)
    model = Transformer(sv, tv, d_model=cfg['d_model'], N=cfg['N'],
                        num_heads=cfg['num_heads'], d_ff=cfg['d_ff'],
                        dropout=cfg['dropout']).to(device)

    loss_fn   = LabelSmoothingLoss(tv, PAD_IDX, smoothing=smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9,0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])

    best_val, best_ckpt = float('inf'), f"best_2_5_{name}.pt"
    tr_hist, va_hist, conf_hist, ppl_hist, epochs = [], [], [], [], []

    for epoch in range(cfg['num_epochs']):
        model.train(); total, tok = 0.0, 0
        for src, tgt in tqdm(trl, desc=f"[{name}] E{epoch}", leave=False):
            src=src.to(device); tgt=tgt.to(device)
            ti=tgt[:,:-1]; tt=tgt[:,1:]
            sm=make_src_mask(src,pad_idx=PAD_IDX).to(device)
            tm=make_tgt_mask(ti, pad_idx=PAD_IDX).to(device)
            out=model(src,ti,sm,tm); lf=out.reshape(-1,out.size(-1)); tf=tt.reshape(-1)
            loss=loss_fn(lf,tf)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); scheduler.step()
            nt=(tf!=PAD_IDX).sum().item(); total+=loss.item()*nt; tok+=nt
        tl = total/max(tok,1)

        model.eval(); vtot,vtok=0.0,0
        with torch.no_grad():
            for src,tgt in val:
                src=src.to(device); tgt=tgt.to(device)
                ti=tgt[:,:-1]; tt=tgt[:,1:]
                sm=make_src_mask(src,pad_idx=PAD_IDX).to(device)
                tm=make_tgt_mask(ti, pad_idx=PAD_IDX).to(device)
                out=model(src,ti,sm,tm); lf=out.reshape(-1,out.size(-1)); tf=tt.reshape(-1)
                loss=loss_fn(lf,tf); nt=(tf!=PAD_IDX).sum().item()
                vtot+=loss.item()*nt; vtok+=nt
        vl = vtot/max(vtok,1)

        stats = confidence_stats(model, val, device, max_batches=20)
        tppl  = math.exp(min(tl,10))
        vppl  = stats['perplexity']

        wandb.log({"epoch/train_loss":      tl,
                   "epoch/val_loss":        vl,
                   "epoch/train_ppl":       tppl,
                   "epoch/val_ppl":         vppl,
                   "epoch/mean_confidence": stats['mean_confidence'],
                   "epoch/mean_top1_prob":  stats['mean_top1_prob'],
                   "epoch/ece":             stats['ece'],
                   "epoch/lr":              optimizer.param_groups[0]['lr'],
                   "epoch":                 epoch})

        tr_hist.append(tl); va_hist.append(vl)
        conf_hist.append(stats['mean_confidence']); ppl_hist.append(vppl)
        epochs.append(epoch)

        # Snapshots
        if epoch in SNAPSHOT_EPOCHS:
            fig = hist_fig(stats['confidences'],
                           f'Confidence Histogram - eps={smoothing} - Epoch {epoch}', color)
            wandb.log({f"conf/hist_epoch_{epoch:02d}": wandb.Image(fig)}); plt.close(fig)
            fig = reliability_fig(stats['top1_probs'], stats['correct_flags'],
                                  f'Reliability Diagram - eps={smoothing} - Epoch {epoch}')
            wandb.log({f"conf/reliability_epoch_{epoch:02d}": wandb.Image(fig)}); plt.close(fig)

        print(f"  [{name}] E{epoch:2d} | train {tl:.4f} | val {vl:.4f} | "
              f"conf {stats['mean_confidence']:.4f} | ece {stats['ece']:.4f} | ppl {vppl:.2f}",
              flush=True)

        if vl < best_val:
            best_val = vl
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

    # BLEU on best checkpoint
    class VW:
        def __init__(self,i): self.itos=i
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    val_bleu = evaluate_bleu(model, val_bleu_loader, VW(train_ds.tgt_itos), device=device)
    test_bleu = evaluate_bleu(model, tel, VW(train_ds.tgt_itos), device=device)
    print(f"  [{name}] Validation BLEU: {val_bleu:.2f}", flush=True)
    print(f"  [{name}] Test BLEU: {test_bleu:.2f}", flush=True)

    final = confidence_stats(model, val, device, max_batches=30)
    fig = hist_fig(final['confidences'], f'Final Confidence - eps={smoothing}', color)
    wandb.log({"conf/final_hist": wandb.Image(fig)}); plt.close(fig)
    fig = reliability_fig(final['top1_probs'], final['correct_flags'],
                          f'Final Reliability Diagram - eps={smoothing}')
    wandb.log({"conf/final_reliability": wandb.Image(fig)}); plt.close(fig)

    wandb.log({"validation/bleu": val_bleu,
               "test/bleu": test_bleu,
               "test/best_val_loss": best_val,
               "test/final_confidence": final['mean_confidence'],
               "test/final_top1_prob": final['mean_top1_prob'],
               "test/final_ppl": final['perplexity'],
               "test/final_ece": final['ece']})

    # Summary table
    rows = [[e,tr_hist[i],va_hist[i],conf_hist[i],ppl_hist[i]]
            for i,e in enumerate(epochs)]
    wandb.log({"epoch_summary": wandb.Table(
        columns=["epoch","train_loss","val_loss","mean_confidence","val_ppl"],
        data=rows)})

    wandb.finish()
    return tr_hist, va_hist, conf_hist, ppl_hist, epochs, val_bleu, test_bleu, final


# ======================================================================
#  MAIN
# ======================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    train_ds, trl, val, val_bleu_loader, tel, length_stats = build_data(CFG)
    print("Length stats:", length_stats, flush=True)

    print("\n" + "="*60 + "\nEXP 1: Label Smoothing eps=0.1\n" + "="*60)
    str_,sva,sc,sp,se,sval_bleu,stest_bleu,sfin = run(
        0.1, CFG, train_ds, trl, val, val_bleu_loader, tel, device, length_stats
    )

    print("\n" + "="*60 + "\nEXP 2: Standard CE eps=0.0\n" + "="*60)
    ctr,cva,cc,cp,ce,cval_bleu,ctest_bleu,cfin = run(
        0.0, CFG, train_ds, trl, val, val_bleu_loader, tel, device, length_stats
    )

    # -- Comparison run ------------------------------------------------
    wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP,
               name="2.5_comparison", config={**CFG, **length_stats}, reinit=True)

    # Loss overlay
    data = []
    for e,(t,v) in enumerate(zip(str_,sva)): data.append([e,"eps=0.1",t,v])
    for e,(t,v) in enumerate(zip(ctr,cva)):  data.append([e,"eps=0.0",t,v])
    tbl = wandb.Table(columns=["epoch","condition","train_loss","val_loss"],data=data)
    wandb.log({"cmp/train_loss": wandb.plot.line(tbl,"epoch","train_loss",stroke="condition",title="Train Loss")})
    wandb.log({"cmp/val_loss":   wandb.plot.line(tbl,"epoch","val_loss",  stroke="condition",title="Val Loss")})

    # Confidence overlay
    cd = []
    for e,c in zip(se,sc): cd.append([e,"eps=0.1 (smoothed)",  c])
    for e,c in zip(ce,cc): cd.append([e,"eps=0.0 (standard CE)",c])
    ct = wandb.Table(columns=["epoch","condition","mean_confidence"],data=cd)
    wandb.log({"cmp/confidence": wandb.plot.line(ct,"epoch","mean_confidence",stroke="condition",
                                                  title="Prediction Confidence per Epoch")})

    # Perplexity overlay
    pd2 = []
    for e,p in zip(se,sp): pd2.append([e,"eps=0.1",p])
    for e,p in zip(ce,cp): pd2.append([e,"eps=0.0",p])
    pt = wandb.Table(columns=["epoch","condition","val_ppl"],data=pd2)
    wandb.log({"cmp/val_ppl": wandb.plot.line(pt,"epoch","val_ppl",stroke="condition",
                                               title="Val Perplexity per Epoch")})

    # BLEU bars
    bt = wandb.Table(columns=["condition","validation_bleu","test_bleu"],
                     data=[["eps=0.1 (smoothed)",sval_bleu,stest_bleu],
                           ["eps=0.0 (std CE)",cval_bleu,ctest_bleu]])
    wandb.log({
        "cmp/validation_bleu": wandb.plot.bar(
            bt, "condition", "validation_bleu", title="Validation BLEU"
        ),
        "cmp/test_bleu": wandb.plot.bar(
            bt, "condition", "test_bleu", title="Test BLEU"
        ),
    })

    # Final calibration summary
    cal = wandb.Table(
        columns=["condition","mean_correct_token_prob","mean_top1_prob","ece","val_ppl"],
        data=[
            ["eps=0.1 (smoothed)", sfin['mean_confidence'], sfin['mean_top1_prob'], sfin['ece'], sfin['perplexity']],
            ["eps=0.0 (std CE)",   cfin['mean_confidence'], cfin['mean_top1_prob'], cfin['ece'], cfin['perplexity']],
        ],
    )
    wandb.log({
        "cmp/final_ece": wandb.plot.bar(cal, "condition", "ece", title="Final Expected Calibration Error"),
        "cmp/final_top1_confidence": wandb.plot.bar(cal, "condition", "mean_top1_prob", title="Final Mean Top-1 Confidence"),
    })

    # Side-by-side final confidence histograms
    fig, axes = plt.subplots(1,2,figsize=(14,4),sharey=True)
    for ax, confs, title, color in zip(
        axes,
        [sfin['confidences'], cfin['confidences']],
        ['eps=0.1 (Label Smoothing)','eps=0.0 (Standard CE)'],
        ['royalblue','tomato'],
    ):
        ax.hist(confs, bins=50, range=(0,1), color=color, alpha=0.8,
                edgecolor='white', linewidth=0.3)
        mu = np.mean(confs)
        ax.axvline(mu, color='black', ls='--', lw=2, label=f'Mean={mu:.3f}')
        ax.set_xlabel('P(correct token)',fontsize=11); ax.set_ylabel('Count',fontsize=11)
        ax.set_title(title,fontsize=11,fontweight='bold'); ax.legend(fontsize=10)
    fig.suptitle('Final Prediction Confidence Distribution',fontsize=13,fontweight='bold')
    plt.tight_layout()
    wandb.log({"cmp/final_confidence_histograms": wandb.Image(fig)}); plt.close(fig)

    # Side-by-side reliability diagrams
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    for ax, stats, title in zip(
        axes,
        [sfin, cfin],
        ['Reliability - eps=0.1','Reliability - eps=0.0'],
    ):
        bins = np.linspace(0,1,11)
        ax.plot([0,1],[0,1],'k--',lw=1.5,label='Perfect')
        for i in range(10):
            lo,hi = bins[i],bins[i+1]
            mask  = [lo<=p<hi for p in stats['top1_probs']]
            if not any(mask): continue
            acc  = np.mean([stats['correct_flags'][j] for j,m in enumerate(mask) if m])
            conf = np.mean([stats['top1_probs'][j]    for j,m in enumerate(mask) if m])
            color= 'tomato' if conf-acc>0.1 else 'steelblue'
            ax.bar(conf,acc,width=0.08,color=color,alpha=0.7)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel('Confidence',fontsize=11); ax.set_ylabel('Accuracy',fontsize=11)
        ax.set_title(title,fontsize=11,fontweight='bold'); ax.legend(fontsize=9)
    fig.suptitle('Calibration: Smoothing vs Standard CE',fontsize=13,fontweight='bold')
    plt.tight_layout()
    wandb.log({"cmp/reliability_diagrams": wandb.Image(fig)}); plt.close(fig)

    print(f"\n{'='*60}")
    print(f"  eps=0.1 | Val BLEU={sval_bleu:.2f} | Test BLEU={stest_bleu:.2f} | "
          f"Conf={sfin['mean_confidence']:.4f} | Top1={sfin['mean_top1_prob']:.4f} | "
          f"ECE={sfin['ece']:.4f} | PPL={sfin['perplexity']:.2f}")
    print(f"  eps=0.0 | Val BLEU={cval_bleu:.2f} | Test BLEU={ctest_bleu:.2f} | "
          f"Conf={cfin['mean_confidence']:.4f} | Top1={cfin['mean_top1_prob']:.4f} | "
          f"ECE={cfin['ece']:.4f} | PPL={cfin['perplexity']:.2f}")
    print(f"{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    main()
