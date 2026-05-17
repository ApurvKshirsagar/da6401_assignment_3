"""
experiment_2_4_positional_encoding.py
DA6401 Assignment 3 - W&B Report: Question 2.4
Positional Encoding vs. Learned Embeddings

Trains two Transformers:
  1. Sinusoidal PE  (fixed formula, original paper)
  2. Learned PE     (torch.nn.Embedding, trained from data)

Usage:
    python experiment_2_4_positional_encoding.py
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
from model import (
    MultiHeadAttention, PositionwiseFeedForward,
    EncoderLayer, DecoderLayer, Encoder, Decoder,
    make_src_mask, make_tgt_mask,
    PositionalEncoding,
)

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']


# ======================================================================
#  LOCAL JSONL DATASET
#  Avoids importing Hugging Face datasets/pyarrow, which is crashing in
#  this Windows venv before the script can print anything.
# ======================================================================

def find_multi30k_snapshot():
    root = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--bentrevett--multi30k" / "snapshots"
    if not root.exists():
        raise FileNotFoundError(
            "Could not find cached Multi30k JSONL files. Expected them under "
            f"{root}. Run your original dataset script in a working environment once."
        )
    for snapshot in root.iterdir():
        if all((snapshot / name).exists() for name in ["train.jsonl", "val.jsonl", "test.jsonl"]):
            return snapshot
    raise FileNotFoundError(f"No snapshot with train.jsonl/val.jsonl/test.jsonl found under {root}")


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


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            smooth_dist[:, self.pad_idx] = 0.0
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0
        loss = self.criterion(log_probs, smooth_dist)
        return loss / (~pad_mask).sum().clamp(min=1)


def save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": model.config,
    }, path)
    print(f"  Checkpoint saved -> {path} (epoch {epoch})")


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
#  LEARNED POSITIONAL EMBEDDING
# ======================================================================

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)

    def forward(self, x):
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


# ======================================================================
#  FLEXIBLE TRANSFORMER
# ======================================================================

class FlexibleTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, N=3, num_heads=8, d_ff=512,
                 dropout=0.1, pe_type="sinusoidal", max_len=256):
        super().__init__()
        self.d_model = d_model
        self.pe_type = pe_type
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        if pe_type == "sinusoidal":
            self.pos_enc = PositionalEncoding(d_model, dropout, max_len=max_len)
        else:
            self.pos_enc = LearnedPositionalEmbedding(d_model, dropout, max_len)
        enc = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc, N)
        self.decoder = Decoder(dec, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)
        self.config  = dict(src_vocab_size=src_vocab_size,
                            tgt_vocab_size=tgt_vocab_size,
                            d_model=d_model, N=N, num_heads=num_heads,
                            d_ff=d_ff, dropout=dropout, pe_type=pe_type)
        self._init()

    def _init(self):
        for name, p in self.named_parameters():
            if 'pos_enc.embedding' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        return self.encoder(
            self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.fc_out(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


# ======================================================================
#  CONFIG
# ======================================================================

CFG = dict(d_model=256, N=3, num_heads=8, d_ff=512, dropout=0.1,
           warmup_steps=2000, batch_size=128, num_epochs=30,
           smoothing=0.1, min_freq=2, max_len=512)

WANDB_PROJECT = "da6401-a3"
WANDB_GROUP   = "2.4_positional_encoding"


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
    for ds in [va, te]:
        ds.src_stoi = tr.src_stoi; ds.src_itos = tr.src_itos
        ds.tgt_stoi = tr.tgt_stoi; ds.tgt_itos = tr.tgt_itos
    for ds in [tr, va, te]:
        ds.process_data()

    length_stats = {
        "train/max_src_len": max(len(src) for src, _ in tr.data),
        "train/max_tgt_len": max(len(tgt) for _, tgt in tr.data),
        "val/max_src_len":   max(len(src) for src, _ in va.data),
        "val/max_tgt_len":   max(len(tgt) for _, tgt in va.data),
        "test/max_src_len":  max(len(src) for src, _ in te.data),
        "test/max_tgt_len":  max(len(tgt) for _, tgt in te.data),
        "pe/max_len":        cfg["max_len"],
    }

    trl = get_dataloader('train',      tr, batch_size=cfg['batch_size'], shuffle=True)
    val = get_dataloader('validation', va, batch_size=cfg['batch_size'], shuffle=False)
    val_bleu = get_dataloader('validation', va, batch_size=1, shuffle=False)
    tel = get_dataloader('test',       te, batch_size=1, shuffle=False)
    return tr, trl, val, val_bleu, tel, length_stats


# ======================================================================
#  PE VISUALISATION
# ======================================================================

def pe_matrix(model):
    with torch.no_grad():
        if model.pe_type == "sinusoidal":
            return model.pos_enc.pe[0].cpu().numpy()      # [max_len, d_model]
        else:
            return model.pos_enc.embedding.weight.detach().cpu().numpy()


def plot_heatmap(mat, title, max_pos=50):
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat[:max_pos], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Dimension'); ax.set_ylabel('Position')
    ax.set_title(title, fontweight='bold'); plt.colorbar(im, ax=ax)
    plt.tight_layout(); return fig


def plot_dim_curves(mat, title, dims=None):
    if dims is None:
        dims = [d for d in [0,1,4,5,16,17,64,65] if d < mat.shape[1]]
    max_p = min(100, mat.shape[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    colors  = plt.cm.tab10(np.linspace(0,1,len(dims)))
    for i, d in enumerate(dims):
        ax.plot(mat[:max_p, d], label=f'dim {d}', color=colors[i], linewidth=1.5)
    ax.set_xlabel('Position'); ax.set_ylabel('Value')
    ax.set_title(title, fontweight='bold')
    ax.legend(ncol=4, fontsize=8); ax.axhline(0, color='k', lw=0.5, ls='--')
    plt.tight_layout(); return fig


def plot_cosine_sim(mat, title, max_pos=30):
    m   = mat[:max_pos]
    n   = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    sim = (m/n) @ (m/n).T
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Position'); ax.set_ylabel('Position')
    ax.set_title(title, fontweight='bold'); plt.colorbar(im, ax=ax)
    plt.tight_layout(); return fig


def plot_extrapolation_view(mat, title, train_max_len, max_pos=160):
    """Show seen vs unseen positional rows for the report discussion."""
    max_pos = min(max_pos, mat.shape[0])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat[:max_pos], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.axhline(train_max_len - 0.5, color='black', lw=2, ls='--',
               label=f'max train length = {train_max_len}')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Position')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


# ======================================================================
#  QUICK VAL BLEU
# ======================================================================

def quick_bleu(model, val_loader, tgt_itos, device, max_batches=15):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    model.eval(); hyps, refs = [], []
    smoother = SmoothingFunction().method1
    with torch.no_grad():
        for i, (src, tgt) in enumerate(val_loader):
            if i >= max_batches: break
            src = src.to(device); tgt = tgt.to(device)
            for j in range(src.size(0)):
                s    = src[j].unsqueeze(0)
                smask = make_src_mask(s, pad_idx=PAD_IDX).to(device)
                ys   = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
                for _ in range(80):
                    tmask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
                    logit = model.decode(model.encode(s, smask), smask, ys, tmask)
                    nxt   = logit[:,-1,:].argmax(-1, keepdim=True)
                    ys    = torch.cat([ys, nxt], 1)
                    if nxt.item() == EOS_IDX: break
                hyp = [tgt_itos[i] for i in ys.squeeze().tolist()
                       if i not in (SOS_IDX, EOS_IDX, PAD_IDX)]
                ref = [tgt_itos[i] for i in tgt[j].tolist()
                       if i not in (SOS_IDX, EOS_IDX, PAD_IDX)]
                hyps.append(hyp); refs.append([ref])
    return round(corpus_bleu(refs, hyps, smoothing_function=smoother)*100, 2)


# ======================================================================
#  TRAINING
# ======================================================================

def run(pe_type, cfg, train_ds, trl, val, val_bleu_loader, tel, device, length_stats):
    name = f"pe_{pe_type}"
    wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP,
               name=name, config={**cfg, **length_stats, "pe_type": pe_type}, reinit=True)

    sv = len(train_ds.src_itos); tv = len(train_ds.tgt_itos)
    model = FlexibleTransformer(sv, tv, d_model=cfg['d_model'], N=cfg['N'],
                                num_heads=cfg['num_heads'], d_ff=cfg['d_ff'],
                                dropout=cfg['dropout'], pe_type=pe_type,
                                max_len=cfg['max_len']).to(device)

    print(f"  [{name}] params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    # Log PE before training
    mat = pe_matrix(model)
    label = "Sinusoidal" if pe_type=="sinusoidal" else "Learned"
    wandb.log({"pe/heatmap_before":   wandb.Image(plot_heatmap(mat, f'{label} PE - Before Training'))})
    wandb.log({"pe/dim_curves_before": wandb.Image(plot_dim_curves(mat, f'{label} PE Dim Curves - Before Training'))})
    wandb.log({"pe/extrapolation_view_before": wandb.Image(
        plot_extrapolation_view(
            mat,
            f'{label} PE Rows: Seen Training Positions vs Longer Positions',
            train_max_len=length_stats["train/max_src_len"],
        )
    )})
    plt.close('all')

    loss_fn   = LabelSmoothingLoss(tv, PAD_IDX, smoothing=cfg['smoothing'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9,0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])

    best_val, best_ckpt = float('inf'), f"best_2_4_{pe_type}.pt"
    tr_hist, va_hist, bleu_hist = [], [], []

    for epoch in range(cfg['num_epochs']):
        model.train(); total, tokens = 0.0, 0
        for src, tgt in tqdm(trl, desc=f"[{name}] E{epoch}", leave=False):
            src = src.to(device); tgt = tgt.to(device)
            ti = tgt[:,:-1]; tt = tgt[:,1:]
            sm = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tm = make_tgt_mask(ti,  pad_idx=PAD_IDX).to(device)
            out = model(src, ti, sm, tm)
            lf  = out.reshape(-1, out.size(-1)); tf = tt.reshape(-1)
            loss = loss_fn(lf, tf)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            nt = (tf != PAD_IDX).sum().item(); total += loss.item()*nt; tokens += nt
        tl = total / max(tokens, 1)

        model.eval(); vtot, vtok = 0.0, 0
        with torch.no_grad():
            for src, tgt in val:
                src=src.to(device); tgt=tgt.to(device)
                ti=tgt[:,:-1]; tt=tgt[:,1:]
                sm=make_src_mask(src,pad_idx=PAD_IDX).to(device)
                tm=make_tgt_mask(ti, pad_idx=PAD_IDX).to(device)
                out=model(src,ti,sm,tm); lf=out.reshape(-1,out.size(-1)); tf=tt.reshape(-1)
                loss=loss_fn(lf,tf); nt=(tf!=PAD_IDX).sum().item()
                vtot+=loss.item()*nt; vtok+=nt
        vl = vtot / max(vtok, 1)

        vb = 0.0
        if epoch % 5 == 0 or epoch == cfg['num_epochs']-1:
            vb = quick_bleu(model, val, train_ds.tgt_itos, device, max_batches=12)

        wandb.log({"epoch/train_loss": tl, "epoch/val_loss": vl,
                   "epoch/val_bleu": vb, "epoch/lr": optimizer.param_groups[0]['lr'],
                   "epoch": epoch})
        tr_hist.append(tl); va_hist.append(vl); bleu_hist.append(vb)
        print(f"  [{name}] E{epoch:2d} | train {tl:.4f} | val {vl:.4f} | bleu {vb:.2f}", flush=True)

        if vl < best_val:
            best_val = vl
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

    # Full validation BLEU is the primary metric requested in Q2.4.
    class VW:
        def __init__(self, i): self.itos = i
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    val_bleu = evaluate_bleu(model, val_bleu_loader, VW(train_ds.tgt_itos), device=device)
    test_bleu = evaluate_bleu(model, tel, VW(train_ds.tgt_itos), device=device)
    print(f"  [{name}] Validation BLEU: {val_bleu:.2f}", flush=True)
    print(f"  [{name}] Test BLEU: {test_bleu:.2f}", flush=True)
    wandb.log({
        "validation/bleu": val_bleu,
        "test/bleu": test_bleu,
        "test/best_val_loss": best_val,
    })

    # Log PE after training
    mat2 = pe_matrix(model)
    wandb.log({"pe/heatmap_after":    wandb.Image(plot_heatmap(mat2, f'{label} PE - After Training'))})
    wandb.log({"pe/dim_curves_after": wandb.Image(plot_dim_curves(mat2, f'{label} PE Dim Curves - After Training'))})
    wandb.log({"pe/cosine_sim":       wandb.Image(plot_cosine_sim(mat2, f'{label} Position Cosine Similarity'))})
    wandb.log({"pe/extrapolation_view_after": wandb.Image(
        plot_extrapolation_view(
            mat2,
            f'{label} PE Rows After Training: Seen vs Longer Positions',
            train_max_len=length_stats["train/max_src_len"],
        )
    )})
    plt.close('all')

    wandb.finish()
    return tr_hist, va_hist, bleu_hist, val_bleu, test_bleu, mat2


# ======================================================================
#  MAIN
# ======================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    train_ds, trl, val, val_bleu_loader, tel, length_stats = build_data(CFG)
    print("Length stats:", length_stats, flush=True)

    print("\n" + "="*60 + "\nEXP 1: Sinusoidal PE\n" + "="*60)
    str_, sva, sbl, sval_bleu, stest_bleu, smat = run(
        "sinusoidal", CFG, train_ds, trl, val, val_bleu_loader, tel, device, length_stats
    )

    print("\n" + "="*60 + "\nEXP 2: Learned PE\n" + "="*60)
    ltr, lva, lbl, lval_bleu, ltest_bleu, lmat = run(
        "learned", CFG, train_ds, trl, val, val_bleu_loader, tel, device, length_stats
    )

    # Comparison run
    wandb.init(project=WANDB_PROJECT, group=WANDB_GROUP,
               name="2.4_comparison", config={**CFG, **length_stats}, reinit=True)

    # Loss overlay
    data = []
    for e,(t,v) in enumerate(zip(str_, sva)): data.append([e,"sinusoidal",t,v])
    for e,(t,v) in enumerate(zip(ltr,  lva)): data.append([e,"learned",   t,v])
    tbl = wandb.Table(columns=["epoch","condition","train_loss","val_loss"], data=data)
    wandb.log({"cmp/train_loss": wandb.plot.line(tbl,"epoch","train_loss",stroke="condition",title="Train Loss Comparison")})
    wandb.log({"cmp/val_loss":   wandb.plot.line(tbl,"epoch","val_loss",  stroke="condition",title="Val Loss Comparison")})

    # BLEU bars. Validation BLEU is the rubric metric; test BLEU is extra context.
    bt = wandb.Table(columns=["condition","validation_bleu","test_bleu"],
                     data=[["sinusoidal",sval_bleu,stest_bleu],
                           ["learned",lval_bleu,ltest_bleu]])
    wandb.log({
        "cmp/validation_bleu": wandb.plot.bar(
            bt, "condition", "validation_bleu",
            title="Validation BLEU: Sinusoidal vs Learned Positional Encoding"
        ),
        "cmp/test_bleu": wandb.plot.bar(
            bt, "condition", "test_bleu",
            title="Test BLEU: Sinusoidal vs Learned Positional Encoding"
        ),
    })

    # Side-by-side PE heatmaps
    fig, axes = plt.subplots(1,2,figsize=(18,5))
    for ax, mat, t in zip(axes,[smat,lmat],
                          ["Sinusoidal PE (After Training)","Learned PE (After Training)"]):
        im = ax.imshow(mat[:50], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('Dim'); ax.set_ylabel('Position')
        ax.set_title(t, fontweight='bold'); plt.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
    fig.suptitle('PE Matrix Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    wandb.log({"cmp/pe_side_by_side": wandb.Image(fig)})
    plt.close(fig)

    # Side-by-side cosine similarity
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    for ax, mat, t in zip(axes,[smat,lmat],
                          ["Sinusoidal - Cosine Sim","Learned - Cosine Sim"]):
        m   = mat[:30]; n = np.linalg.norm(m,axis=1,keepdims=True)+1e-9
        sim = (m/n)@(m/n).T
        im  = ax.imshow(sim,cmap='Blues',vmin=0,vmax=1)
        ax.set_xlabel('Position'); ax.set_ylabel('Position')
        ax.set_title(t, fontweight='bold'); plt.colorbar(im,ax=ax)
    fig.suptitle('Position Cosine Similarity (first 30 positions)',fontsize=12,fontweight='bold')
    plt.tight_layout()
    wandb.log({"cmp/cosine_side_by_side": wandb.Image(fig)})
    plt.close(fig)

    # Seen-vs-longer-position visualisation for the theoretical extrapolation part.
    fig, axes = plt.subplots(1,2,figsize=(18,5))
    for ax, mat, t in zip(axes,[smat,lmat],
                          ["Sinusoidal PE: formula-defined rows",
                           "Learned PE: rows learned only if observed"]):
        max_pos = min(160, mat.shape[0])
        im = ax.imshow(mat[:max_pos], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.axhline(length_stats["train/max_src_len"] - 0.5,
                   color='black', lw=2, ls='--', label='max train source length')
        ax.set_xlabel('Dim'); ax.set_ylabel('Position')
        ax.set_title(t, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im,ax=ax,fraction=0.03,pad=0.02)
    fig.suptitle('Extrapolation View: Positions Seen During Training vs Longer Positions',
                 fontsize=12,fontweight='bold')
    plt.tight_layout()
    wandb.log({"cmp/extrapolation_side_by_side": wandb.Image(fig)})
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"  Sinusoidal PE - Validation BLEU: {sval_bleu:.2f} | Test BLEU: {stest_bleu:.2f}")
    print(f"  Learned PE    - Validation BLEU: {lval_bleu:.2f} | Test BLEU: {ltest_bleu:.2f}")
    print(f"{'='*60}")
    wandb.finish()


if __name__ == "__main__":
    main()
