"""
experiment_2_2_scaling_factor.py
DA6401 Assignment 3 — W&B Report: Question 2.2
Ablation: The Scaling Factor 1/sqrt(dk)

Trains two Transformers:
  1. WITH scaling    : scores = Q·Kᵀ / sqrt(dk)   [standard]
  2. WITHOUT scaling : scores = Q·Kᵀ               [ablation]

Key logging:
  - Gradient norms of W_q and W_k in every encoder/decoder layer
    for the first 1000 steps  (per-step)
  - Training loss and val loss per epoch
  - Attention entropy per layer (measures softmax sharpness)
  - Final test BLEU

Usage:
    python experiment_2_2_scaling_factor.py
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

# ── import your existing modules ────────────────────────────────────────
from dataset import (
    Multi30kDataset, get_dataloader,
    PAD_IDX, SOS_IDX, EOS_IDX,
)
from lr_scheduler import NoamScheduler
from train import (
    LabelSmoothingLoss, save_checkpoint,
    evaluate_bleu, greedy_decode,
)
from model import (
    PositionalEncoding, PositionwiseFeedForward,
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    make_src_mask, make_tgt_mask,
)


# ══════════════════════════════════════════════════════════════════════
#  PATCHED ATTENTION — toggle the 1/sqrt(dk) scaling
# ══════════════════════════════════════════════════════════════════════

def attention_with_flag(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled (or unscaled) dot-product attention.
    use_scale=True  → divide by sqrt(dk)   [paper default]
    use_scale=False → raw dot products      [ablation]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scale:
        scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  PATCHED MHA — stores last attention weights for entropy logging
# ══════════════════════════════════════════════════════════════════════

class PatchedMHA(nn.Module):
    """MultiHeadAttention with configurable scaling and weight storage."""

    def __init__(self, d_model: int, num_heads: int,
                 dropout: float = 0.1, use_scale: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.use_scale = use_scale

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # will hold last forward's attention weights: [B, H, Sq, Sk]
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        out, attn_w = attention_with_flag(Q, K, V, mask, use_scale=self.use_scale)
        self.last_attn_weights = attn_w.detach()   # store for entropy

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  PATCHED ENCODER / DECODER LAYERS
# ══════════════════════════════════════════════════════════════════════

class PatchedEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scale=True):
        super().__init__()
        self.self_attn = PatchedMHA(d_model, num_heads, dropout, use_scale)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class PatchedDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_scale=True):
        super().__init__()
        self.self_attn  = PatchedMHA(d_model, num_heads, dropout, use_scale)
        self.cross_attn = PatchedMHA(d_model, num_heads, dropout, use_scale)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  PATCHED TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class PatchedTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, N=3, num_heads=8, d_ff=512,
                 dropout=0.1, use_scale=True):
        super().__init__()
        self.d_model   = d_model
        self.use_scale = use_scale

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        enc_layer    = PatchedEncoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        dec_layer    = PatchedDecoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        self.config = dict(
            src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
            d_model=d_model, N=N, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    # ── helpers to collect W_q / W_k grad norms ──────────────────────
    def get_qk_grad_norms(self):
        """
        Returns dict:
          enc_layer_{i}/W_q_grad_norm, enc_layer_{i}/W_k_grad_norm
          dec_layer_{i}/W_q_grad_norm, dec_layer_{i}/W_k_grad_norm
        """
        norms = {}
        for i, layer in enumerate(self.encoder.layers):
            wq_grad = layer.self_attn.W_q.weight.grad
            wk_grad = layer.self_attn.W_k.weight.grad
            if wq_grad is not None:
                norms[f"grad_norm/enc_layer{i}_Wq"] = wq_grad.norm().item()
                norms[f"grad_norm/enc_layer{i}_Wk"] = wk_grad.norm().item()
        for i, layer in enumerate(self.decoder.layers):
            wq_grad = layer.self_attn.W_q.weight.grad
            wk_grad = layer.self_attn.W_k.weight.grad
            if wq_grad is not None:
                norms[f"grad_norm/dec_layer{i}_self_Wq"] = wq_grad.norm().item()
                norms[f"grad_norm/dec_layer{i}_self_Wk"] = wk_grad.norm().item()
            wq_grad = layer.cross_attn.W_q.weight.grad
            wk_grad = layer.cross_attn.W_k.weight.grad
            if wq_grad is not None:
                norms[f"grad_norm/dec_layer{i}_cross_Wq"] = wq_grad.norm().item()
                norms[f"grad_norm/dec_layer{i}_cross_Wk"] = wk_grad.norm().item()
        return norms

    def get_avg_attention_entropy(self):
        """
        Compute average attention entropy across all heads and encoder layers.
        High entropy = uniform (unfocused) attention.
        Low entropy  = peaked (focused) attention.
        """
        entropies = []
        for layer in self.encoder.layers:
            attn_w = layer.self_attn.last_attn_weights  # [B, H, Sq, Sk]
            if attn_w is None:
                continue
            # entropy per (B, H, Sq) position: -sum(p * log(p+eps))
            eps = 1e-9
            ent = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # [B,H,Sq]
            entropies.append(ent.mean().item())
        return sum(entropies) / len(entropies) if entropies else 0.0


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

BASE_CONFIG = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    warmup_steps = 2000,
    batch_size   = 128,
    num_epochs   = 30,
    smoothing    = 0.1,
    min_freq     = 2,
    grad_log_steps = 1000,  # log grad norms for first N steps
)

WANDB_PROJECT = "da6401-a3"
WANDB_GROUP   = "2.2_scaling_factor"


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def build_data(cfg):
    print("Building datasets...")
    train_ds = Multi30kDataset(split='train')
    train_ds.build_vocab(min_freq=cfg['min_freq'])

    val_ds  = Multi30kDataset(split='validation')
    val_ds.src_stoi = train_ds.src_stoi; val_ds.src_itos = train_ds.src_itos
    val_ds.tgt_stoi = train_ds.tgt_stoi; val_ds.tgt_itos = train_ds.tgt_itos

    test_ds = Multi30kDataset(split='test')
    test_ds.src_stoi = train_ds.src_stoi; test_ds.src_itos = train_ds.src_itos
    test_ds.tgt_stoi = train_ds.tgt_stoi; test_ds.tgt_itos = train_ds.tgt_itos

    train_ds.process_data()
    val_ds.process_data()
    test_ds.process_data()

    train_loader = get_dataloader('train',      train_ds, batch_size=cfg['batch_size'], shuffle=True)
    val_loader   = get_dataloader('validation', val_ds,   batch_size=cfg['batch_size'], shuffle=False)
    test_loader  = get_dataloader('test',       test_ds,  batch_size=1,                shuffle=False)

    return train_ds, train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_experiment(condition_name, use_scale, cfg,
                   train_ds, train_loader, val_loader, test_loader, device):

    wandb.init(
        project = WANDB_PROJECT,
        group   = WANDB_GROUP,
        name    = condition_name,
        config  = {**cfg, "use_scale": use_scale, "condition": condition_name},
        reinit  = True,
    )

    src_vocab_size = len(train_ds.src_itos)
    tgt_vocab_size = len(train_ds.tgt_itos)

    model = PatchedTransformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model   = cfg['d_model'],
        N         = cfg['N'],
        num_heads = cfg['num_heads'],
        d_ff      = cfg['d_ff'],
        dropout   = cfg['dropout'],
        use_scale = use_scale,
    ).to(device)

    loss_fn   = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=cfg['smoothing'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])

    best_val_loss = float('inf')
    best_ckpt     = f"best_ckpt_2_2_{condition_name}.pt"
    global_step   = 0

    train_loss_history = []
    val_loss_history   = []

    # ── accumulate grad norms for the first 1000 steps (for W&B table) ─
    grad_norm_log = []   # list of dicts

    for epoch in range(cfg['num_epochs']):

        # ── TRAIN ────────────────────────────────────────────────────
        model.train()
        total_loss, total_tokens = 0.0, 0

        pbar = tqdm(train_loader,
                    desc=f"[{condition_name}] Epoch {epoch} [train]",
                    leave=False)

        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

            logits      = model(src, tgt_input, src_mask, tgt_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = tgt_target.reshape(-1)

            loss = loss_fn(logits_flat, target_flat)

            optimizer.zero_grad()
            loss.backward()

            # ── Gradient norm logging (first 1000 steps) ─────────────
            if global_step < cfg['grad_log_steps']:
                qk_norms = model.get_qk_grad_norms()

                # also log total grad norm
                total_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5

                step_log = {
                    "global_step":           global_step,
                    "grad_norm/total":       total_grad_norm,
                    **qk_norms,
                }
                grad_norm_log.append(step_log)
                wandb.log(step_log)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # ── Attention entropy (every step, always) ────────────────
            with torch.no_grad():
                attn_entropy = model.get_avg_attention_entropy()

            n_tokens    = (target_flat != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            global_step  += 1

            wandb.log({
                "step/train_loss":      loss.item(),
                "step/attn_entropy":    attn_entropy,
                "step/learning_rate":   optimizer.param_groups[0]['lr'],
                "global_step":          global_step,
            })

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_tokens, 1)

        # ── VALIDATION ───────────────────────────────────────────────
        model.eval()
        val_total_loss, val_total_tokens = 0.0, 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_input  = tgt[:, :-1]
                tgt_target = tgt[:, 1:]

                src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
                tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

                logits      = model(src, tgt_input, src_mask, tgt_mask)
                logits_flat = logits.reshape(-1, logits.size(-1))
                target_flat = tgt_target.reshape(-1)

                loss     = loss_fn(logits_flat, target_flat)
                n_tokens = (target_flat != PAD_IDX).sum().item()
                val_total_loss  += loss.item() * n_tokens
                val_total_tokens += n_tokens

        val_loss   = val_total_loss / max(val_total_tokens, 1)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch/train_loss": train_loss,
            "epoch/val_loss":   val_loss,
            "epoch/lr":         current_lr,
            "epoch":            epoch,
        })

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"  [{condition_name}] Epoch {epoch:3d} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

    # ── BLEU ─────────────────────────────────────────────────────────
    class VocabWrapper:
        def __init__(self, itos): self.itos = itos

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    bleu = evaluate_bleu(model, test_loader, VocabWrapper(train_ds.tgt_itos), device=device)
    print(f"  [{condition_name}] Test BLEU: {bleu:.2f}")

    wandb.log({
        "test/bleu":          bleu,
        "test/best_val_loss": best_val_loss,
    })

    # ── Log grad norm summary table ───────────────────────────────────
    # Average Wq and Wk grad norms across all logged steps, per layer
    if grad_norm_log:
        # Collect all unique keys
        all_keys = set()
        for d in grad_norm_log:
            all_keys.update(d.keys())
        all_keys -= {"global_step", "grad_norm/total"}

        # Build a per-step summary table for enc_layer0 (most representative)
        enc0_wq_key = "grad_norm/enc_layer0_Wq"
        enc0_wk_key = "grad_norm/enc_layer0_Wk"
        summary_rows = [
            [d["global_step"], d.get(enc0_wq_key, 0), d.get(enc0_wk_key, 0),
             d.get("grad_norm/total", 0)]
            for d in grad_norm_log
        ]
        grad_table = wandb.Table(
            columns=["step", "enc_layer0_Wq_norm", "enc_layer0_Wk_norm", "total_grad_norm"],
            data=summary_rows,
        )
        wandb.log({
            "grad_norm/enc_layer0_summary_table": grad_table,
            "grad_norm/enc_layer0_Wq_plot": wandb.plot.line(
                grad_table, "step", "enc_layer0_Wq_norm",
                title=f"W_q Grad Norm — Enc Layer 0 [{condition_name}] (first 1000 steps)"
            ),
            "grad_norm/enc_layer0_Wk_plot": wandb.plot.line(
                grad_table, "step", "enc_layer0_Wk_norm",
                title=f"W_k Grad Norm — Enc Layer 0 [{condition_name}] (first 1000 steps)"
            ),
        })

    wandb.finish()
    return train_loss_history, val_loss_history, bleu


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    cfg = BASE_CONFIG.copy()
    train_ds, train_loader, val_loader, test_loader = build_data(cfg)
    print(f"  src vocab: {len(train_ds.src_itos)}, tgt vocab: {len(train_ds.tgt_itos)}")

    # ── Experiment 1: WITH scaling ─────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 1: WITH scaling factor 1/sqrt(dk)")
    print("="*60)
    scaled_train, scaled_val, scaled_bleu = run_experiment(
        condition_name = "with_scaling",
        use_scale      = True,
        cfg            = cfg,
        train_ds       = train_ds,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        device         = device,
    )

    # ── Experiment 2: WITHOUT scaling ─────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 2: WITHOUT scaling factor (raw dot products)")
    print("="*60)
    unscaled_train, unscaled_val, unscaled_bleu = run_experiment(
        condition_name = "without_scaling",
        use_scale      = False,
        cfg            = cfg,
        train_ds       = train_ds,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        device         = device,
    )

    # ── Combined comparison overlay run ───────────────────────────────
    print("\n" + "="*60)
    print("Logging combined comparison run...")
    print("="*60)

    wandb.init(
        project = WANDB_PROJECT,
        group   = WANDB_GROUP,
        name    = "2.2_comparison_overlay",
        config  = cfg,
        reinit  = True,
    )

    comparison_data = []
    for e in range(len(scaled_train)):
        comparison_data.append([e, "with_scaling",    scaled_train[e],   scaled_val[e]])
    for e in range(len(unscaled_train)):
        comparison_data.append([e, "without_scaling", unscaled_train[e], unscaled_val[e]])

    comp_table = wandb.Table(
        columns=["epoch", "condition", "train_loss", "val_loss"],
        data=comparison_data,
    )
    wandb.log({
        "comparison/train_loss_overlay": wandb.plot.line(
            comp_table, "epoch", "train_loss", stroke="condition",
            title="Train Loss: With vs Without Scaling"
        ),
        "comparison/val_loss_overlay": wandb.plot.line(
            comp_table, "epoch", "val_loss", stroke="condition",
            title="Val Loss: With vs Without Scaling"
        ),
    })

    bleu_table = wandb.Table(
        columns=["condition", "test_bleu"],
        data=[
            ["with_scaling",    scaled_bleu],
            ["without_scaling", unscaled_bleu],
        ],
    )
    wandb.log({
        "comparison/bleu_bar": wandb.plot.bar(
            bleu_table, "condition", "test_bleu",
            title="Test BLEU: With vs Without Scaling"
        ),
        "with_scaling/test_bleu":    scaled_bleu,
        "without_scaling/test_bleu": unscaled_bleu,
    })

    print(f"\n{'='*60}")
    print(f"  With scaling    — Test BLEU: {scaled_bleu:.2f}")
    print(f"  Without scaling — Test BLEU: {unscaled_bleu:.2f}")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    main()
