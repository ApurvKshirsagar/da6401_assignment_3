"""
experiment_2_3_attention_heads.py
DA6401 Assignment 3 — W&B Report: Question 2.3
Ablation: The Role of Multiple Attention Heads

Trains the same Transformer under four different head counts,
keeping d_model fixed so each head's d_k = d_model / num_heads.

Conditions (all share d_model=256, so d_k varies):
    1. num_heads = 1   → d_k = 256  (single large head)
    2. num_heads = 2   → d_k = 128
    3. num_heads = 4   → d_k = 64
    4. num_heads = 8   → d_k = 32   (paper default for d_model=256)

Key metrics logged to W&B:
  Per epoch:
    - train loss, val loss, learning rate
  Per step (first grad_log_steps steps):
    - per-head attention entropy in encoder layer 0
      (measures whether heads specialise or stay uniform)
    - W_q / W_k gradient norms (encoder layer 0)
  Final:
    - Test BLEU
    - Head-specialisation score (std-dev of per-head entropy across heads)
    - Comparison overlay (all four conditions on same axes)

Usage:
    python experiment_2_3_attention_heads.py
"""

import math
import copy
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from dataset import (
    Multi30kDataset, get_dataloader,
    PAD_IDX, SOS_IDX, EOS_IDX,
)
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, save_checkpoint, evaluate_bleu
from model import (
    PositionalEncoding, PositionwiseFeedForward,
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    make_src_mask, make_tgt_mask,
)


# ══════════════════════════════════════════════════════════════════════
#  INSTRUMENTED MHA — stores per-head attention weights for analysis
# ══════════════════════════════════════════════════════════════════════

class InstrumentedMHA(nn.Module):
    """
    MultiHeadAttention that caches last attention weights per head.
    Allows downstream entropy and specialisation analysis.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # Shape after forward: [B, num_heads, seq_q, seq_k]
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_w = F.softmax(scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)

        # Cache for analysis — detached to avoid holding computation graph
        self.last_attn_weights = attn_w.detach()   # [B, H, Sq, Sk]

        out = torch.matmul(attn_w, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════════
#  INSTRUMENTED ENCODER / DECODER LAYERS
# ══════════════════════════════════════════════════════════════════════

class InstrumentedEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = InstrumentedMHA(d_model, num_heads, dropout)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class InstrumentedDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = InstrumentedMHA(d_model, num_heads, dropout)
        self.cross_attn = InstrumentedMHA(d_model, num_heads, dropout)
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
#  INSTRUMENTED TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class InstrumentedTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout)

        enc_layer    = InstrumentedEncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = InstrumentedDecoderLayer(d_model, num_heads, d_ff, dropout)
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

    # ── Analysis helpers ──────────────────────────────────────────────

    def get_per_head_entropy(self, layer_idx: int = 0) -> List[float]:
        """
        Returns per-head average attention entropy for encoder layer `layer_idx`.
        Shape of last_attn_weights: [B, H, Sq, Sk]

        Entropy = -sum_k p(k) * log(p(k)+eps)  averaged over B and Sq.
        Returns a list of length num_heads.
        """
        attn_w = self.encoder.layers[layer_idx].self_attn.last_attn_weights
        if attn_w is None:
            return []
        eps = 1e-9
        # entropy per (B, H, Sq) position
        ent = -(attn_w * (attn_w + eps).log()).sum(dim=-1)  # [B, H, Sq]
        # average over batch and query positions → [H]
        per_head = ent.mean(dim=(0, 2))                     # [H]
        return per_head.tolist()

    def get_head_specialisation(self, layer_idx: int = 0) -> float:
        """
        Standard deviation of per-head entropy across heads.
        High std → heads are specialised (different focus patterns).
        Low std  → heads are redundant (similar patterns).
        """
        entropies = self.get_per_head_entropy(layer_idx)
        if len(entropies) < 2:
            return 0.0
        t = torch.tensor(entropies)
        return t.std().item()

    def get_qk_grad_norms_enc0(self) -> dict:
        """Gradient norms of W_q and W_k for encoder layer 0 self-attention."""
        layer = self.encoder.layers[0]
        norms = {}
        wq_g = layer.self_attn.W_q.weight.grad
        wk_g = layer.self_attn.W_k.weight.grad
        if wq_g is not None:
            norms["grad_norm/enc0_Wq"] = wq_g.norm().item()
            norms["grad_norm/enc0_Wk"] = wk_g.norm().item()
        return norms


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

HEAD_CONFIGS = [1, 2, 4, 8]    # num_heads to ablate

BASE_CONFIG = dict(
    d_model        = 256,       # fixed; d_k = d_model / num_heads
    N              = 3,
    d_ff           = 512,
    dropout        = 0.1,
    warmup_steps   = 2000,
    batch_size     = 128,
    num_epochs     = 30,
    smoothing      = 0.1,
    min_freq       = 2,
    grad_log_steps = 1000,      # log grad norms / entropy for first N steps
)

WANDB_PROJECT = "da6401-a3"
WANDB_GROUP   = "2.3_attention_heads"


# ══════════════════════════════════════════════════════════════════════
#  DATA
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
#  SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_experiment(
    num_heads: int,
    cfg: dict,
    train_ds,
    train_loader,
    val_loader,
    test_loader,
    device: str,
):
    condition_name = f"heads_{num_heads}"
    d_k = cfg['d_model'] // num_heads

    wandb.init(
        project = WANDB_PROJECT,
        group   = WANDB_GROUP,
        name    = condition_name,
        config  = {
            **cfg,
            "num_heads":       num_heads,
            "d_k":             d_k,
            "condition":       condition_name,
        },
        reinit = True,
    )

    src_vocab_size = len(train_ds.src_itos)
    tgt_vocab_size = len(train_ds.tgt_itos)

    model = InstrumentedTransformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model   = cfg['d_model'],
        N         = cfg['N'],
        num_heads = num_heads,
        d_ff      = cfg['d_ff'],
        dropout   = cfg['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{condition_name}] d_k={d_k} | params: {n_params:,}")

    loss_fn   = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=cfg['smoothing'])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps']
    )

    best_val_loss  = float('inf')
    best_ckpt      = f"best_ckpt_2_3_{condition_name}.pt"
    global_step    = 0

    train_loss_history = []
    val_loss_history   = []

    # Accumulate per-head entropy traces for summary table
    entropy_trace = []          # list of dicts: step → entropy per head

    for epoch in range(cfg['num_epochs']):

        # ── TRAIN ────────────────────────────────────────────────────
        model.train()
        total_loss, total_tokens = 0.0, 0

        pbar = tqdm(
            train_loader,
            desc=f"[{condition_name}] Epoch {epoch} [train]",
            leave=False,
        )

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

            # ── Gradient norms (first grad_log_steps steps) ───────────
            if global_step < cfg['grad_log_steps']:
                qk_norms = model.get_qk_grad_norms_enc0()
                total_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                wandb.log({
                    "global_step":     global_step,
                    "grad_norm/total": total_grad_norm,
                    **qk_norms,
                })

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # ── Per-head entropy (every step) ─────────────────────────
            with torch.no_grad():
                per_head_ent = model.get_per_head_entropy(layer_idx=0)
                specialisation = model.get_head_specialisation(layer_idx=0)

            avg_entropy = sum(per_head_ent) / max(len(per_head_ent), 1)

            step_log = {
                "step/train_loss":       loss.item(),
                "step/avg_attn_entropy": avg_entropy,
                "step/head_specialisation": specialisation,
                "step/learning_rate":    optimizer.param_groups[0]['lr'],
                "global_step":           global_step,
            }
            # Log individual head entropies
            for h_idx, h_ent in enumerate(per_head_ent):
                step_log[f"step/head_{h_idx}_entropy"] = h_ent

            wandb.log(step_log)

            if global_step < cfg['grad_log_steps']:
                entropy_trace.append({"step": global_step, **{
                    f"head_{h_idx}": h_ent
                    for h_idx, h_ent in enumerate(per_head_ent)
                }})

            n_tokens    = (target_flat != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            global_step  += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", spec=f"{specialisation:.3f}")

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

        print(
            f"  [{condition_name}] Epoch {epoch:3d} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | lr {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

    # ── BLEU ──────────────────────────────────────────────────────────
    class VocabWrapper:
        def __init__(self, itos): self.itos = itos

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    bleu = evaluate_bleu(
        model, test_loader, VocabWrapper(train_ds.tgt_itos), device=device
    )
    print(f"  [{condition_name}] Test BLEU: {bleu:.2f}")

    # Final specialisation: run one validation batch through the loaded model
    with torch.no_grad():
        src, tgt = next(iter(val_loader))
        src = src.to(device); tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        src_mask  = make_src_mask(src, pad_idx=PAD_IDX).to(device)
        tgt_mask  = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)
        _ = model(src, tgt_input, src_mask, tgt_mask)
        final_specialisation = model.get_head_specialisation(layer_idx=0)
        final_per_head_ent   = model.get_per_head_entropy(layer_idx=0)

    wandb.log({
        "test/bleu":                  bleu,
        "test/best_val_loss":         best_val_loss,
        "test/head_specialisation":   final_specialisation,
    })
    for h_idx, h_ent in enumerate(final_per_head_ent):
        wandb.log({f"test/head_{h_idx}_entropy": h_ent})

    # ── Log entropy-trace summary table (first grad_log_steps steps) ──
    if entropy_trace and num_heads > 1:
        cols = ["step"] + [f"head_{h}" for h in range(num_heads)]
        rows = [[d["step"]] + [d.get(f"head_{h}", 0.0) for h in range(num_heads)]
                for d in entropy_trace]
        ent_table = wandb.Table(columns=cols, data=rows)

        # Plot per-head entropy for head 0 and head 1 (representative)
        wandb.log({
            "entropy_trace/head0": wandb.plot.line(
                ent_table, "step", "head_0",
                title=f"Head-0 Entropy — {condition_name} (first {cfg['grad_log_steps']} steps)"
            ),
        })
        if num_heads >= 2:
            wandb.log({
                "entropy_trace/head1": wandb.plot.line(
                    ent_table, "step", "head_1",
                    title=f"Head-1 Entropy — {condition_name} (first {cfg['grad_log_steps']} steps)"
                ),
            })

    wandb.finish()
    return train_loss_history, val_loss_history, bleu, final_specialisation


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    cfg = BASE_CONFIG.copy()
    train_ds, train_loader, val_loader, test_loader = build_data(cfg)
    print(f"  src vocab: {len(train_ds.src_itos)}, tgt vocab: {len(train_ds.tgt_itos)}")

    results = {}   # num_heads → (train_hist, val_hist, bleu, specialisation)

    for num_heads in HEAD_CONFIGS:
        print("\n" + "=" * 60)
        print(f"EXPERIMENT: num_heads = {num_heads}  (d_k = {cfg['d_model'] // num_heads})")
        print("=" * 60)
        train_hist, val_hist, bleu, spec = run_experiment(
            num_heads    = num_heads,
            cfg          = cfg,
            train_ds     = train_ds,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            device       = device,
        )
        results[num_heads] = (train_hist, val_hist, bleu, spec)

    # ── Combined comparison overlay run ───────────────────────────────
    print("\n" + "=" * 60)
    print("Logging combined comparison run...")
    print("=" * 60)

    wandb.init(
        project = WANDB_PROJECT,
        group   = WANDB_GROUP,
        name    = "2.3_comparison_overlay",
        config  = cfg,
        reinit  = True,
    )

    # Training / val loss overlay
    comparison_data = []
    for num_heads, (train_hist, val_hist, bleu, spec) in results.items():
        cond = f"heads_{num_heads}"
        for e in range(len(train_hist)):
            comparison_data.append([e, cond, train_hist[e], val_hist[e]])

    comp_table = wandb.Table(
        columns=["epoch", "condition", "train_loss", "val_loss"],
        data=comparison_data,
    )
    wandb.log({
        "comparison/train_loss_overlay": wandb.plot.line(
            comp_table, "epoch", "train_loss", stroke="condition",
            title="Train Loss by Number of Heads"
        ),
        "comparison/val_loss_overlay": wandb.plot.line(
            comp_table, "epoch", "val_loss", stroke="condition",
            title="Val Loss by Number of Heads"
        ),
    })

    # BLEU bar chart
    bleu_rows = [[f"heads_{h}", results[h][2]] for h in HEAD_CONFIGS]
    bleu_table = wandb.Table(columns=["condition", "test_bleu"], data=bleu_rows)
    wandb.log({
        "comparison/bleu_bar": wandb.plot.bar(
            bleu_table, "condition", "test_bleu",
            title="Test BLEU by Number of Heads"
        ),
    })

    # Head-specialisation bar chart
    spec_rows = [[f"heads_{h}", results[h][3]] for h in HEAD_CONFIGS]
    spec_table = wandb.Table(columns=["condition", "head_specialisation"], data=spec_rows)
    wandb.log({
        "comparison/specialisation_bar": wandb.plot.bar(
            spec_table, "condition", "head_specialisation",
            title="Head Specialisation (entropy std-dev) by Number of Heads"
        ),
    })

    # BLEU vs num_heads scatter
    scatter_rows = [[h, results[h][2], results[h][3]] for h in HEAD_CONFIGS]
    scatter_table = wandb.Table(
        columns=["num_heads", "test_bleu", "head_specialisation"],
        data=scatter_rows,
    )
    wandb.log({
        "comparison/bleu_vs_heads": wandb.plot.scatter(
            scatter_table, "num_heads", "test_bleu",
            title="Test BLEU vs Number of Heads"
        ),
    })

    print(f"\n{'=' * 60}")
    print(f"  {'Heads':>8} | {'d_k':>5} | {'Test BLEU':>10} | {'Specialisation':>16}")
    print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*10}-+-{'-'*16}")
    for h in HEAD_CONFIGS:
        d_k = cfg['d_model'] // h
        bleu = results[h][2]
        spec = results[h][3]
        print(f"  {h:>8} | {d_k:>5} | {bleu:>10.2f} | {spec:>16.4f}")
    print(f"{'=' * 60}")

    wandb.finish()


if __name__ == "__main__":
    main()