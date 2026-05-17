"""
experiment_2_3_attention_heatmaps.py
DA6401 Assignment 3 - Q2.3 missing report artifacts

This script does NOT retrain.
It loads one trained attention-head checkpoint, extracts attention weights from
the LAST encoder layer for one source sentence, logs one heatmap per head, and
logs simple head-role / redundancy metrics for the report.

Example:
    python experiment_2_3_attention_heatmaps.py --checkpoint best_ckpt_2_3_heads_4.pt

Optional:
    python experiment_2_3_attention_heatmaps.py --checkpoint best_ckpt_2_3_heads_8.pt --no-wandb
"""

import argparse
import base64
import html
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from model import (
    PositionalEncoding,
    PositionwiseFeedForward,
    Encoder,
    Decoder,
    make_src_mask,
)


WANDB_PROJECT = "da6401-a3"
WANDB_GROUP = "2.3_attention_rollout_heatmaps"
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
DEFAULT_SOURCE_SENTENCE = "ein mann in einem roten hemd sitzt auf einer bank ."


class InstrumentedMHA(nn.Module):
    """Multi-head attention that stores the latest per-head attention weights."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        self.last_attn_weights = attn.detach()

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)


class InstrumentedEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = InstrumentedMHA(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class InstrumentedDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = InstrumentedMHA(d_model, num_heads, dropout)
        self.cross_attn = InstrumentedMHA(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class InstrumentedTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        enc_layer = InstrumentedEncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = InstrumentedDecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.config = dict(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            N=N,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def load_source_vocab(vocab_checkpoint: Optional[str]) -> Tuple[dict, List[str]]:
    """
    The Q2.3 checkpoints did not save vocabulary mappings, so we borrow the
    source vocabulary from an earlier checkpoint that did save src_stoi.
    This avoids importing dataset.py, which can crash on some Windows pyarrow
    installs before the script prints anything.
    """
    candidates = []
    if vocab_checkpoint:
        candidates.append(Path(vocab_checkpoint))
    candidates.extend([
        Path("best_checkpoint.pt"),
        Path("checkpoint.pt"),
        Path("best_ckpt_noam_scheduler.pt"),
        Path("best_ckpt_fixed_lr.pt"),
    ])

    for path in candidates:
        if not path.exists():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_config", {})
        src_stoi = cfg.get("src_stoi")
        if src_stoi:
            max_idx = max(src_stoi.values())
            src_itos = ["<unk>"] * (max_idx + 1)
            for token, idx in src_stoi.items():
                src_itos[idx] = token
            return src_stoi, src_itos

    raise RuntimeError(
        "Could not find src_stoi in any checkpoint. Pass a checkpoint with "
        "--vocab-checkpoint, for example --vocab-checkpoint best_checkpoint.pt"
    )


def simple_tokenize(text: str) -> List[str]:
    for ch in ".,!?;:":
        text = text.replace(ch, f" {ch} ")
    return text.lower().split()


def encode_source_sentence(sentence: str, src_stoi: dict) -> Tuple[torch.Tensor, List[str]]:
    tokens = ["<sos>"] + simple_tokenize(sentence) + ["<eos>"]
    ids = [src_stoi.get(tok, UNK_IDX) for tok in tokens]
    return torch.tensor(ids, dtype=torch.long), tokens


def ids_to_tokens(ids, itos):
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx == PAD_IDX:
            break
        tokens.append(itos[idx])
    return tokens


def attention_role_scores(attn: torch.Tensor, tokens: List[str]) -> List[dict]:
    """Return simple interpretable scores for each head."""
    num_heads, seq_len, _ = attn.shape
    device = attn.device

    row = torch.arange(seq_len, device=device).unsqueeze(1)
    col = torch.arange(seq_len, device=device).unsqueeze(0)

    diag_mask = (row == col).float()
    next_mask = (col == row + 1).float()
    prev_mask = (col == row - 1).float()
    long_mask = ((row - col).abs() >= 4).float()

    special_cols = torch.tensor(
        [tok in ("<sos>", "<eos>", "<unk>", ".", ",") for tok in tokens],
        device=device,
        dtype=torch.bool,
    )

    rows = []
    for head_idx in range(num_heads):
        a = attn[head_idx]
        entropy = -(a * (a + 1e-9).log()).sum(dim=-1).mean().item()
        avg_max = a.max(dim=-1).values.mean().item()
        diag = (a * diag_mask).sum(dim=-1).mean().item()
        nxt = (a * next_mask).sum(dim=-1).mean().item()
        prev = (a * prev_mask).sum(dim=-1).mean().item()
        long = (a * long_mask).sum(dim=-1).mean().item()
        special = a[:, special_cols].sum(dim=-1).mean().item() if special_cols.any() else 0.0

        role_scores = {
            "self/diagonal": diag,
            "next-token": nxt,
            "previous-token": prev,
            "long-range": long,
            "special/punctuation": special,
        }

        rows.append({
            "head": head_idx,
            "entropy": entropy,
            "avg_max_attention": avg_max,
            "self_diagonal_mass": diag,
            "next_token_mass": nxt,
            "previous_token_mass": prev,
            "long_range_mass": long,
            "special_token_mass": special,
            "predicted_role": max(role_scores, key=role_scores.get),
        })

    return rows


def redundancy_scores(attn: torch.Tensor) -> Tuple[float, List[List[float]]]:
    """Pairwise cosine similarity between flattened attention heatmaps."""
    num_heads = attn.size(0)
    if num_heads < 2:
        return 0.0, []

    flat = attn.reshape(num_heads, -1)
    flat = F.normalize(flat, p=2, dim=1)
    sim = flat @ flat.T

    vals = []
    rows = []
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            val = sim[i, j].item()
            vals.append(val)
            rows.append([i, j, val])
    return sum(vals) / len(vals), rows


def save_and_log_heatmaps(attn, tokens, out_dir, condition_name, use_wandb):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    seq_len = len(tokens)

    for head_idx in range(attn.size(0)):
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * seq_len), max(6, 0.55 * seq_len)))
        im = ax.imshow(attn[head_idx].numpy(), cmap="viridis", vmin=0.0)
        ax.set_title(f"{condition_name}: Last Encoder Layer, Head {head_idx}")
        ax.set_xlabel("Key token attended to")
        ax.set_ylabel("Query token")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        path = out_dir / f"{condition_name}_last_encoder_head_{head_idx}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        image_paths.append(path)

    if use_wandb:
        # Log from saved PNG paths rather than matplotlib figures. This avoids
        # W&B media panels showing "No matching media" on some Windows setups.
        wandb.log({
            f"attention_heatmap/head_{idx}": wandb.Image(
                str(path),
                caption=f"{condition_name}, last encoder layer, head {idx}",
            )
            for idx, path in enumerate(image_paths)
        })

        table = wandb.Table(columns=["head", "heatmap"])
        for idx, path in enumerate(image_paths):
            table.add_data(idx, wandb.Image(str(path)))
        wandb.log({"attention_heatmap/all_heads_table": table})

    return image_paths


def save_and_log_combined_heatmap(attn, tokens, out_dir, condition_name, use_wandb):
    out_dir.mkdir(parents=True, exist_ok=True)
    num_heads = attn.size(0)
    ncols = 2 if num_heads <= 4 else 4
    nrows = math.ceil(num_heads / ncols)
    seq_len = len(tokens)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(10, 0.55 * seq_len * ncols), max(7, 0.55 * seq_len * nrows)),
        squeeze=False,
    )

    vmax = float(attn.max().item())
    for head_idx in range(num_heads):
        ax = axes[head_idx // ncols][head_idx % ncols]
        im = ax.imshow(attn[head_idx].numpy(), cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(f"Head {head_idx}")
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

    for empty_idx in range(num_heads, nrows * ncols):
        axes[empty_idx // ncols][empty_idx % ncols].axis("off")

    fig.suptitle(f"{condition_name}: Last Encoder Layer Attention Heads", y=1.01)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    fig.tight_layout()

    path = out_dir / f"{condition_name}_last_encoder_all_heads.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if use_wandb:
        wandb.log({
            "attention_heatmap/all_heads_grid": wandb.Image(
                str(path),
                caption=f"{condition_name}: all last-encoder-layer heads",
            )
        })

    return path


def write_html_attention_report(
    image_paths,
    combined_path,
    role_rows,
    redundancy_rows,
    avg_similarity,
    specialisation,
    tokens,
    out_dir,
    condition_name,
):
    def image_to_data_uri(path):
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"

    role_trs = []
    for r in role_rows:
        role_trs.append(
            "<tr>"
            f"<td>{r['head']}</td>"
            f"<td>{html.escape(r['predicted_role'])}</td>"
            f"<td>{r['entropy']:.3f}</td>"
            f"<td>{r['avg_max_attention']:.3f}</td>"
            f"<td>{r['self_diagonal_mass']:.3f}</td>"
            f"<td>{r['next_token_mass']:.3f}</td>"
            f"<td>{r['previous_token_mass']:.3f}</td>"
            f"<td>{r['long_range_mass']:.3f}</td>"
            "</tr>"
        )

    red_trs = []
    for i, j, sim in redundancy_rows:
        red_trs.append(f"<tr><td>{i}</td><td>{j}</td><td>{sim:.3f}</td></tr>")

    individual = []
    for idx, path in enumerate(image_paths):
        individual.append(
            "<section>"
            f"<h3>Head {idx}</h3>"
            f"<img src='{image_to_data_uri(path)}' alt='Head {idx} heatmap' />"
            "</section>"
        )

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
body {{
  font-family: Inter, Arial, sans-serif;
  color: #202124;
  margin: 20px;
}}
h1, h2, h3 {{ margin: 0 0 10px; }}
p {{ margin: 6px 0 14px; }}
.summary {{
  display: grid;
  grid-template-columns: repeat(3, minmax(160px, 1fr));
  gap: 12px;
  margin: 14px 0 20px;
}}
.metric {{
  border: 1px solid #ddd;
  padding: 10px 12px;
  border-radius: 6px;
}}
.metric b {{
  display: block;
  font-size: 20px;
  margin-top: 4px;
}}
img {{
  width: 100%;
  max-width: 1200px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: white;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(320px, 1fr));
  gap: 18px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 10px 0 22px;
  font-size: 13px;
}}
th, td {{
  border: 1px solid #ddd;
  padding: 7px 8px;
  text-align: right;
}}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{
  text-align: left;
}}
th {{ background: #f6f8fa; }}
</style>
</head>
<body>
<h1>Q2.3 Last Encoder Layer Attention Heatmaps - {html.escape(condition_name)}</h1>
<p><b>Source sentence:</b> {html.escape(" ".join(tokens))}</p>

<div class="summary">
  <div class="metric">Encoder layer<b>Last layer</b></div>
  <div class="metric">Head specialisation, entropy std<b>{specialisation:.3f}</b></div>
  <div class="metric">Avg pairwise head similarity<b>{avg_similarity:.3f}</b></div>
</div>

<h2>All Heads</h2>
<img src="{image_to_data_uri(combined_path)}" alt="All heads heatmap grid" />

<h2>Head Role Analysis</h2>
<table>
<thead>
<tr>
<th>Head</th><th>Predicted role</th><th>Entropy</th><th>Avg max attn</th>
<th>Self/diag</th><th>Next token</th><th>Previous token</th><th>Long range</th>
</tr>
</thead>
<tbody>
{''.join(role_trs)}
</tbody>
</table>

<h2>Head Redundancy</h2>
<table>
<thead><tr><th>Head i</th><th>Head j</th><th>Cosine similarity</th></tr></thead>
<tbody>{''.join(red_trs)}</tbody>
</table>

<h2>Individual Heads</h2>
<div class="grid">
{''.join(individual)}
</div>
</body>
</html>
"""
    path = out_dir / f"{condition_name}_attention_heatmap_report.html"
    path.write_text(doc, encoding="utf-8")
    return path


def main():
    print("[Q2.3] Script started", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained best_ckpt_2_3_heads_X.pt")
    parser.add_argument(
        "--vocab-checkpoint",
        default=None,
        help="Checkpoint containing src_stoi. Default: auto-detect best_checkpoint.pt/checkpoint.pt.",
    )
    parser.add_argument(
        "--sentence",
        default=DEFAULT_SOURCE_SENTENCE,
        help="Source sentence to feed to the encoder. Use German for this De->En model.",
    )
    parser.add_argument("--max-tokens", type=int, default=18)
    parser.add_argument("--out-dir", default="q2_3_attention_heatmaps")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=WANDB_GROUP)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = Path(args.checkpoint)
    condition_name = checkpoint_path.stem.replace("best_ckpt_2_3_", "")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path.resolve()}")

    print(f"[Q2.3] Device: {device}", flush=True)
    print(f"[Q2.3] Loading checkpoint: {checkpoint_path.resolve()}", flush=True)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})

    print("[Q2.3] Loading source vocabulary from saved checkpoint...", flush=True)
    src_stoi, src_itos = load_source_vocab(args.vocab_checkpoint)
    src_ids, tokens = encode_source_sentence(args.sentence, src_stoi)
    src_ids = src_ids[:args.max_tokens]
    tokens = tokens[:args.max_tokens]
    print(f"[Q2.3] Source sentence: {' '.join(tokens)}", flush=True)

    print("[Q2.3] Building instrumented model...", flush=True)
    model = InstrumentedTransformer(
        src_vocab_size=cfg.get("src_vocab_size"),
        tgt_vocab_size=cfg.get("tgt_vocab_size"),
        d_model=cfg.get("d_model", 256),
        N=cfg.get("N", 3),
        num_heads=cfg.get("num_heads", 4),
        d_ff=cfg.get("d_ff", 512),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("[Q2.3] Checkpoint loaded into instrumented model.", flush=True)

    src = src_ids.unsqueeze(0).to(device)
    src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)

    use_wandb = not args.no_wandb
    run = None
    if use_wandb:
        os.environ.setdefault("WANDB_MODE", "online")
        print("[Q2.3] Initialising W&B run...", flush=True)
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"{condition_name}_last_encoder_heatmaps",
            config={
                "checkpoint": str(checkpoint_path),
                "condition": condition_name,
                "num_heads": model.config["num_heads"],
                "d_model": model.config["d_model"],
                "d_k": model.config["d_model"] // model.config["num_heads"],
                "sentence": " ".join(tokens),
            },
        )
        print(f"[Q2.3] W&B run URL: {run.url}", flush=True)

    print("[Q2.3] Running one encoder forward pass...", flush=True)
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    last_layer_idx = len(model.encoder.layers) - 1
    attn = model.encoder.layers[last_layer_idx].self_attn.last_attn_weights
    if attn is None:
        raise RuntimeError("No attention weights were captured.")

    attn = attn[0].detach().cpu()
    seq_len = len(tokens)
    attn = attn[:, :seq_len, :seq_len]

    out_dir = Path(args.out_dir) / condition_name
    print("[Q2.3] Saving/logging heatmaps...", flush=True)
    image_paths = save_and_log_heatmaps(attn, tokens, out_dir, condition_name, use_wandb)
    combined_path = save_and_log_combined_heatmap(attn, tokens, out_dir, condition_name, use_wandb)

    role_rows = attention_role_scores(attn, tokens)
    avg_similarity, redundancy_rows = redundancy_scores(attn)
    entropy_values = [r["entropy"] for r in role_rows]
    specialisation = torch.tensor(entropy_values).std().item() if len(entropy_values) > 1 else 0.0
    html_report_path = write_html_attention_report(
        image_paths=image_paths,
        combined_path=combined_path,
        role_rows=role_rows,
        redundancy_rows=redundancy_rows,
        avg_similarity=avg_similarity,
        specialisation=specialisation,
        tokens=tokens,
        out_dir=out_dir,
        condition_name=condition_name,
    )

    if use_wandb:
        role_table = wandb.Table(
            columns=[
                "head",
                "entropy",
                "avg_max_attention",
                "self_diagonal_mass",
                "next_token_mass",
                "previous_token_mass",
                "long_range_mass",
                "special_token_mass",
                "predicted_role",
            ],
            data=[
                [
                    r["head"],
                    r["entropy"],
                    r["avg_max_attention"],
                    r["self_diagonal_mass"],
                    r["next_token_mass"],
                    r["previous_token_mass"],
                    r["long_range_mass"],
                    r["special_token_mass"],
                    r["predicted_role"],
                ]
                for r in role_rows
            ],
        )
        redundancy_table = wandb.Table(
            columns=["head_i", "head_j", "cosine_similarity"],
            data=redundancy_rows,
        )
        wandb.log({
            "attention/example_source_sentence": " ".join(tokens),
            "attention/last_encoder_layer": last_layer_idx,
            "attention_analysis/head_role_table": role_table,
            "attention_analysis/pairwise_head_redundancy": redundancy_table,
            "attention_analysis/avg_pairwise_head_similarity": avg_similarity,
            "attention_analysis/head_specialisation_entropy_std": specialisation,
            "attention_heatmap/html_report": wandb.Html(html_report_path),
        })
        print("[Q2.3] Finishing W&B run...", flush=True)
        wandb.finish()

    print("\nQ2.3 attention heatmaps generated")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Sentence: {' '.join(tokens)}")
    print(f"Last encoder layer: {last_layer_idx}")
    print(f"Saved heatmaps to: {out_dir.resolve()}")
    for path in image_paths:
        print(f"  - {path}")
    print(f"  - {combined_path}")
    print(f"  - {html_report_path}")
    print("\nHead role summary:")
    for r in role_rows:
        print(
            f"  Head {r['head']}: role={r['predicted_role']}, "
            f"entropy={r['entropy']:.3f}, max={r['avg_max_attention']:.3f}, "
            f"diag={r['self_diagonal_mass']:.3f}, next={r['next_token_mass']:.3f}, "
            f"prev={r['previous_token_mass']:.3f}, long={r['long_range_mass']:.3f}"
        )
    print(f"\nHead specialisation entropy std: {specialisation:.3f}")
    print(f"Average pairwise head cosine similarity: {avg_similarity:.3f}")
    print("Higher pairwise similarity means more head redundancy.")


if __name__ == "__main__":
    main()
