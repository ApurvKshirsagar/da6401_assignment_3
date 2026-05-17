"""
experiment_2_1_noam_vs_fixed_lr.py
DA6401 Assignment 3 — W&B Report: Question 2.1
The Necessity of the Noam Scheduler

Trains the Transformer under two conditions:
  1. Noam Scheduler  (linear warmup + inverse sqrt decay)
  2. Fixed LR = 1e-4 (constant, no warmup)

Logs to W&B with overlaid training/val curves and an LR schedule plot.

Usage:
    python experiment_2_1_noam_vs_fixed_lr.py
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    Multi30kDataset, TranslationDataset, get_dataloader,
    PAD_IDX, SOS_IDX, EOS_IDX,
)
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, run_epoch, evaluate_bleu, save_checkpoint


# ─────────────────────────────────────────────────────────────────────
#  Shared hyperparameters
# ─────────────────────────────────────────────────────────────────────
BASE_CONFIG = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    warmup_steps = 2000,
    batch_size   = 128,
    num_epochs   = 30,          # enough to see divergence / convergence clearly
    smoothing    = 0.1,
    min_freq     = 2,
)

FIXED_LR    = 1e-4
WANDB_PROJECT = "da6401-a3"
WANDB_GROUP   = "2.1_noam_scheduler"


# ─────────────────────────────────────────────────────────────────────
#  Build shared data (only loaded once)
# ─────────────────────────────────────────────────────────────────────

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

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────
#  Build a fresh Transformer
# ─────────────────────────────────────────────────────────────────────

def build_model(cfg, src_vocab_size, tgt_vocab_size, device):
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model   = cfg['d_model'],
        N         = cfg['N'],
        num_heads = cfg['num_heads'],
        d_ff      = cfg['d_ff'],
        dropout   = cfg['dropout'],
    ).to(device)
    return model


# ─────────────────────────────────────────────────────────────────────
#  Core training function — logs every step + every epoch to W&B
# ─────────────────────────────────────────────────────────────────────

def run_one_experiment(
    condition_name: str,   # "noam_scheduler" or "fixed_lr"
    use_noam: bool,
    cfg: dict,
    train_ds,
    train_loader,
    val_loader,
    test_loader,
    device: str,
):
    """Train for `cfg['num_epochs']` epochs and log everything to W&B."""

    wandb.init(
        project = WANDB_PROJECT,
        group   = WANDB_GROUP,
        name    = condition_name,
        config  = {**cfg, "condition": condition_name, "use_noam": use_noam},
        reinit  = True,
    )

    src_vocab_size = len(train_ds.src_itos)
    tgt_vocab_size = len(train_ds.tgt_itos)

    model = build_model(cfg, src_vocab_size, tgt_vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [{condition_name}] params: {n_params:,}")

    # Store vocab for checkpoint compatibility
    model.config['src_stoi'] = train_ds.src_stoi
    model.config['tgt_stoi'] = train_ds.tgt_stoi
    model.config['tgt_itos'] = train_ds.tgt_itos

    loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=cfg['smoothing'])

    if use_noam:
        # Paper-style: base LR = 1.0, scheduler does all the work
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])
    else:
        # Constant LR — no scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=FIXED_LR, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = None

    # ── Log the LR schedule for the first 10 000 steps ────────────────
    if use_noam:
        lr_data = []
        total_steps_to_log = min(10_000, cfg['num_epochs'] * len(train_loader))
        _opt_tmp  = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        _sched_tmp = NoamScheduler(_opt_tmp, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])
        for s in range(total_steps_to_log):
            lr_data.append([s + 1, _opt_tmp.param_groups[0]['lr']])
            _opt_tmp.step()
            _sched_tmp.step()

        lr_table = wandb.Table(columns=["step", "lr"], data=lr_data)
        wandb.log({
            "lr_schedule/noam_curve": wandb.plot.line(
                lr_table, "step", "lr",
                title="Noam LR Schedule (first 10k steps)"
            )
        })
    else:
        # Log a flat line for fixed LR
        lr_data = [[s, FIXED_LR] for s in range(0, 10_001, 100)]
        lr_table = wandb.Table(columns=["step", "lr"], data=lr_data)
        wandb.log({
            "lr_schedule/fixed_lr_curve": wandb.plot.line(
                lr_table, "step", "lr",
                title="Fixed LR Schedule (1e-4)"
            )
        })

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_ckpt     = f"best_ckpt_{condition_name}.pt"

    train_loss_history = []
    val_loss_history   = []
    lr_history         = []

    global_step = 0

    for epoch in range(cfg['num_epochs']):

        # ---------- train ----------
        model.train()
        total_loss, total_tokens = 0.0, 0
        from tqdm import tqdm

        pbar = tqdm(train_loader, desc=f"[{condition_name}] Epoch {epoch} [train]", leave=False)
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            current_lr  = optimizer.param_groups[0]['lr']
            n_tokens    = (target_flat != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            global_step  += 1

            # Log step-level metrics
            wandb.log({
                "step/train_loss": loss.item(),
                "step/learning_rate": current_lr,
                "global_step": global_step,
            })

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

        train_loss = total_loss / max(total_tokens, 1)

        # ---------- validation ----------
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

        val_loss = val_total_loss / max(val_total_tokens, 1)
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch-level metrics
        wandb.log({
            "epoch/train_loss":    train_loss,
            "epoch/val_loss":      val_loss,
            "epoch/learning_rate": current_lr,
            "epoch":               epoch,
        })

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        lr_history.append(current_lr)

        print(f"  [{condition_name}] Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler if scheduler else 
                           NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps']),
                           epoch, path=best_ckpt)

    # ── Final BLEU ────────────────────────────────────────────────────

    class VocabWrapper:
        def __init__(self, itos): self.itos = itos

    tgt_vocab = VocabWrapper(train_ds.tgt_itos)

    # Load best checkpoint for BLEU evaluation
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"  [{condition_name}] Final Test BLEU: {bleu:.2f}")

    wandb.log({
        "test/bleu":          bleu,
        "test/best_val_loss": best_val_loss,
    })

    # ── Summary table: epoch-wise train & val loss ────────────────────
    summary_data = [
        [e, train_loss_history[e], val_loss_history[e], lr_history[e]]
        for e in range(len(train_loss_history))
    ]
    summary_table = wandb.Table(
        columns=["epoch", "train_loss", "val_loss", "learning_rate"],
        data=summary_data,
    )
    wandb.log({"epoch_summary": summary_table})

    wandb.finish()
    return train_loss_history, val_loss_history, bleu


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    cfg = BASE_CONFIG.copy()

    # Build datasets once
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_data(cfg)
    print(f"  src vocab: {len(train_ds.src_itos)}, tgt vocab: {len(train_ds.tgt_itos)}")

    # ── Experiment 1: Noam Scheduler ──────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 1: Noam Scheduler")
    print("="*60)
    noam_train, noam_val, noam_bleu = run_one_experiment(
        condition_name = "noam_scheduler",
        use_noam       = True,
        cfg            = cfg,
        train_ds       = train_ds,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        device         = device,
    )

    # ── Experiment 2: Fixed LR ────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERIMENT 2: Fixed LR = 1e-4")
    print("="*60)
    fixed_train, fixed_val, fixed_bleu = run_one_experiment(
        condition_name = "fixed_lr",
        use_noam       = False,
        cfg            = cfg,
        train_ds       = train_ds,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        device         = device,
    )

    # ── Overlay comparison run ─────────────────────────────────────────
    # Log both curves into a single W&B run for easy side-by-side charts
    print("\n" + "="*60)
    print("Logging combined comparison run...")
    print("="*60)

    wandb.init(
        project = WANDB_PROJECT,
        group   = WANDB_GROUP,
        name    = "2.1_comparison_overlay",
        config  = cfg,
        reinit  = True,
    )

    # Build comparison tables for custom charts
    comparison_data = []
    for e in range(len(noam_train)):
        comparison_data.append([e, "noam_scheduler", noam_train[e], noam_val[e]])
    for e in range(len(fixed_train)):
        comparison_data.append([e, "fixed_lr",       fixed_train[e], fixed_val[e]])

    comparison_table = wandb.Table(
        columns=["epoch", "condition", "train_loss", "val_loss"],
        data=comparison_data,
    )

    # Log line plots for both conditions
    wandb.log({
        "comparison/train_loss_overlay": wandb.plot.line(
            comparison_table, x="epoch", y="train_loss",
            stroke="condition",
            title="Train Loss: Noam vs Fixed LR"
        ),
        "comparison/val_loss_overlay": wandb.plot.line(
            comparison_table, x="epoch", y="val_loss",
            stroke="condition",
            title="Val Loss: Noam vs Fixed LR"
        ),
    })

    # Final BLEU comparison bar chart
    bleu_table = wandb.Table(
        columns=["condition", "test_bleu"],
        data=[
            ["noam_scheduler", noam_bleu],
            ["fixed_lr",       fixed_bleu],
        ],
    )
    wandb.log({
        "comparison/bleu_bar": wandb.plot.bar(
            bleu_table, "condition", "test_bleu",
            title="Test BLEU: Noam vs Fixed LR"
        ),
        "noam_scheduler/test_bleu": noam_bleu,
        "fixed_lr/test_bleu":       fixed_bleu,
    })

    print(f"\n{'='*60}")
    print(f"  Noam Scheduler — Test BLEU: {noam_bleu:.2f}")
    print(f"  Fixed LR 1e-4  — Test BLEU: {fixed_bleu:.2f}")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    main()
