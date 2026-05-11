"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    Multi30kDataset, TranslationDataset, get_dataloader,
    PAD_IDX, SOS_IDX, EOS_IDX
)
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        # KLDivLoss expects log-probabilities as input
        self.criterion  = nn.KLDivLoss(reduction='sum')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)   # [N, vocab_size]

        # Build smoothed target distribution
        with torch.no_grad():
            # fill every class with eps / (V - 2)  (-2: exclude <pad> and true class)
            smooth_dist = torch.full_like(log_probs,
                                          self.smoothing / (self.vocab_size - 2))
            # place 1 - eps on the correct class
            smooth_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            # <pad> positions contribute zero probability
            smooth_dist[:, self.pad_idx] = 0.0
            # zero out rows where the target IS <pad>
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        loss = self.criterion(log_probs, smooth_dist)

        # Normalise by number of non-pad tokens
        n_tokens = (~pad_mask).sum().clamp(min=1)
        return loss / n_tokens


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    mode = "train" if is_train else "val"

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{mode}]", leave=False)
        for src, tgt in pbar:
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Decoder input: all tokens except last (<eos>)
            tgt_input  = tgt[:, :-1]
            # Decoder target (what we predict): all tokens except first (<sos>)
            tgt_target = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=PAD_IDX).to(device)

            # Forward pass → logits [batch, tgt_len-1, vocab_size]
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Flatten for loss: [batch*(tgt_len-1), vocab_size]
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = tgt_target.reshape(-1)

            loss = loss_fn(logits_flat, target_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping helps stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens     = (target_flat != PAD_IDX).sum().item()
            total_loss  += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_tokens, 1)

    # Log to W&B if a run is active
    try:
        wandb.log({f"{mode}/loss": avg_loss, "epoch": epoch_num})
    except Exception:
        pass

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)   # [1, src_len, d_model]

        # Start with <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)  # [1, cur_len, vocab]
            # Greedy: pick the highest-probability token at the last position
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : object with .itos list  (tgt_vocab.itos[idx] → word)
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    model.eval()
    hypotheses = []   # list of token lists (predictions)
    references = []   # list of [token list] (gold, one ref per sentence)

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)               # [1, src_len]
                src_mask = make_src_mask(src_i, pad_idx=PAD_IDX).to(device)

                pred = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )
                pred_ids = pred.squeeze(0).tolist()

                # Strip <sos>, stop at <eos>
                hyp_tokens = []
                for idx in pred_ids:
                    if idx == EOS_IDX:
                        break
                    if idx != SOS_IDX:
                        hyp_tokens.append(tgt_vocab.itos[idx])

                # Reference: strip <sos>, <eos>, <pad>
                ref_ids = tgt[i].tolist()
                ref_tokens = [tgt_vocab.itos[idx]
                              for idx in ref_ids
                              if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)]

                hypotheses.append(hyp_tokens)
                references.append([ref_tokens])   # nltk expects list of refs

    # corpus_bleu expects: list[list[list[str]]], list[list[str]]
    smoother = SmoothingFunction().method1
    score = corpus_bleu(references, hypotheses, smoothing_function=smoother)
    return score * 100   # convert 0-1 → 0-100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':         model.config,   # set in Transformer.__init__
    }, path)
    print(f"  ✔ Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f"  ✔ Checkpoint loaded ← {path}  (epoch {ckpt['epoch']})")
    return ckpt['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """

    # ── Hyperparameters ───────────────────────────────────────────────
    config = dict(
        d_model      = 512,
        N            = 6,
        num_heads    = 8,
        d_ff         = 2048,
        dropout      = 0.1,
        warmup_steps = 4000,
        batch_size   = 128,
        num_epochs   = 20,
        smoothing    = 0.1,
        min_freq     = 2,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Init W&B (set mode='disabled' to skip W&B during quick tests)
    wandb.init(project="da6401-a3", config=config, mode="online")
    cfg = wandb.config

    # 2. Build datasets & vocabs
    print("Building datasets...")
    train_ds = Multi30kDataset(split='train')
    train_ds.build_vocab(min_freq=cfg.min_freq)

    val_ds  = Multi30kDataset(split='validation')
    val_ds.src_stoi = train_ds.src_stoi;  val_ds.src_itos = train_ds.src_itos
    val_ds.tgt_stoi = train_ds.tgt_stoi;  val_ds.tgt_itos = train_ds.tgt_itos

    test_ds = Multi30kDataset(split='test')
    test_ds.src_stoi = train_ds.src_stoi;  test_ds.src_itos = train_ds.src_itos
    test_ds.tgt_stoi = train_ds.tgt_stoi;  test_ds.tgt_itos = train_ds.tgt_itos

    train_ds.process_data()
    val_ds.process_data()
    test_ds.process_data()

    # 3. DataLoaders
    train_loader = get_dataloader('train',      train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = get_dataloader('validation', val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = get_dataloader('test',       test_ds,  batch_size=1,              shuffle=False)

    src_vocab_size = len(train_ds.src_itos)
    tgt_vocab_size = len(train_ds.tgt_itos)
    print(f"  src vocab: {src_vocab_size}, tgt vocab: {tgt_vocab_size}")

    # 4. Model
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model   = cfg.d_model,
        N         = cfg.N,
        num_heads = cfg.num_heads,
        d_ff      = cfg.d_ff,
        dropout   = cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {n_params:,}")

    # 5. Optimizer (paper: β1=0.9, β2=0.98, ε=1e-9)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    # 6. Noam scheduler
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # 7. Loss
    loss_fn = LabelSmoothingLoss(
        vocab_size = tgt_vocab_size,
        pad_idx    = PAD_IDX,
        smoothing  = cfg.smoothing,
    )

    # 8. Training loop
    best_val_loss = float('inf')
    best_ckpt     = "best_checkpoint.pt"

    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
        wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

        # Also save latest checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, path="checkpoint.pt")

    # 9. Final BLEU on test set

    # Give model vocab access for infer()
    model.src_stoi = train_ds.src_stoi
    model.tgt_stoi = train_ds.tgt_stoi
    model.tgt_itos = train_ds.tgt_itos

    # Load best weights for evaluation
    load_checkpoint(best_ckpt, model)

    # Simple vocab wrapper so evaluate_bleu can call tgt_vocab.itos[idx]
    class VocabWrapper:
        def __init__(self, itos): self.itos = itos
    tgt_vocab = VocabWrapper(train_ds.tgt_itos)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"\nTest BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()