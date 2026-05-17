# DA6401 Assignment 3 — Implementing a Transformer for Machine Translation

**Student:** Apurv Kshirsagar | CE22B042 | IIT Madras

## Links

- **WandB Report:** https://wandb.ai/ce22b042-iit-madras/da6401-a3/reports/DA6401-Assignment-3--VmlldzoxNjg4MzM4OA?accessToken=tu3g1xgwcsbe488zsqz2u1s9smsl3exrxm22nj1opjcnqo5i7fw7y5q3kvfb5s96
- **GitHub Repo:** https://github.com/ApurvKshirsagar/da6401_assignment_3

---

## Results (Autograder: 50/50)

| Test | Description | Score |
|------|-------------|-------|
| 1.1 | MHA: Correctness of output shape | 2/2 |
| 1.2 | MHA: Attention weights sum to 1 (Softmax Check) | 2/2 |
| 1.3 | MHA: Masked positions receive zero weight | 2/2 |
| 1.4 | MHA: Output shape under varying d_model and num_heads | 2/2 |
| 1.5 | MHA: Causal masking produces different outputs | 2/2 |
| 2.1 | Positional Encoding: Output shape preservation | 2/2 |
| 2.2 | Positional Encoding: Even-indexed dims = sin(0) = 0 at pos 0 | 2/2 |
| 2.3 | Positional Encoding: Odd-indexed dims = cos(0) = 1 at pos 0 | 2/2 |
| 2.4 | Positional Encoding: Formula correctness | 2/2 |
| 2.5 | Positional Encoding: Registered as buffer | 2/2 |
| 3.1 | Noam Scheduler: Monotonically increasing during warm-up | 2/2 |
| 3.2 | Noam Scheduler: Peak occurs near warmup_steps | 2/2 |
| 3.3 | Noam Scheduler: Monotonically decreasing after warm-up | 2/2 |
| 3.4 | Noam Scheduler: Peak value matches closed-form formula | 2/2 |
| 3.5 | Noam Scheduler: LR at step 1 matches formula | 2/2 |
| 4.1 | Translation: BLEU Score > 20.0 | 5/5 |
| 4.2 | Translation: BLEU Score > 25.0 | 5/5 |
| 4.3 | Translation: BLEU Score > 30.0 | 5/5 |
| 4.4 | Translation: BLEU Score > 35.0 | 5/5 |

---

## Architecture

- Full encoder-decoder Transformer implemented from scratch in PyTorch
- **Scaled Dot-Product Attention** with optional causal and padding masks
- **Multi-Head Attention** — no `nn.MultiheadAttention` used; implemented via `nn.Linear` projections
- **Sinusoidal Positional Encoding** registered as a non-trainable buffer
- **Point-wise Feed-Forward Network**: two `nn.Linear` layers with ReLU in between
- **Post-LayerNorm** (Add & Norm after each sub-layer, matching the original paper)
- **Noam LR Scheduler** with linear warmup and inverse square root decay
- **Label Smoothing** (ε = 0.1) via custom `LabelSmoothingLoss`
- **Greedy Decoding** for inference, token-by-token with `<sos>`/`<eos>` termination

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| N (layers) | 3 |
| num_heads | 8 |
| d_ff | 512 |
| dropout | 0.1 |
| warmup_steps | 2000 |
| batch_size | 128 |
| num_epochs | 50 |
| label smoothing | 0.1 |
| min_freq | 2 |

> The paper's full model (d_model=512, N=6, d_ff=2048) is oversized for Multi30k's 29k training pairs and overfits. A smaller model converges faster and generalizes better on this dataset, achieving 35+ corpus BLEU.

## Design Choices

- **Post-LayerNorm** chosen to match the original "Attention Is All You Need" paper exactly
- **spaCy `de_core_news_sm`** used for German tokenization; auto-downloaded at inference time if not installed
- Vocab and tokenizer state saved inside the checkpoint's `model_config` so `Transformer()` is fully self-contained — no external vocab files needed
- **Greedy decoding** (not beam search) as required; outputs stop at `<eos>` or `max_len`
- `<pad>`, `<unk>`, `<sos>`, `<eos>` special tokens added to both source and target vocabularies

---

## Usage

### Training

```bash
python train.py
```

Trains on the Multi30k De→En dataset. Saves `best_checkpoint.pt` (best validation BLEU) and `checkpoint.pt` (latest epoch) to the working directory. Logs training/validation loss and BLEU to W&B.

### Inference

```python
from model import Transformer

model = Transformer()          # downloads checkpoint automatically from Google Drive
model.eval()

translation = model.infer("Ein Mann sitzt auf einer Bank.")
print(translation)             # e.g. "A man is sitting on a bench."
```

### Evaluation (BLEU)

```python
from train import evaluate_bleu, get_dataloader
from dataset import TranslationDataset

# build test loader (vocab must match training)
bleu = evaluate_bleu(model, test_loader, tgt_itos)
print(f"Corpus BLEU: {bleu:.2f}")
```

---

## Project Structure

```
da6401_assignment_3/
├── model.py          # Transformer, MHA, PositionalEncoding, greedy infer()
├── train.py          # training loop, Noam scheduler, label smoothing, BLEU eval
├── dataset.py        # Multi30k dataset loader, vocab builder, DataLoader
├── lr_scheduler.py   # NoamScheduler implementation
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install torch numpy matplotlib scikit-learn wandb datasets spacy tqdm gdown bleu nltk
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```