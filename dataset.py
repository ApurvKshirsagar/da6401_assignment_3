"""
dataset.py — Data loading and preprocessing for Multi30k (De→En)
DA6401 Assignment 3: "Attention Is All You Need"
"""

from collections import Counter

import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset


# ══════════════════════════════════════════════════════════════════════
#  SPECIAL TOKEN INDICES  (fixed positions in both vocabularies)
# ══════════════════════════════════════════════════════════════════════
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']


class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split

        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        self.raw = load_dataset("bentrevett/multi30k", split=split)

        # Load spacy tokenizers for German (src) and English (tgt)
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

    # ------------------------------------------------------------------
    # Tokenizers
    # ------------------------------------------------------------------
    def tokenize_de(self, text: str):
        """Lowercase German text into a list of tokens."""
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str):
        """Lowercase English text into a list of tokens."""
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ------------------------------------------------------------------
    def build_vocab(self, min_freq: int = 2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # Always build vocab from the training split regardless of self.split
        train_raw = load_dataset("bentrevett/multi30k", split='train')

        de_counter: Counter = Counter()
        en_counter: Counter = Counter()

        for item in train_raw:
            de_counter.update(self.tokenize_de(item['de']))
            en_counter.update(self.tokenize_en(item['en']))

        # src vocab: specials first, then tokens that appear >= min_freq times
        self.src_itos = SPECIALS + [w for w, c in de_counter.items() if c >= min_freq]
        self.src_stoi = {w: i for i, w in enumerate(self.src_itos)}

        # tgt vocab
        self.tgt_itos = SPECIALS + [w for w, c in en_counter.items() if c >= min_freq]
        self.tgt_stoi = {w: i for i, w in enumerate(self.tgt_itos)}

    # ------------------------------------------------------------------
    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.

        Returns:
            list of (src_ids, tgt_ids) tuples where both are Python lists of ints.
        """
        self.data = []
        for item in self.raw:
            src_ids = (
                [SOS_IDX]
                + [self.src_stoi.get(t, UNK_IDX) for t in self.tokenize_de(item['de'])]
                + [EOS_IDX]
            )
            tgt_ids = (
                [SOS_IDX]
                + [self.tgt_stoi.get(t, UNK_IDX) for t in self.tokenize_en(item['en'])]
                + [EOS_IDX]
            )
            self.data.append((src_ids, tgt_ids))
        return self.data


# ══════════════════════════════════════════════════════════════════════
#  TORCH DATASET WRAPPER
# ══════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """Thin wrapper around processed (src_ids, tgt_ids) pairs."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════
#  COLLATE — pads a batch to the same length
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """
    Pads source and target sequences in a batch to equal length.

    Returns:
        src_batch : LongTensor [batch, max_src_len]
        tgt_batch : LongTensor [batch, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


# ══════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION — returns ready-to-use DataLoaders
# ══════════════════════════════════════════════════════════════════════

def get_dataloader(split: str, dataset_obj: Multi30kDataset, batch_size: int = 128,
                   shuffle: bool = True) -> DataLoader:
    """
    Build a DataLoader for the given split.

    Args:
        split        : 'train', 'validation', or 'test'
        dataset_obj  : A Multi30kDataset that has already had
                       build_vocab() and process_data() called.
        batch_size   : Batch size (default 128).
        shuffle      : Whether to shuffle (default True for train).

    Returns:
        torch.utils.data.DataLoader
    """
    torch_ds = TranslationDataset(dataset_obj.data)
    return DataLoader(
        torch_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=0,   # 0 is safest on Windows
    )


# ══════════════════════════════════════════════════════════════════════
#  QUICK SMOKE TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading train split...")
    train_ds = Multi30kDataset(split='train')

    print("Building vocab...")
    train_ds.build_vocab(min_freq=2)
    print(f"  src vocab size: {len(train_ds.src_itos)}")
    print(f"  tgt vocab size: {len(train_ds.tgt_itos)}")

    print("Processing data...")
    train_ds.process_data()
    print(f"  training pairs: {len(train_ds.data)}")
    print(f"  sample src ids: {train_ds.data[0][0][:10]}")
    print(f"  sample tgt ids: {train_ds.data[0][1][:10]}")

    loader = get_dataloader('train', train_ds, batch_size=32)
    src_b, tgt_b = next(iter(loader))
    print(f"  batch src shape: {src_b.shape}")
    print(f"  batch tgt shape: {tgt_b.shape}")
    print("dataset.py smoke test passed!")