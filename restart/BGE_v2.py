"""
BGE embedding training: Skip-Gram with negative sampling for (aid, type) pairs.

This script learns vector representations for (item, interaction_type) pairs
using a Skip-Gram architecture with negative sampling, following the approach
used in the BGE (Billion-scale General Embedding) framework.

The model is trained on sequential co-occurrence patterns within a sliding
window over user interaction sequences, where each event is represented as a
(aid, interaction_type) tuple. The resulting embeddings capture both item
identity and interaction semantics.
"""

import json
import numpy as np
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "train.xlsx"
OUTPUT_DIR = ROOT / "result"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def iter_sessions(path):
    """
    Load session interaction sequences from the given file.

    Each session contains a sequence of user-item interaction events,
    where each event consists of an item identifier and interaction type.
    """
    df = pd.read_excel(path)

    for _, row in df.iterrows():
        events = row["events"]

        if isinstance(events, str):
            events = json.loads(events)

        yield {
            "session": int(row["session"]),
            "events": events
        }


# ============================================================
# Hyperparameters
# ============================================================

WINDOW_SIZE = 3
NUM_NEG_SAMPLES = 5
EMBEDDING_DIM = 64
BATCH_SIZE = 512
EPOCHS = 3


# ============================================================
# Build pair vocabulary from training data
# ============================================================

pair_to_id = {}
id_to_pair = {}
pair_count = 0


def get_pair_id(aid, interaction_type):
    """
    Map each (item, interaction_type) pair to a unique identifier.

    This representation treats different interaction types (clicks, carts,
    orders) on the same item as distinct entities, allowing the model to
    learn type-specific embeddings.
    """
    global pair_count

    key = (aid, interaction_type)

    if key not in pair_to_id:
        pair_to_id[key] = pair_count
        id_to_pair[pair_count] = key
        pair_count += 1

    return pair_to_id[key]


print("Building training pairs from sliding windows...")

all_pairs = []
train_sessions = list(iter_sessions(TRAIN_PATH))

for session_data in tqdm(train_sessions, desc="Extracting co-occurrence pairs"):
    events = session_data["events"]

    # Convert events to pair IDs
    pair_sequence = []
    for event in events:
        pair_id = get_pair_id(event["aid"], event["type"])
        pair_sequence.append(pair_id)

    n = len(pair_sequence)

    # Extract positive pairs within the sliding window
    for i, center_pair in enumerate(pair_sequence):
        left = max(0, i - WINDOW_SIZE)
        right = min(n, i + WINDOW_SIZE + 1)

        for j in range(left, right):
            if i != j:
                all_pairs.append((center_pair, pair_sequence[j]))


print(f"Vocabulary size (# of item-type pairs): {pair_count:,}")
print(f"Number of training pairs: {len(all_pairs):,}")
print(f"Number of training sessions: {len(train_sessions):,}")

train_indices = [(center, context) for center, context in all_pairs]

del all_pairs
gc.collect()


# ============================================================
# PyTorch Dataset implementation
# ============================================================

class BGEDataset(Dataset):
    """
    Dataset for Skip-Gram training with negative sampling.

    For each positive (center, context) pair, the dataset generates
    a fixed number of negative samples drawn uniformly from the
    entire vocabulary.
    """
    def __init__(self, pairs, num_items, num_neg):
        self.pairs = pairs
        self.num_items = num_items
        self.num_neg = num_neg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]

        # Sample negative items uniformly from vocabulary
        neg_samples = np.random.randint(0, self.num_items, size=self.num_neg)

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long)
        )


dataset = BGEDataset(train_indices, pair_count, NUM_NEG_SAMPLES)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# ============================================================
# Skip-Gram model definition
# ============================================================

class SkipGramModel(nn.Module):
    """
    Skip-Gram with negative sampling for learning item-type pair embeddings.

    The model maintains two embedding matrices:
    - input embeddings: representations for center items
    - output embeddings: representations for context items

    Training objective: maximize similarity between center and positive
    context pairs while minimizing similarity with negative samples.

    Loss formulation:
        L_pos = -log(sigmoid(center · context))
        L_neg = -log(sigmoid(-center · negative))

        Total Loss = L_pos + L_neg
    """
    def __init__(self, num_items, embedding_dim):
        super().__init__()

        self.in_emb = nn.Embedding(num_items, embedding_dim)
        self.out_emb = nn.Embedding(num_items, embedding_dim)

        # Xavier initialization for stable training
        nn.init.xavier_uniform_(self.in_emb.weight)
        nn.init.xavier_uniform_(self.out_emb.weight)

    def forward(self, center, pos_context, neg_context):
        """
        Forward pass computing both positive and negative losses.

        Args:
            center: batch of center item-type pair IDs
            pos_context: batch of positive context pair IDs
            neg_context: batch of negative context pair IDs (K per sample)

        Returns:
            Total loss (positive + negative)
        """
        center_emb = self.in_emb(center)
        pos_emb = self.out_emb(pos_context)
        neg_emb = self.out_emb(neg_context)

        # Positive loss: maximize similarity with positive context
        pos_score = torch.sum(center_emb * pos_emb, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()

        # Negative loss: minimize similarity with negative samples
        # Shape: (batch_size, num_neg, embedding_dim) x (batch_size, embedding_dim, 1)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()

        return pos_loss + neg_loss

    def get_item_vectors(self):
        """
        Extract trained input embeddings as NumPy array.

        Returns:
            np.ndarray of shape (num_items, embedding_dim)
        """
        return self.in_emb.weight.data.cpu().numpy()


# ============================================================
# Training
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SkipGramModel(pair_count, EMBEDDING_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Starting training on device: {device}")

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        center, pos_context, neg_context = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()
        loss = model(center, pos_context, neg_context)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")


# ============================================================
# Export embeddings
# ============================================================

VECTORS_PATH = OUTPUT_DIR / "bge_vectors.npy"
MAPPING_PATH = OUTPUT_DIR / "bge_id_mapping.json"

# Convert to contiguous NumPy array for efficient storage
item_vectors = np.ascontiguousarray(
    model.get_item_vectors(),
    dtype=np.float32
)

# Build ID-to-(aid, type) mapping
id_mapping = {
    str(pair_id): [int(aid), interaction_type]
    for pair_id, (aid, interaction_type) in id_to_pair.items()
}

# Validate consistency
assert item_vectors.shape == (pair_count, EMBEDDING_DIM)
assert len(id_mapping) == pair_count

# Save to disk
np.save(VECTORS_PATH, item_vectors)

with open(MAPPING_PATH, "w", encoding="utf-8") as f:
    json.dump(id_mapping, f)


print(f"Embedding matrix shape: {item_vectors.shape}, dtype={item_vectors.dtype}")
print(f"Vocabulary size: {pair_count:,}, mapping entries: {len(id_mapping):,}")
print(f"Vectors saved to: {VECTORS_PATH}")
print(f"Mapping saved to: {MAPPING_PATH}")

# Clean up
del model, dataloader, dataset, item_vectors
gc.collect()
