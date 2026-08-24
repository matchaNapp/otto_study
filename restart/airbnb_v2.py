"""
Airbnb embedding training: Skip-Gram with global context for (aid, type) pairs.

This script implements an embedding learning approach inspired by the Airbnb
paper, extending the standard Skip-Gram with negative sampling by incorporating
a global context signal: the last order item within each session.

The model learns vector representations for (item, interaction_type) pairs
where:
- Local context: items co-occurring within a sliding window
- Global context: the final order item in the session serves as an additional
  positive signal for all items in that session

This encourages the embedding space to align with the ultimate conversion
behavior (orders) in addition to local co-occurrence patterns.
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
EPOCHS = 2
LAMBDA_GLOBAL = 0.1


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
session_last_order = {}

train_sessions = list(iter_sessions(TRAIN_PATH))

for session_data in tqdm(train_sessions, desc="Extracting co-occurrence pairs"):
    session_id = session_data["session"]
    events = session_data["events"]

    # Convert events to pair IDs
    pair_sequence = []
    for event in events:
        pair_id = get_pair_id(event["aid"], event["type"])
        pair_sequence.append(pair_id)

    n = len(pair_sequence)

    # Record the last order event in this session (if any)
    # Traverse in reverse to find the most recent order
    for event in reversed(events):
        if event["type"] == "orders":
            session_last_order[session_id] = get_pair_id(event["aid"], event["type"])
            break

    # Extract positive pairs within the sliding window
    for i, center_pair in enumerate(pair_sequence):
        left = max(0, i - WINDOW_SIZE)
        right = min(n, i + WINDOW_SIZE + 1)

        for j in range(left, right):
            if i != j:
                all_pairs.append((center_pair, pair_sequence[j], session_id))


print(f"Vocabulary size (# of item-type pairs): {pair_count:,}")
print(f"Number of training pairs: {len(all_pairs):,}")
print(f"Number of training sessions: {len(train_sessions):,}")
print(f"Sessions with order behavior: {len(session_last_order):,}")

train_indices = [(center, context, sid) for center, context, sid in all_pairs]
session_last_order_idx = {sid: pid for sid, pid in session_last_order.items()}

del all_pairs
gc.collect()


# ============================================================
# PyTorch Dataset implementation
# ============================================================

class AirbnbDataset(Dataset):
    """
    Dataset for Skip-Gram training with global context.

    For each positive (center, context) pair, the dataset generates
    a fixed number of negative samples and also records whether the
    session has an order event and its ID.

    The global context signal provides an additional positive target
    for the entire session, encouraging embeddings to align with
    conversion behavior.
    """
    def __init__(self, pairs, num_items, num_neg, session_last_order_idx):
        self.pairs = pairs
        self.num_items = num_items
        self.num_neg = num_neg
        self.session_last_order_idx = session_last_order_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context, session_id = self.pairs[idx]

        # Sample negative items uniformly from vocabulary
        neg_samples = np.random.randint(0, self.num_items, size=self.num_neg)

        # Global context: whether the session has an order and its pair ID
        has_global = session_id in self.session_last_order_idx
        booked_idx = self.session_last_order_idx.get(session_id, 0)

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
            torch.tensor(has_global, dtype=torch.bool),
            torch.tensor(booked_idx, dtype=torch.long)
        )


dataset = AirbnbDataset(
    train_indices,
    pair_count,
    NUM_NEG_SAMPLES,
    session_last_order_idx
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)


# ============================================================
# Airbnb model definition
# ============================================================

class AirbnbModel(nn.Module):
    """
    Skip-Gram with global context for embedding learning.

    The model extends standard Skip-Gram with negative sampling by
    incorporating a global context loss: for sessions that have an
    order event, all center items in that session are encouraged to
    be similar to the order item's embedding.

    Loss formulation:
        L_skipgram = -log(sigmoid(center · context)) +
                     -log(sigmoid(-center · negative))

        L_global = -log(sigmoid(center · order_item))

        Total Loss = L_skipgram + lambda * L_global

    where lambda controls the strength of the global context signal.
    """
    def __init__(self, num_items, embedding_dim, lambda_global=0.1):
        super().__init__()

        self.in_emb = nn.Embedding(num_items, embedding_dim)
        self.out_emb = nn.Embedding(num_items, embedding_dim)
        self.lambda_global = lambda_global

        # Xavier initialization for stable training
        nn.init.xavier_uniform_(self.in_emb.weight)
        nn.init.xavier_uniform_(self.out_emb.weight)

    def forward(self, center, pos_context, neg_context, has_global, booked_idx):
        """
        Forward pass computing Skip-Gram loss and optional global context loss.

        Args:
            center: batch of center item-type pair IDs
            pos_context: batch of positive context pair IDs
            neg_context: batch of negative context pair IDs (K per sample)
            has_global: boolean mask indicating which samples have order context
            booked_idx: pair IDs of the last order item for each sample

        Returns:
            Total loss (Skip-Gram + weighted global context)
        """
        center_emb = self.in_emb(center)
        pos_emb = self.out_emb(pos_context)
        neg_emb = self.out_emb(neg_context)

        # Positive loss: maximize similarity with positive context
        pos_score = torch.sum(center_emb * pos_emb, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()

        # Negative loss: minimize similarity with negative samples
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()

        skipgram_loss = pos_loss + neg_loss

        # Global context loss: only for sessions with order events
        if has_global.any():
            mask = has_global.bool()

            center_masked = center_emb[mask]
            booked_masked = self.in_emb(booked_idx[mask])

            global_score = torch.sum(center_masked * booked_masked, dim=1)
            global_loss = -torch.log(torch.sigmoid(global_score) + 1e-8).mean()

            return skipgram_loss + self.lambda_global * global_loss

        return skipgram_loss

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

model = AirbnbModel(
    pair_count,
    EMBEDDING_DIM,
    lambda_global=LAMBDA_GLOBAL
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Starting training on device: {device}")

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        center, pos_context, neg_context, has_global, booked_idx = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()
        loss = model(center, pos_context, neg_context, has_global, booked_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")


# ============================================================
# Export embeddings
# ============================================================

VECTORS_PATH = OUTPUT_DIR / "airbnb_vectors.npy"
MAPPING_PATH = OUTPUT_DIR / "airbnb_id_mapping.json"

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
