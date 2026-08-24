import json
import numpy as np
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path
import heapq

# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "train.xlsx"
TEST_PATH = ROOT / "valid_history.xlsx"

OUTPUT_DIR = ROOT / "result"
LABELS_PATH = ROOT / "valid_labels.json"
SCORES_PATH = OUTPUT_DIR / "scores.txt"

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
# Construct item-type pair representation
# ============================================================

pair_to_id = {}
id_to_pair = {}

pair_count = 0


def get_pair_id(aid, interaction_type):
    """
    Map each (item, interaction_type) pair to a unique identifier.
    """
    global pair_count

    key = (aid, interaction_type)

    if key not in pair_to_id:
        pair_to_id[key] = pair_count
        id_to_pair[pair_count] = key
        pair_count += 1

    return pair_to_id[key]


pair_popularity = defaultdict(int)

# ============================================================
# Estimate pair frequency statistics
# ============================================================

print("First pass: computing pair popularity...")
train_sessions = list(iter_sessions(TRAIN_PATH))

for session_data in tqdm(
        train_sessions,
        desc="First pass"
):
    for event in session_data["events"]:
        pair_id = get_pair_id(
            event["aid"],
            event["type"]
        )
        pair_popularity[pair_id] += 1

# ============================================================
# Build sparse co-occurrence matrix
# ============================================================

print("\nSecond pass: constructing co-occurrence matrix...")

rows = []
cols = []

for session_data in tqdm(
        train_sessions,
        desc="Second pass"
):
    pair_sequence = [
        get_pair_id(event["aid"], event["type"])
        for event in session_data["events"]
    ]

    # Model sequential co-occurrence relationships
    for i in range(len(pair_sequence) - 1):
        source = pair_sequence[i]
        target = pair_sequence[i + 1]

        rows.append(source)
        cols.append(target)

        rows.append(target)
        cols.append(source)

co_occur_sparse = sparse.coo_matrix(
    (
        np.ones(len(rows), dtype=np.float32),
        (rows, cols)
    ),
    shape=(pair_count, pair_count)
).tocsr()

print(f"Co-occurrence matrix shape: {co_occur_sparse.shape}")
print(f"Non-zero entries: {co_occur_sparse.nnz:,}")

del rows, cols

# ============================================================
# Compute cosine similarity with popularity weighting
# ============================================================

print("\nComputing cosine similarity...")

frequency_vector = np.array(
    [pair_popularity[i] for i in range(pair_count)],
    dtype=np.float64
)
frequency_norm = np.sqrt(frequency_vector)

similarity_rows = []
similarity_cols = []
similarity_values = []

for i in tqdm(
        range(pair_count),
        desc="Computing similarity"
):
    if co_occur_sparse[i].nnz == 0:
        continue

    norm_i = frequency_norm[i]
    if norm_i == 0:
        continue

    neighbors = co_occur_sparse[i].indices
    weights = co_occur_sparse[i].data

    for j, co_value in zip(neighbors, weights):
        norm_j = frequency_norm[j]
        if norm_j == 0:
            continue

        similarity = co_value / (norm_i * norm_j)

        similarity_rows.append(i)
        similarity_cols.append(j)
        similarity_values.append(similarity)

similarity_sparse = sparse.coo_matrix(
    (
        similarity_values,
        (similarity_rows, similarity_cols)
    ),
    shape=(pair_count, pair_count)
).tocsr()

print(f"Similarity matrix non-zero entries: {similarity_sparse.nnz:,}")

del (
    co_occur_sparse,
    frequency_vector,
    frequency_norm,
    similarity_rows,
    similarity_cols,
    similarity_values
)

# ============================================================
# Top-K neighbor selection
# ============================================================

print("\nExtracting Top-20 neighbors...")
TOP_K = 20
top20_similarity = {}

for i in tqdm(
        range(pair_count),
        desc="Top-K selection"
):
    if similarity_sparse[i].nnz == 0:
        top20_similarity[i] = []
        continue

    indices = similarity_sparse[i].indices
    scores = similarity_sparse[i].data

    if len(scores) <= TOP_K:
        selected = np.argsort(scores)[::-1]
    else:
        selected = np.argpartition(scores, -TOP_K)[-TOP_K:]
        selected = selected[np.argsort(scores[selected])[::-1]]

    top20_similarity[i] = [(indices[k], scores[k]) for k in selected]

print(f"Total Top-20 records: {sum(len(v) for v in top20_similarity.values()):,}")

del similarity_sparse

# ============================================================
# Recommendation generation with temporal and type weighting
# ============================================================

print("\nGenerating recommendations for valid set...")

TARGET_TYPES = ['clicks', 'carts', 'orders']
type_to_weight = {'clicks': 1, 'carts': 10, 'orders': 3}

submission_rows = []

valid_sessions = list(iter_sessions(TEST_PATH))

for session_data in tqdm(
        valid_sessions,
        desc="Prediction progress"
):
    session_id = session_data["session"]
    events = session_data["events"]

    aid_scores = {t: defaultdict(float) for t in TARGET_TYPES}

    for target_type in TARGET_TYPES:
        # Temporal decay: earlier events have lower influence
        seq_weight = 0.1 if target_type == 'clicks' else 0.5
        time_weights = np.power(2, np.linspace(seq_weight, 1, len(events))) - 1

        for idx, event in enumerate(events):
            pid = get_pair_id(event['aid'], event['type'])

            # Combine interaction type weight with temporal decay
            mix_weight = type_to_weight[event['type']] * time_weights[idx]

            # Propagate similarity scores from neighboring pairs
            neighbors = top20_similarity.get(pid) or []
            if neighbors:
                sum_sim = sum(sim for _, sim in neighbors)
                for neighbor_id, sim in neighbors:
                    aid = id_to_pair[neighbor_id][0]
                    aid_scores[target_type][aid] += mix_weight * sim / sum_sim

            # Direct contribution from the event itself
            aid_scores[target_type][event['aid']] += mix_weight

        scored = aid_scores[target_type]
        top20 = heapq.nlargest(20, scored.items(), key=lambda x: x[1])
        preds = " ".join(str(aid) for aid, _ in top20)

        submission_rows.append({
            "session_type": f"{session_id}_{target_type}",
            "labels": preds
        })

print(f"Predictions generated: {len(submission_rows):,} rows")

# ============================================================
# Save predictions and evaluate
# ============================================================

submission_path = OUTPUT_DIR / "itemCF_v2.csv"

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(submission_df.head(10))

from otto_metrics import evaluate_and_log

evaluate_and_log(
    "itemCF_v2",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
