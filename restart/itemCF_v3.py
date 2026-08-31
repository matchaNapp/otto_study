# ============================================================
# ItemCF v3: Candidate pool generation with pair-level item similarity
# ============================================================
#
# This script constructs a pair-level co-occurrence matrix and computes
# cosine similarity between (item, type) pairs. The resulting similarity
# scores are used to generate candidate pools and rankings.
#
# Key differences from v2:
#   - Retains full candidate pool (Top-100 per target type)
#   - Stores itemcf_last_sim, itemcf_mean_sim, itemcf_max_sim as features
#   - Preserves Top-20 neighbor selection in similarity matrix
#
# Outputs:
#   - candidates_itemcf_v3.parquet / .csv: Candidate pool with similarity features
#   - itemCF_v3.csv: Submission file (Top-20 predictions)
# ============================================================

import json
import pickle
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm

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

print(f"Training sessions: {len(train_sessions):,}")
print(f"(item, type) pairs: {pair_count:,}")

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
# Top-K neighbor selection (Top-20 only)
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

# ============================================================
# Build similarity lookup table for feature extraction
# ============================================================

print("\nBuilding similarity lookup table...")

pair_to_aid_sim = {}

for hist_pid, neighbors in top20_similarity.items():
    inner = {}
    for nid, weight in neighbors:
        aid = id_to_pair[nid][0]
        prev = inner.get(aid)
        if prev is None or weight > prev:
            inner[aid] = float(weight)
    pair_to_aid_sim[int(hist_pid)] = inner

# Save lookup table for later use
SIM_PKL = OUTPUT_DIR / "itemcf_pair_sim.pkl"

with open(SIM_PKL, "wb") as f:
    pickle.dump(
        {
            "pair_to_aid_sim": pair_to_aid_sim,
            "pair_to_id": {f"{aid}|{typ}": pid for (aid, typ), pid in pair_to_id.items()},
            "id_to_pair": {str(pid): [int(aid), typ] for pid, (aid, typ) in id_to_pair.items()},
        },
        f,
    )

print(f"Similarity lookup table saved: {SIM_PKL}")
print(f"  keys: {len(pair_to_aid_sim):,}")

del similarity_sparse

# ============================================================
# Recommendation generation with candidate pool (Top-100)
# ============================================================

print("\nGenerating candidate pool (Top-100 per target type)...")

TARGET_TYPES = ['clicks', 'carts', 'orders']
type_to_weight = {'clicks': 1, 'carts': 10, 'orders': 3}

CAND_TOPK = 100

submission_rows = []
candidate_rows = []

valid_sessions = list(iter_sessions(TEST_PATH))


def _itemcf_last_mean_max(cand_aid, events):
    """
    Compute last/mean/max similarity between a candidate item and the session history.

    For each historical event, look up the precomputed similarity between the
    historical (item, type) pair and the candidate item. Return:
      - last_sim: similarity with the most recent event
      - mean_sim: average similarity across all events
      - max_sim: maximum similarity across all events
    """
    if not events:
        return -1.0, -1.0, -1.0

    sims = []

    for event in events:
        hist_pid = pair_to_id.get((int(event["aid"]), event["type"]))

        if hist_pid is None:
            sims.append(0.0)
            continue

        sims.append(
            float(pair_to_aid_sim.get(hist_pid, {}).get(int(cand_aid), 0.0))
        )

    if not sims:
        return -1.0, -1.0, -1.0

    return float(sims[-1]), float(np.mean(sims)), float(np.max(sims))


for session_data in tqdm(
        valid_sessions,
        desc="Prediction progress"
):
    session_id = session_data["session"]
    events = session_data["events"]

    # Sort by timestamp for proper temporal ordering
    events_sorted = sorted(events, key=lambda e: int(e["ts"]))

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

        # Select Top-100 candidates
        scored = aid_scores[target_type]
        topk = heapq.nlargest(CAND_TOPK, scored.items(), key=lambda x: x[1])

        # Store candidate pool with similarity features
        for rank, (aid, score) in enumerate(topk):
            last_s, mean_s, max_s = _itemcf_last_mean_max(int(aid), events_sorted)

            candidate_rows.append({
                "session": session_id,
                "type": target_type,
                "aid": int(aid),
                "rank": rank,
                "score_itemcf": float(score),
                "itemcf_last_sim": last_s,
                "itemcf_mean_sim": mean_s,
                "itemcf_max_sim": max_s,
            })

        # Prepare submission: Top-20 predictions
        top20 = topk[:20]
        preds = " ".join(str(aid) for aid, _ in top20)

        submission_rows.append({
            "session_type": f"{session_id}_{target_type}",
            "labels": preds
        })

print(f"Predictions generated: {len(submission_rows):,} rows")
print(f"Candidate pool entries: {len(candidate_rows):,} rows")

# ============================================================
# Save candidate pool
# ============================================================

cand_df = pd.DataFrame(candidate_rows)

CAND_PARQUET = OUTPUT_DIR / "candidates_itemcf_v3.parquet"
CAND_CSV = OUTPUT_DIR / "candidates_itemcf_v3.csv"

try:
    cand_df.to_parquet(CAND_PARQUET, index=False)
    print(f"Candidate pool saved: {CAND_PARQUET}")
except Exception as e:
    print(f"Parquet write failed ({e}), writing CSV only")

cand_df.to_csv(CAND_CSV, index=False)
print(f"Candidate pool saved: {CAND_CSV}")

# ============================================================
# Save submission file
# ============================================================

submission_path = OUTPUT_DIR / "itemCF_v3.csv"

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission saved to {submission_path}")
print(f"Total rows: {len(submission_df):,}")
print(submission_df.head(10))

# ============================================================
# Evaluate predictions
# ============================================================

from otto_metrics import evaluate_and_log

evaluate_and_log(
    "itemCF_v3",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
