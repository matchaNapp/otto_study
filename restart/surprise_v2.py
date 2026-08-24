import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
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


# Surprise-specific data structures
pair_timestamps = defaultdict(list)
pair_after_count = defaultdict(lambda: defaultdict(int))
pair_total_count = defaultdict(int)
aid_after_prob = defaultdict(dict)


# ============================================================
# First pass: count sequential transitions and timestamps
# ============================================================

print("Processing training data: counting sequential transitions...")

train_sessions = list(iter_sessions(TRAIN_PATH))
train_session_count = 0
aid_appear_count = defaultdict(int)

for session_data in tqdm(
        train_sessions,
        desc="Training data"
):
    train_session_count += 1

    events = session_data["events"]

    pair_sequence = []
    for event in events:
        pair_id = get_pair_id(event['aid'], event['type'])
        pair_sequence.append((pair_id, event['ts']))
        pair_timestamps[pair_id].append(event['ts'])
        pair_total_count[pair_id] += 1
        aid_appear_count[event['aid']] += 1

    # Count only adjacent transitions
    for i in range(len(pair_sequence) - 1):
        source, _ = pair_sequence[i]
        target, _ = pair_sequence[i + 1]
        pair_after_count[source][target] += 1

print(f"Sessions processed: {train_session_count:,}")
print(f"(aid, type) pairs: {pair_count:,}")


# ============================================================
# Aggregate to item-level correlations
# ============================================================

print("Building item-level correlation matrix...")

aid_after_count = defaultdict(lambda: defaultdict(int))
aid_total = defaultdict(int)

for source, destinations in pair_after_count.items():
    source_aid, _ = id_to_pair[source]
    aid_total[source_aid] += pair_total_count[source]

    for target, transition_count in destinations.items():
        target_aid, _ = id_to_pair[target]
        if source_aid != target_aid:
            aid_after_count[source_aid][target_aid] += transition_count

# Estimate conditional probabilities P(target_aid | source_aid)
for source_aid, destinations in aid_after_count.items():
    total = aid_total[source_aid]
    for target_aid, transition_count in destinations.items():
        aid_after_prob[source_aid][target_aid] = transition_count / total

print(f"Item-level correlations: {sum(len(v) for v in aid_after_prob.values()):,}")


# ============================================================
# Compute Surprise similarity with Top-20 truncation
# ============================================================

print("Computing Surprise similarity (Top-20 truncation)...")

TOP_K = 20
pair_similarity = {}

for source, destinations in tqdm(
        pair_after_count.items(),
        desc="Computing similarity"
):
    source_aid, source_type = id_to_pair[source]
    source_total = pair_total_count[source]

    neighbors = []

    for target, transition_count in destinations.items():
        target_aid, target_type = id_to_pair[target]

        # If item-level prior exists, apply time-weighted adjustment
        if (source_aid in aid_after_prob and
            target_aid in aid_after_prob.get(source_aid, {})):

            target_total = pair_total_count[target]

            # Temporal decay: larger time gap reduces similarity
            timestamps_i = pair_timestamps[source]
            timestamps_j = pair_timestamps[target]

            avg_ts_i = np.mean(timestamps_i)
            avg_ts_j = np.mean(timestamps_j)
            time_gap = abs(avg_ts_j - avg_ts_i)
            time_weight = 1.0 / (1.0 + np.log1p(time_gap / 1000))

            surprise_score = time_weight / (source_total * target_total) if (source_total > 0 and target_total > 0) else 0

        else:
            # Fallback: use raw transition probability
            surprise_score = transition_count / source_total if source_total > 0 else 0

        if surprise_score > 0:
            neighbors.append((target, surprise_score))

    # Retain only Top-K most similar pairs
    neighbors.sort(key=lambda x: x[1], reverse=True)
    pair_similarity[source] = neighbors[:TOP_K]

print(f"Similarity records: {sum(len(v) for v in pair_similarity.values()):,}")

del pair_after_count, pair_timestamps, aid_after_prob, aid_after_count


# ============================================================
# Recommendation generation with temporal and type weighting
# ============================================================

print("Generating recommendations for valid set...")

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
            source = get_pair_id(event['aid'], event['type'])

            # Combine interaction type weight with temporal decay
            mix_weight = type_to_weight[event['type']] * time_weights[idx]

            # Propagate similarity scores from neighboring pairs
            neighbors = pair_similarity.get(source) or []
            if neighbors:
                sum_sim = sum(sim for _, sim in neighbors)
                for target, sim in neighbors:
                    aid = id_to_pair[target][0]
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

submission_path = OUTPUT_DIR / "surprise_v2.csv"

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(submission_df.head(10))

from otto_metrics import evaluate_and_log

evaluate_and_log(
    "surprise_v2",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
