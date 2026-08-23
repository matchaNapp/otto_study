import json
import numpy as np
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path

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


pair_frequency = defaultdict(int)


# ============================================================
# Estimate pair frequency statistics
# ============================================================

train_sessions = list(iter_sessions(TRAIN_PATH))

for session_data in tqdm(
        train_sessions,
        desc="Computing pair statistics"
):
    for event in session_data["events"]:

        pair_id = get_pair_id(
            event["aid"],
            event["type"]
        )

        pair_frequency[pair_id] += 1



# ============================================================
# Build sparse transition matrix
# ============================================================

rows = []
cols = []

for session_data in tqdm(
        train_sessions,
        desc="Building transition matrix"
):

    pair_sequence = []

    for event in session_data["events"]:

        pair_sequence.append(
            get_pair_id(
                event["aid"],
                event["type"]
            )
        )


    # Model sequential transition relationships
    for i in range(len(pair_sequence) - 1):

        source = pair_sequence[i]
        target = pair_sequence[i + 1]

        rows.append(source)
        cols.append(target)

        rows.append(target)
        cols.append(source)



transition_matrix = sparse.coo_matrix(
    (
        np.ones(len(rows), dtype=np.float32),
        (rows, cols)
    ),
    shape=(pair_count, pair_count)
).tocsr()


del rows, cols



# ============================================================
# Compute cosine similarity
# ============================================================

frequency_vector = np.array(
    [
        pair_frequency[i]
        for i in range(pair_count)
    ],
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

    if transition_matrix[i].nnz == 0:
        continue


    norm_i = frequency_norm[i]

    if norm_i == 0:
        continue


    neighbors = transition_matrix[i].indices
    weights = transition_matrix[i].data


    for j, co_value in zip(neighbors, weights):

        norm_j = frequency_norm[j]

        if norm_j == 0:
            continue


        similarity = co_value / (
            norm_i * norm_j
        )


        similarity_rows.append(i)
        similarity_cols.append(j)
        similarity_values.append(similarity)



pair_similarity_matrix = sparse.coo_matrix(
    (
        similarity_values,
        (
            similarity_rows,
            similarity_cols
        )
    ),
    shape=(pair_count, pair_count)
).tocsr()


del (
    transition_matrix,
    frequency_vector,
    frequency_norm,
    similarity_rows,
    similarity_cols,
    similarity_values
)



# ============================================================
# Top-K neighbor selection
# ============================================================

TOP_K_NEIGHBORS = 20

topk_neighbors = {}


for i in tqdm(
        range(pair_count),
        desc="Selecting top-K neighbors"
):

    if pair_similarity_matrix[i].nnz == 0:

        topk_neighbors[i] = []

        continue


    indices = pair_similarity_matrix[i].indices
    scores = pair_similarity_matrix[i].data


    if len(scores) <= TOP_K_NEIGHBORS:

        selected = np.argsort(scores)[::-1]

    else:

        selected = np.argpartition(
            scores,
            -TOP_K_NEIGHBORS
        )[-TOP_K_NEIGHBORS:]

        selected = selected[
            np.argsort(scores[selected])[::-1]
        ]


    topk_neighbors[i] = [
        (
            indices[k],
            scores[k]
        )
        for k in selected
    ]



del pair_similarity_matrix



# ============================================================
# Recommendation generation
# ============================================================

TARGET_TYPES = [
    "clicks",
    "carts",
    "orders"
]


submission_rows = []


valid_sessions = list(
    iter_sessions(TEST_PATH)
)


for session_data in tqdm(
        valid_sessions,
        desc="Generating recommendations"
):

    session_id = session_data["session"]

    observed_pairs = set()


    for event in session_data["events"]:

        observed_pairs.add(
            get_pair_id(
                event["aid"],
                event["type"]
            )
        )


    for target_type in TARGET_TYPES:

        candidate_scores = defaultdict(float)


        for pair_id in observed_pairs:

            if pair_id not in topk_neighbors:
                continue


            for neighbor_id, similarity in topk_neighbors[pair_id]:

                aid, interaction_type = id_to_pair[neighbor_id]


                if interaction_type == target_type:

                    candidate_scores[aid] += similarity



        ranked_items = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]


        if ranked_items:

            predictions = " ".join(
                str(aid)
                for aid, _ in ranked_items
            )

        else:

            predictions = "129004"



        submission_rows.append(
            {
                "session_type":
                    f"{session_id}_{target_type}",

                "labels":
                    predictions
            }
        )



# ============================================================
# Save predictions and evaluate
# ============================================================

submission_path = OUTPUT_DIR / "itemCF_1.csv"


submission_df = pd.DataFrame(
    submission_rows
)


submission_df.to_csv(
    submission_path,
    index=False
)


print(
    f"Submission saved to {submission_path}"
)


from otto_metrics import evaluate_and_log


evaluate_and_log(
    "itemCF_1",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
