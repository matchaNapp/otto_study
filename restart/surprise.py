import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
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


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



def iter_sessions(path):
    """
    Load session interaction sequences.

    Each event contains:
    - item identifier (aid)
    - interaction type
    - timestamp
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
# Item-type pair representation
# ============================================================

pair_to_id = {}

id_to_pair = {}

pair_count = 0



def get_pair_id(aid, interaction_type):
    """
    Encode (item, interaction type) pairs into integer ids.
    """

    global pair_count


    key = (
        aid,
        interaction_type
    )


    if key not in pair_to_id:

        pair_to_id[key] = pair_count

        id_to_pair[pair_count] = key

        pair_count += 1


    return pair_to_id[key]



# ============================================================
# Transition statistics construction
# ============================================================

pair_timestamps = defaultdict(list)

pair_transition_count = defaultdict(
    lambda: defaultdict(int)
)

pair_frequency = defaultdict(int)


aid_transition_prob = defaultdict(dict)



train_sessions = list(
    iter_sessions(TRAIN_PATH)
)



for session_data in tqdm(
        train_sessions,
        desc="Building transition statistics"
):

    events = session_data["events"]


    pair_sequence = []


    for event in events:

        pid = get_pair_id(
            event["aid"],
            event["type"]
        )


        pair_sequence.append(
            (
                pid,
                event["ts"]
            )
        )


        pair_timestamps[pid].append(
            event["ts"]
        )


        pair_frequency[pid] += 1



    # Sequential transition modeling

    for i in range(
        len(pair_sequence)-1
    ):

        pid_i, _ = pair_sequence[i]

        pid_j, _ = pair_sequence[i+1]


        pair_transition_count[pid_i][pid_j] += 1



# ============================================================
# Item-level transition probability estimation
# ============================================================

aid_transition_count = defaultdict(
    lambda: defaultdict(int)
)

aid_frequency = defaultdict(int)



for source_pid, destinations in pair_transition_count.items():

    source_aid, _ = id_to_pair[source_pid]


    aid_frequency[source_aid] += (
        pair_frequency[source_pid]
    )


    for target_pid, count in destinations.items():

        target_aid, _ = id_to_pair[target_pid]


        if source_aid != target_aid:

            aid_transition_count[source_aid][target_aid] += count



for source_aid, destinations in aid_transition_count.items():

    total = aid_frequency[source_aid]


    for target_aid, count in destinations.items():

        aid_transition_prob[source_aid][target_aid] = (
            count / total
            if total > 0
            else 0
        )



# ============================================================
# Surprise similarity computation
# ============================================================

TOP_K = 20


pair_similarity_topk = {}



for source_pid, destinations in tqdm(
        pair_transition_count.items(),
        desc="Computing similarity"
):

    source_aid, _ = id_to_pair[source_pid]


    source_frequency = (
        pair_frequency[source_pid]
    )


    neighbors = []


    for target_pid, transition_count in destinations.items():

        target_aid, _ = id_to_pair[target_pid]


        if target_aid in aid_transition_prob.get(
                source_aid,
                {}
        ):

            target_frequency = (
                pair_frequency[target_pid]
            )


            source_time = np.mean(
                pair_timestamps[source_pid]
            )


            target_time = np.mean(
                pair_timestamps[target_pid]
            )


            time_difference = abs(
                target_time - source_time
            )


            temporal_weight = (
                1.0 /
                (
                    1.0 +
                    np.log1p(
                        time_difference / 1000
                    )
                )
            )


            score = (
                temporal_weight /
                (
                    source_frequency *
                    target_frequency
                )
                if source_frequency > 0
                and target_frequency > 0
                else 0
            )


        else:

            score = (
                transition_count / source_frequency
                if source_frequency > 0
                else 0
            )


        if score > 0:

            neighbors.append(
                (
                    target_pid,
                    score
                )
            )


    neighbors.sort(
        key=lambda x: x[1],
        reverse=True
    )


    pair_similarity_topk[source_pid] = (
        neighbors[:TOP_K]
    )



del (
    pair_transition_count,
    pair_timestamps,
    aid_transition_prob,
    aid_transition_count
)



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



        for pid in observed_pairs:


            if pid not in pair_similarity_topk:

                continue



            for neighbor_pid, score in pair_similarity_topk[pid]:


                aid, interaction_type = (
                    id_to_pair[neighbor_pid]
                )


                if interaction_type == target_type:

                    candidate_scores[aid] += score



        ranked_items = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]



        prediction = (
            " ".join(
                str(aid)
                for aid, _ in ranked_items
            )
            if ranked_items
            else "116473"
        )



        submission_rows.append(
            {
                "session_type":
                    f"{session_id}_{target_type}",

                "labels":
                    prediction
            }
        )



# ============================================================
# Export predictions and evaluation
# ============================================================

submission_path = (
    OUTPUT_DIR /
    "surprise_1.csv"
)



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
    "surprise_1",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
