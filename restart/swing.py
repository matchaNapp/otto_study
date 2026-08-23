import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import sqlite3
import os
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "train.xlsx"
TEST_PATH = ROOT / "valid_history.xlsx"

OUTPUT_DIR = ROOT / "result"

DB_PATH = OUTPUT_DIR / "pair_sessions.db"

LABELS_PATH = ROOT / "valid_labels.json"
SCORES_PATH = OUTPUT_DIR / "scores.txt"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



def iter_sessions(path):
    """
    Load session interaction sequences.
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
# Item-type pair encoding
# ============================================================

pair_to_id = {}

id_to_pair = {}

pair_count = 0



def get_pair_id(aid, interaction_type):

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
# Build inverted index
# ============================================================

if os.path.exists(DB_PATH):

    os.remove(DB_PATH)



connection = sqlite3.connect(
    DB_PATH
)

cursor = connection.cursor()


cursor.execute(
    """
    CREATE TABLE pair_session (
        pid INTEGER,
        sid INTEGER
    )
    """
)


cursor.execute(
    "CREATE INDEX idx_pid ON pair_session(pid)"
)


cursor.execute(
    "CREATE INDEX idx_sid ON pair_session(sid)"
)



pair_frequency = defaultdict(int)

session_pair_count = {}



train_sessions = list(
    iter_sessions(TRAIN_PATH)
)



for session_idx, data in enumerate(
        tqdm(train_sessions,
             desc="Building session index"),
        start=1
):

    events = data["events"]


    session_pairs = set()


    for event in events:

        pid = get_pair_id(
            event["aid"],
            event["type"]
        )


        session_pairs.add(pid)

        pair_frequency[pid] += 1



    session_pair_count[session_idx] = (
        len(session_pairs)
    )


    cursor.executemany(
        "INSERT INTO pair_session VALUES (?, ?)",
        [
            (
                pid,
                session_idx
            )
            for pid in session_pairs
        ]
    )



connection.commit()



# ============================================================
# Swing similarity computation
# ============================================================

session_weight = {
    sid:
        1.0 / np.sqrt(size)
    for sid, size
    in session_pair_count.items()
}



ALPHA = 1.0

TOP_K = 20


pair_similarity_topk = {}



pair_ids = list(
    pair_frequency.keys()
)



for pid_i in tqdm(
        pair_ids,
        desc="Computing Swing similarity"
):

    cursor.execute(
        """
        SELECT sid
        FROM pair_session
        WHERE pid = ?
        """,
        (pid_i,)
    )


    sessions_i = {
        row[0]
        for row in cursor.fetchall()
    }



    if not sessions_i:

        continue



    placeholders = ",".join(
        ["?"] * len(sessions_i)
    )


    cursor.execute(
        f"""
        SELECT DISTINCT pid
        FROM pair_session
        WHERE sid IN ({placeholders})
        AND pid != ?
        """,
        list(sessions_i) + [pid_i]
    )


    candidate_pairs = [
        row[0]
        for row in cursor.fetchall()
    ]



    neighbors = []



    for pid_j in candidate_pairs:


        cursor.execute(
            """
            SELECT sid
            FROM pair_session
            WHERE pid = ?
            """,
            (pid_j,)
        )


        sessions_j = {
            row[0]
            for row in cursor.fetchall()
        }



        common_sessions = (
            sessions_i &
            sessions_j
        )


        if len(common_sessions) < 2:

            continue



        swing_score = 0.0


        common_sessions = list(
            common_sessions
        )


        for i in range(
            len(common_sessions)
        ):

            for j in range(
                i + 1,
                len(common_sessions)
            ):

                u = common_sessions[i]

                v = common_sessions[j]


                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM pair_session
                    WHERE sid = ?
                    AND pid IN (
                        SELECT pid
                        FROM pair_session
                        WHERE sid = ?
                    )
                    """,
                    (u, v)
                )


                common_pairs = (
                    cursor.fetchone()[0]
                )


                swing_score += (
                    session_weight[u] *
                    session_weight[v]
                    /
                    (
                        ALPHA +
                        common_pairs
                    )
                )



        if swing_score > 0:

            neighbors.append(
                (
                    pid_j,
                    swing_score
                )
            )



    neighbors.sort(
        key=lambda x: x[1],
        reverse=True
    )


    pair_similarity_topk[pid_i] = (
        neighbors[:TOP_K]
    )



connection.close()



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



for data in tqdm(
        valid_sessions,
        desc="Generating recommendations"
):

    session_id = data["session"]


    observed_pairs = set()


    for event in data["events"]:

        observed_pairs.add(
            get_pair_id(
                event["aid"],
                event["type"]
            )
        )



    for target_type in TARGET_TYPES:


        scores = defaultdict(float)



        for pid in observed_pairs:


            if pid not in pair_similarity_topk:

                continue



            for neighbor_pid, similarity in (
                pair_similarity_topk[pid]
            ):

                aid, interaction_type = (
                    id_to_pair[neighbor_pid]
                )


                if interaction_type == target_type:

                    scores[aid] += similarity



        ranked_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]



        prediction = (
            " ".join(
                str(aid)
                for aid, _ in ranked_items
            )
            if ranked_items
            else "129004"
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
# Export and evaluation
# ============================================================

submission_path = (
    OUTPUT_DIR /
    "swing_1.csv"
)



submission_df = pd.DataFrame(
    submission_rows
)



submission_df.to_csv(
    submission_path,
    index=False
)



if os.path.exists(DB_PATH):

    os.remove(DB_PATH)



from otto_metrics import evaluate_and_log


evaluate_and_log(
    "swing_1",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
