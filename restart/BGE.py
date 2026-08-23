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
TEST_PATH = ROOT / "valid_history.xlsx"

OUTPUT_DIR = ROOT / "result"
LABELS_PATH = ROOT / "valid_labels.json"
SCORES_PATH = OUTPUT_DIR / "scores.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)



def iter_sessions(path):
    """
    Load session-based interaction sequences.

    Each event contains an item identifier and interaction type.
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
# Hyper-parameters
# ============================================================

WINDOW_SIZE = 3
NUM_NEG_SAMPLES = 5

EMBEDDING_DIM = 64
BATCH_SIZE = 512
EPOCHS = 3



# ============================================================
# Item-type pair encoding
# ============================================================

pair_to_id = {}
id_to_pair = {}

pair_count = 0



def get_pair_id(aid, interaction_type):
    """
    Encode each (item, interaction_type) pair into a unique index.
    """

    global pair_count

    key = (aid, interaction_type)

    if key not in pair_to_id:

        pair_to_id[key] = pair_count
        id_to_pair[pair_count] = key

        pair_count += 1

    return pair_to_id[key]



# ============================================================
# Generate Skip-Gram training pairs
# ============================================================

train_sessions = list(
    iter_sessions(TRAIN_PATH)
)


training_pairs = []


for session_data in tqdm(
        train_sessions,
        desc="Generating training pairs"
):

    events = session_data["events"]

    pair_sequence = []


    for event in events:

        pair_sequence.append(
            get_pair_id(
                event["aid"],
                event["type"]
            )
        )


    length = len(pair_sequence)


    for idx, center_id in enumerate(pair_sequence):

        left = max(
            0,
            idx - WINDOW_SIZE
        )

        right = min(
            length,
            idx + WINDOW_SIZE + 1
        )


        for context_idx in range(left, right):

            if idx != context_idx:

                training_pairs.append(
                    (
                        center_id,
                        pair_sequence[context_idx]
                    )
                )



train_indices = training_pairs

del training_pairs

gc.collect()



# ============================================================
# Skip-Gram dataset with negative sampling
# ============================================================

class SkipGramDataset(Dataset):

    def __init__(
            self,
            pairs,
            num_items,
            num_negative_samples
    ):

        self.pairs = pairs
        self.num_items = num_items
        self.num_negative_samples = num_negative_samples


    def __len__(self):

        return len(self.pairs)


    def __getitem__(self, index):

        center, context = self.pairs[index]


        negatives = np.random.randint(
            0,
            self.num_items,
            size=self.num_negative_samples
        )


        return (
            torch.tensor(
                center,
                dtype=torch.long
            ),

            torch.tensor(
                context,
                dtype=torch.long
            ),

            torch.tensor(
                negatives,
                dtype=torch.long
            )
        )



dataset = SkipGramDataset(
    train_indices,
    pair_count,
    NUM_NEG_SAMPLES
)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)



# ============================================================
# Skip-Gram recommendation embedding model
# ============================================================

class SkipGramModel(nn.Module):

    def __init__(
            self,
            num_items,
            embedding_dim
    ):

        super().__init__()

        self.input_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )

        self.output_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )


        nn.init.xavier_uniform_(
            self.input_embedding.weight
        )

        nn.init.xavier_uniform_(
            self.output_embedding.weight
        )


    def forward(
            self,
            center,
            positive_context,
            negative_context
    ):

        center_embedding = self.input_embedding(center)

        positive_embedding = self.output_embedding(
            positive_context
        )

        negative_embedding = self.output_embedding(
            negative_context
        )


        positive_score = torch.sum(
            center_embedding * positive_embedding,
            dim=1
        )


        positive_loss = (
            -torch.log(
                torch.sigmoid(
                    positive_score
                )
                + 1e-8
            )
        ).mean()



        negative_score = torch.bmm(
            negative_embedding,
            center_embedding.unsqueeze(2)
        ).squeeze(2)


        negative_loss = (
            -torch.log(
                torch.sigmoid(
                    -negative_score
                )
                + 1e-8
            )
        ).mean()



        return positive_loss + negative_loss



    def get_embeddings(self):

        return (
            self.input_embedding
            .weight
            .detach()
            .cpu()
            .numpy()
        )



# ============================================================
# Model optimization
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = SkipGramModel(
    pair_count,
    EMBEDDING_DIM
).to(device)



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



for epoch in range(EPOCHS):

    epoch_loss = 0.0


    for batch in tqdm(
            dataloader,
            desc=f"Training epoch {epoch + 1}/{EPOCHS}"
    ):

        center, positive, negative = [
            x.to(device)
            for x in batch
        ]


        optimizer.zero_grad()


        loss = model(
            center,
            positive,
            negative
        )


        loss.backward()

        optimizer.step()


        epoch_loss += loss.item()



# ============================================================
# Embedding normalization and retrieval preparation
# ============================================================

embedding_matrix = model.get_embeddings()


num_items = embedding_matrix.shape[0]


W_emb = np.ascontiguousarray(
    embedding_matrix,
    dtype=np.float32
)


del embedding_matrix


norm = (
    np.linalg.norm(
        W_emb,
        axis=1,
        keepdims=True
    )
    + 1e-8
)


W_norm = (
    W_emb / norm
).astype(
    np.float32,
    copy=False
)



aid_per_pid = np.empty(
    num_items,
    dtype=np.int64
)


type_masks = {
    t: np.zeros(
        num_items,
        dtype=bool
    )
    for t in [
        "clicks",
        "carts",
        "orders"
    ]
}



for pid in range(num_items):

    aid, interaction_type = id_to_pair[pid]

    aid_per_pid[pid] = aid


    if interaction_type in type_masks:

        type_masks[interaction_type][pid] = True



del model
del dataset
del dataloader

gc.collect()



# ============================================================
# Similarity retrieval
# ============================================================

TARGET_TYPES = [
    "clicks",
    "carts",
    "orders"
]



def top_aids_excluding(
        similarities,
        aids,
        seen_aids,
        top_k=20
):

    n = len(aids)

    if n == 0:

        return []


    seen_size = len(seen_aids)


    candidate_size = min(
        n,
        top_k + seen_size + 128
    )


    while True:


        indices = np.argpartition(
            -similarities,
            candidate_size - 1
        )[:candidate_size]


        indices = indices[
            np.argsort(
                -similarities[indices]
            )
        ]


        results = []


        for idx in indices:

            aid = int(aids[idx])


            if aid not in seen_aids:

                results.append(aid)


                if len(results) >= top_k:

                    return results



        if candidate_size >= n:

            return results


        candidate_size = min(
            n,
            candidate_size + top_k + seen_size + 256
        )



# ============================================================
# Generate recommendations
# ============================================================

submission_rows = []


valid_sessions = list(
    iter_sessions(TEST_PATH)
)



for session_data in tqdm(
        valid_sessions,
        desc="Generating recommendations"
):

    session_id = session_data["session"]


    session_pairs = []

    seen_aids = set()



    for event in session_data["events"]:

        pid = get_pair_id(
            event["aid"],
            event["type"]
        )


        if pid < num_items:

            session_pairs.append(pid)


        seen_aids.add(
            event["aid"]
        )



    if session_pairs:

        session_vector = W_emb[
            session_pairs
        ].mean(axis=0)

    else:

        session_vector = np.zeros(
            EMBEDDING_DIM,
            dtype=np.float32
        )



    session_vector /= (
        np.linalg.norm(session_vector)
        + 1e-8
    )


    similarities = (
        W_norm @ session_vector
    )



    for target_type in TARGET_TYPES:


        mask = type_masks[target_type]


        recommendations = top_aids_excluding(
            similarities[mask],
            aid_per_pid[mask],
            seen_aids,
            20
        )


        prediction = (
            " ".join(
                str(aid)
                for aid in recommendations
            )
            if recommendations
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
# Export results and evaluation
# ============================================================

submission_path = OUTPUT_DIR / "BGE_1.csv"


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
    "BGE_1",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
