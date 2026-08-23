import json
import numpy as np
from collections import defaultdict
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


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



def iter_sessions(path):
    """
    Load session-based interaction sequences.

    Each interaction contains:
    item id, interaction type and timestamp.
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

DTYPE = np.float32

WINDOW_SIZE = 3

NUM_NEG_SAMPLES = 5

EMBEDDING_DIM = 64

BATCH_SIZE = 512

EPOCHS = 2

LAMBDA_GLOBAL = 0.1



# ============================================================
# Item-type pair representation
# ============================================================

pair_to_id = {}

id_to_pair = {}

pair_count = 0



def get_pair_id(aid, interaction_type):
    """
    Encode (item, interaction type) pair into integer index.
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
# Generate local-context training samples
# ============================================================

training_pairs = []

session_order_context = {}



train_sessions = list(
    iter_sessions(TRAIN_PATH)
)



for session_data in tqdm(
        train_sessions,
        desc="Generating training samples"
):

    session_id = session_data["session"]

    events = session_data["events"]


    pair_sequence = []


    for event in events:

        pid = get_pair_id(
            event["aid"],
            event["type"]
        )


        pair_sequence.append(pid)



    # Extract final order behavior as global preference signal

    for event in reversed(events):

        if event["type"] == "orders":

            session_order_context[session_id] = (
                get_pair_id(
                    event["aid"],
                    event["type"]
                )
            )

            break



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
                        pair_sequence[context_idx],
                        session_id
                    )
                )



train_indices = training_pairs


del training_pairs

gc.collect()



# ============================================================
# Dataset with global order context
# ============================================================

class GlobalContextSkipGramDataset(Dataset):

    def __init__(
            self,
            pairs,
            num_items,
            num_negative_samples,
            order_context
    ):

        self.pairs = pairs

        self.num_items = num_items

        self.num_negative_samples = (
            num_negative_samples
        )

        self.order_context = order_context



    def __len__(self):

        return len(self.pairs)



    def __getitem__(self, index):

        center, context, session_id = (
            self.pairs[index]
        )


        negatives = np.random.randint(
            0,
            self.num_items,
            size=self.num_negative_samples
        )


        has_global = (
            session_id in self.order_context
        )


        order_item = self.order_context.get(
            session_id,
            0
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
            ),

            torch.tensor(
                has_global,
                dtype=torch.bool
            ),

            torch.tensor(
                order_item,
                dtype=torch.long
            )
        )



dataset = GlobalContextSkipGramDataset(
    train_indices,
    pair_count,
    NUM_NEG_SAMPLES,
    session_order_context
)



dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)



# ============================================================
# Global-context Skip-Gram model
# ============================================================

class GlobalContextSkipGramModel(nn.Module):

    def __init__(
            self,
            num_items,
            embedding_dim,
            lambda_global
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


        self.lambda_global = lambda_global



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
            negative_context,
            has_global,
            order_item
    ):

        center_embedding = (
            self.input_embedding(center)
        )


        positive_embedding = (
            self.output_embedding(
                positive_context
            )
        )


        negative_embedding = (
            self.output_embedding(
                negative_context
            )
        )



        positive_score = torch.sum(
            center_embedding *
            positive_embedding,
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



        skipgram_loss = (
            positive_loss +
            negative_loss
        )



        if has_global.any():

            mask = has_global.bool()


            current_embedding = (
                center_embedding[mask]
            )


            order_embedding = (
                self.input_embedding(
                    order_item[mask]
                )
            )


            global_score = torch.sum(
                current_embedding *
                order_embedding,
                dim=1
            )


            global_loss = (
                -torch.log(
                    torch.sigmoid(
                        global_score
                    )
                    + 1e-8
                )
            ).mean()



            return (
                skipgram_loss +
                self.lambda_global *
                global_loss
            )


        return skipgram_loss



    def get_embeddings(self):

        return (
            self.input_embedding
            .weight
            .detach()
            .cpu()
            .numpy()
        )



# ============================================================
# Model training
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



model = GlobalContextSkipGramModel(
    pair_count,
    EMBEDDING_DIM,
    LAMBDA_GLOBAL
).to(device)



optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



for epoch in range(EPOCHS):

    for batch in tqdm(
            dataloader,
            desc=f"Training epoch {epoch+1}/{EPOCHS}"
    ):

        (
            center,
            positive,
            negative,
            has_global,
            order_item
        ) = [
            x.to(device)
            for x in batch
        ]


        optimizer.zero_grad()


        loss = model(
            center,
            positive,
            negative,
            has_global,
            order_item
        )


        loss.backward()


        optimizer.step()



# ============================================================
# Embedding extraction
# ============================================================

embedding_matrix = (
    model.get_embeddings()
)



embedding_lookup = {
    idx: embedding_matrix[idx]
    for idx in range(pair_count)
}



del (
    model,
    dataset,
    dataloader
)

gc.collect()



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

    events = session_data["events"]


    session_embeddings = []


    for event in events:

        pid = get_pair_id(
            event["aid"],
            event["type"]
        )


        if pid in embedding_lookup:

            session_embeddings.append(
                embedding_lookup[pid]
            )



    if session_embeddings:

        session_vector = np.mean(
            session_embeddings,
            axis=0
        )

    else:

        session_vector = np.zeros(
            EMBEDDING_DIM,
            dtype=DTYPE
        )



    for target_type in TARGET_TYPES:


        scores = defaultdict(float)



        for pid, vector in embedding_lookup.items():

            aid, interaction_type = (
                id_to_pair[pid]
            )


            if interaction_type != target_type:

                continue



            similarity = (
                np.dot(
                    session_vector,
                    vector
                )
                /
                (
                    np.linalg.norm(session_vector)
                    *
                    np.linalg.norm(vector)
                    +
                    1e-8
                )
            )


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
    "airbnb_1.csv"
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
    "airbnb_1",
    submission_df,
    LABELS_PATH,
    SCORES_PATH
)
