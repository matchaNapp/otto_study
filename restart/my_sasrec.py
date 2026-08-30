import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.xlsx"
VALID_HISTORY_PATH = ROOT / "valid_history.xlsx"
TRAIN_PATHS = [TRAIN_PATH, VALID_HISTORY_PATH]
OUTPUT_DIR = ROOT / "result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MS_24H = 24 * 60 * 60 * 1000
SEQ_LEN = 50
NUM_NEG = 5
EMBEDDING_DIM = 64
NUM_HEADS = 2
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
DEBUG_PRINT_BATCH = True


def iter_sessions(path: Path):
    df = pd.read_excel(path)
    for _, row in df.iterrows():
        events = row["events"]
        if isinstance(events, str):
            events = json.loads(events)
        yield {"session": int(row["session"]), "events": events}


def left_pad(ids, max_len, pad_id=0):
    ids = ids[-max_len:]
    mask = [0.0] * (max_len - len(ids)) + [1.0] * len(ids)
    padded = [pad_id] * (max_len - len(ids)) + ids
    return padded, mask


def build_vocab_and_samples(paths):
    """seq=排除末 24h 后的 last-20；pos=末 24h 窗内 aid（去重后每 aid 一行）。"""
    aid_to_id = {}
    samples = []
    session_count = 0
    pos_aid_total = 0

    for path in paths:
        for data in tqdm(list(iter_sessions(path)), desc=f"build samples ({path.name})"):
            events = sorted(data["events"], key=lambda e: int(e["ts"]))
            if len(events) < 2:
                continue

            for event in events:
                aid = int(event["aid"])
                if aid not in aid_to_id:
                    aid_to_id[aid] = len(aid_to_id)

            t_anchor = int(events[-1]["ts"])
            hist_aids = [
                aid_to_id[int(e["aid"])]
                for e in events
                if int(e["ts"]) <= t_anchor - MS_24H
            ]
            pos_aids = list(
                dict.fromkeys(
                    aid_to_id[int(e["aid"])]
                    for e in events
                    if t_anchor - MS_24H < int(e["ts"]) <= t_anchor
                )
            )
            if not pos_aids:
                continue

            session_count += 1
            pos_aid_total += len(pos_aids)
            seq_ids, seq_mask = left_pad(hist_aids, SEQ_LEN)

            for pos_id in pos_aids:
                samples.append({"seq": seq_ids, "mask": seq_mask, "pos": pos_id})

    id_to_aid = {idx: aid for aid, idx in aid_to_id.items()}
    stats = {
        "sessions": session_count,
        "pos_aids": pos_aid_total,
        "avg_pos_per_session": pos_aid_total / session_count if session_count else 0.0,
    }
    return samples, aid_to_id, id_to_aid, stats


class SASRecDataset(Dataset):
    def __init__(self, samples, num_items, num_neg):
        self.samples = samples
        self.num_items = num_items
        self.num_neg = num_neg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        neg = np.random.randint(0, self.num_items, size=self.num_neg)
        while (neg == row["pos"]).any():
            neg = np.random.randint(0, self.num_items, size=self.num_neg)
        return (
            torch.tensor(row["seq"], dtype=torch.long),
            torch.tensor(row["mask"], dtype=torch.float32),
            torch.tensor(row["pos"], dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


class SASRecModel(nn.Module):
    def __init__(self, num_items, seq_len, embedding_dim, num_heads):
        super().__init__()
        self.seq_len = seq_len
        self.emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, embedding_dim)
    #因为Transformer 的 Attention 机制是无序的，只关心相关性，因此需要有pos emb，让模型知道pos的位置和普通的aid是不同的
        self.attn = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True
        )
    #num_heads 的数目，代表以多少个角度对数据进行打分处理，一般取决于向量维度，过大会导致过拟合，64 dim对于2 heads是合理的
    #batch_first=True 表示输入的batch是第一维度,输出也是
        self.ln = nn.LayerNorm(embedding_dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def causal_mask(self, seq_len, device):
        """手写练习核心：上三角=True 表示未来位置，attention 时必须屏蔽。"""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        #torch.triu是生成上三角矩阵，diagonal=1是把对角线本身也置零，变成严格上三角。
        # 对于模型来说，上三角=True 表示未来位置，attention 时必须屏蔽

    def encode_session(self, seq, mask):
        b, l = seq.shape
        x = self.emb(seq) + self.pos_emb(torch.arange(l, device=seq.device))
    #在 PyTorch 中，维度数不同的张量相加时，会自动触发广播，两者右对其

        key_padding = mask == 0  # True = pad 位置忽略

        attn_mask = self.causal_mask(l, seq.device)
    #attn_mask：模型在计算 Q@K^T 得到分数后，会把这些0的位置的结果替换成 -inf，Softmax 后权重变为 0。
    # 这样就实现了因果（Causal）
        out, _ = self.attn(
            x, x, x,
        #自注意力，query, key, value都是x
            attn_mask=attn_mask,
            key_padding_mask=key_padding,
            need_weights=False,
        #不返回注意力，只返回向量，加快计算速度
        )
        out = self.ln(out)
        # 取每个样本最后一个有效位置（mask 最右一个 1）
        idx = mask.long().sum(dim=1).clamp_min(1) - 1  # (B,)
        #这里是因为，创建mask的时候用的是布尔，要计算有多少有效数字，要先转换格式，-1是转换成有效索引
        batch_idx = torch.arange(b, device=seq.device)
        #这是为了一行一行进行
        return out[batch_idx, idx, :]


    def forward(self, seq, mask, pos, neg):
        h_u = self.encode_session(seq, mask)
        e_pos = self.emb(pos)
        e_neg = self.emb(neg)

        pos_score = (h_u * e_pos).sum(dim=1)
        neg_score = torch.bmm(e_neg, h_u.unsqueeze(2)).squeeze(2)

        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()
        return pos_loss + neg_loss

    def get_item_vectors(self):
        return self.emb.weight.data.cpu().numpy()


def main():
    print("[sasrec] building vocab and samples (pos=24h window)...")
    samples, aid_to_id, id_to_aid, stats = build_vocab_and_samples(TRAIN_PATHS)
    num_items = len(aid_to_id)
    print(f"  num_aids={num_items:,}, samples={len(samples):,}")
    print(
        f"  sessions={stats['sessions']:,}, "
        f"pos_aids={stats['pos_aids']:,}, "
        f"avg_pos/session={stats['avg_pos_per_session']:.2f}"
    )

    dataset = SASRecDataset(samples, num_items, NUM_NEG)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if DEBUG_PRINT_BATCH:
        batch = next(iter(dataloader))
        names = ["seq(exclude-24h last-20)", "mask", "pos(24h窗)", "neg"]
        print("[debug] batch shapes:")
        for name, t in zip(names, batch):
            print(f"  {name}: {tuple(t.shape)} dtype={t.dtype}")
        print("[debug] sample seq (row0):", batch[0][0].tolist())
        print("[debug] sample mask (row0):", batch[1][0].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SASRecModel(num_items, SEQ_LEN, EMBEDDING_DIM, NUM_HEADS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[sasrec] training (device={device})...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            batch = [x.to(device) for x in batch]
            optimizer.zero_grad()
            loss = model(*batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1} avg loss: {total_loss / len(dataloader):.4f}")

    vectors_path = OUTPUT_DIR / "sasrec_item_vectors.npy"
    mapping_path = OUTPUT_DIR / "sasrec_id_mapping.json"

    item_vectors = np.ascontiguousarray(model.get_item_vectors(), dtype=np.float32)
    id_mapping = {str(idx): int(aid) for idx, aid in id_to_aid.items()}

    assert item_vectors.shape == (num_items, EMBEDDING_DIM)
    assert len(id_mapping) == num_items

    np.save(vectors_path, item_vectors)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f)

    print(f"  saved {vectors_path} shape={item_vectors.shape}")
    print(f"  saved {mapping_path} n={len(id_mapping):,}")

    del model, dataloader, dataset, item_vectors
    gc.collect()


if __name__ == "__main__":
    main()
