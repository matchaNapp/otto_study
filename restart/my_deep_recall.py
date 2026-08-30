import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.xlsx"
VALID_HISTORY_PATH = ROOT / "valid_history.xlsx"
TRAIN_PATHS = [TRAIN_PATH, VALID_HISTORY_PATH]
#合并训练数据集和测试集的历史样本
OUTPUT_DIR = ROOT / "result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- 超参（可按需改）---
MS_24H = 24 * 60 * 60 * 1000
MS_7D = 7 * 24 * 60 * 60 * 1000
MS_21D = 21 * 24 * 60 * 60 * 1000
MAX_SHORT_LEN = 50
MAX_MID_LEN = 100
CART_LEN = 10
NUM_NEG = 5
EMBEDDING_DIM = 64
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
TEMPERATURE = 1.0
#对相似度的缩放系数，这里不进行缩放
CART_TYPES = {"carts", "orders"}
DEBUG_PRINT_BATCH = False  

def iter_sessions(path: Path):
    df = pd.read_excel(path)
    for _, row in df.iterrows():
        #df.iterrows() 每次迭代返回的是一个元组 (index, row)
        events = row["events"]
        if isinstance(events, str):
            events = json.loads(events)
        #判断 events 是不是一个字符串，如果是，就用 json.loads() 把它解析成 Python 对象。
        yield {"session": int(row["session"]), "events": events}


def left_pad(ids, max_len, pad_id=0):
    ids = ids[-max_len:]
    mask = [0.0]*(max_len -len(ids)) + [1.0]*len(ids)
    #细节：这里使用浮点，因为后面涉及除法，避免信息丢失
    padded = [pad_id]*(max_len -len(ids)) + ids




    return padded, mask


def _dedupe_aids_in_order(aids):
    return list(dict.fromkeys(aids))


def split_events_by_time(events, aid_to_id, t_anchor):
    """按 t_anchor 切 short(7d~24h) / mid(21d~7d) / cart / pos(24h 窗)。"""
    hist_events = [e for e in events if int(e["ts"]) <= t_anchor - MS_24H]

    short_raw = []
    mid_raw = []
    cart_raw = []
    for event in hist_events:
        ts = int(event["ts"])
        aid = aid_to_id[int(event["aid"])]
        if t_anchor - MS_7D < ts <= t_anchor - MS_24H:
            short_raw.append(aid)
        elif t_anchor - MS_21D < ts <= t_anchor - MS_7D:
            mid_raw.append(aid)
        if event["type"] in CART_TYPES:
            cart_raw.append(aid)

    pos_raw = []
    for event in events:
        ts = int(event["ts"])
        if t_anchor - MS_24H < ts <= t_anchor:
            pos_raw.append(aid_to_id[int(event["aid"])])

    return (
        _dedupe_aids_in_order(short_raw),
        _dedupe_aids_in_order(mid_raw),
        _dedupe_aids_in_order(cart_raw),
        _dedupe_aids_in_order(pos_raw),
    )


def build_vocab_and_samples(paths):
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
            short_aids, mid_aids, cart_aids, pos_aids = split_events_by_time(
                events, aid_to_id, t_anchor
            )
            if not pos_aids:
                continue

            session_count += 1
            pos_aid_total += len(pos_aids)

            short_ids, short_mask = left_pad(short_aids, MAX_SHORT_LEN)
            mid_ids, mid_mask = left_pad(mid_aids, MAX_MID_LEN)
            cart_ids, cart_mask = left_pad(cart_aids, CART_LEN)

            for pos_id in pos_aids:
                samples.append(
                    {
                        "short": short_ids,
                        "short_mask": short_mask,
                        "mid": mid_ids,
                        "mid_mask": mid_mask,
                        "cart": cart_ids,
                        "cart_mask": cart_mask,
                        "pos": pos_id,
                    }
                )

    id_to_aid = {idx: aid for aid, idx in aid_to_id.items()}
    stats = {
        "sessions": session_count,
        "pos_aids": pos_aid_total,
        "avg_pos_per_session": pos_aid_total / session_count if session_count else 0.0,
    }
    return samples, aid_to_id, id_to_aid, stats


class DeepRecallDataset(Dataset):
    def __init__(self, samples, num_items, num_neg):
        self.samples = samples
        self.num_items = num_items
        self.num_neg = num_neg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        neg = np.random.randint(0, self.num_items, size=self.num_neg)
        #负样本随机选取而不是增添策略也是为了添加随机噪声，防止过拟合
        while (neg == row["pos"]).any():
            neg = np.random.randint(0, self.num_items, size=self.num_neg)
        #细节：防止出现正负样本数量相同的情况
        return (
            torch.tensor(row["short"], dtype=torch.long),
            torch.tensor(row["short_mask"], dtype=torch.float32),
            torch.tensor(row["mid"], dtype=torch.long),
            torch.tensor(row["mid_mask"], dtype=torch.float32),
            torch.tensor(row["cart"], dtype=torch.long),
            torch.tensor(row["cart_mask"], dtype=torch.float32),
            torch.tensor(row["pos"], dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


class DeepRecallModel(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        #就是告诉机器，希望他训练某维度的emb，从而使得loss下降，这段代码会自动生成对应维度的self.emb.weight
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        nn.init.xavier_uniform_(self.emb.weight)

    def masked_mean(self, ids, mask):
        emb = self.emb(ids)
        mask = mask.unsqueeze(-1)
        #细节：为了能和 emb 张量做逐元素乘法，需要在增加一个维度，
        # mask.unsqueeze(-1)是在末尾增加一个维度。
        summed = (emb*mask).sum(dim=1)
        #在第一维度上求和，变成(B, D)，原因是：我们只需要得到每一个session的emb，
        # 不需要得到分别aid的emb，因此对选中emb进行求和，刻画出该session对应维度的emb
        denom = mask.sum(dim=1).clamp_min(1.0)
         #小于一的除以1，防止mask全部都是0的异常情况出现
        return summed / denom#维度是(B, D)

    def forward(self, short, short_mask, mid, mid_mask, cart, cart_mask, pos, neg):
        # short=7d~24h, mid=21d~7d, cart=历史加购（均排除末 24h 泄漏窗）
        h_s = self.masked_mean(short, short_mask)
        h_m = self.masked_mean(mid, mid_mask)
        h_c = self.masked_mean(cart, cart_mask)
        h_u = self.mlp(torch.cat([h_s, h_m, h_c], dim=1))


        e_pos = self.emb(pos)
        e_neg = self.emb(neg)


        cos_pos = F.cosine_similarity(h_u, e_pos, dim=1) / TEMPERATURE
        #使用pytorch进行余弦相似度计算，而非手写
        cos_neg = F.cosine_similarity(h_u.unsqueeze(1), e_neg, dim=2) / TEMPERATURE
        #维度的不匹配用unsqueeze广播机制解决，即逐个计算，得到的cos_neg维度是(B,neg_num)
        # 多正样本已在 build_vocab_and_samples 中展开为多条 sample；
        # batch 内 pos 形状为 (B,)，故 cos_pos 用 dim=1。
        # neg 形状为 (B, NUM_NEG)，故 cos_neg 用 unsqueeze + dim=2。
        pos_loss = -torch.log(torch.sigmoid(cos_pos) + 1e-8).mean()
        neg_loss = -torch.log(torch.sigmoid(-cos_neg) + 1e-8).mean()
        return pos_loss + neg_loss

    def get_item_vectors(self):
        return self.emb.weight.data.cpu().numpy()
        #把训练好的商品向量从 GPU/内存中提取出来，转换成 NumPy 格式


def main():
    print("[deep_recall] building vocab and samples (time windows)...")
    samples, aid_to_id, id_to_aid, stats = build_vocab_and_samples(TRAIN_PATHS)
    num_items = len(aid_to_id)
    print(f"  num_aids={num_items:,}, samples={len(samples):,}")
    print(
        f"  sessions={stats['sessions']:,}, "
        f"pos_aids={stats['pos_aids']:,}, "
        f"avg_pos/session={stats['avg_pos_per_session']:.2f}"
    )

    dataset = DeepRecallDataset(samples, num_items, NUM_NEG)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if DEBUG_PRINT_BATCH:
        batch = next(iter(dataloader))
        names = [
            "short(7d~24h)",
            "short_mask",
            "mid(21d~7d)",
            "mid_mask",
            "cart",
            "cart_mask",
            "pos(24h窗)",
            "neg",
        ]
        print("[debug] batch shapes:")
        for name, t in zip(names, batch):
            print(f"  {name}: {tuple(t.shape)} dtype={t.dtype}")
        print("[debug] short ids (row0):", batch[0][0].tolist())
        print("[debug] short mask (row0):", batch[1][0].tolist())
        print("[debug] mid effective len (row0):", int(batch[3][0].sum().item()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepRecallModel(num_items, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[deep_recall] training (device={device})...")
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

    vectors_path = OUTPUT_DIR / "deep_item_vectors.npy"
    mapping_path = OUTPUT_DIR / "deep_id_mapping.json"
# 时间窗 short/mid + cart 三段池化 → 用户向量；正样本为末 event 前 24h 内各 aid
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
