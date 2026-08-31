## Day05：序列召回与自注意力建模

### 1. 今天做了什么

今天主要读两个方向的内容：

1. `GRU4Rec.ipynb`：了解作者在序列建模上的尝试，特别是数据预处理的方式和对序列模型输入格式的处理；
2. 基于前期的数据分析结果，自己动手实现两条深度召回路线：
   - [`my_deep_recall.py`](../restart/my_deep_recall.py)：将 session 按时间切分为短/中/长三段，分别做平均池化后拼接，用 MLP 输出用户向量，与正/负样本计算余弦对比损失；
   - [`my_sasrec.py`](../restart/my_sasrec.py)：在序列建模中加入自注意力机制，使用因果 mask 禁止未来信息泄露，用点积对比损失训练。

这两条路线分别对应了“用时间分段做兴趣表征”和“用 Transformer 做序列建模”两种思路，为后续排序阶段提供可接入候选表的深度特征。

---

### 2. `GRU4Rec.ipynb`：开源项目中的序列建模尝试

#### 2.1 文件内容分析
这是 `others` 文件夹下的 notebook 文件，里面包含了作者尝试过但是没有并入模型主路的模型。和我之前思路不同，我是把神经网络用于空间建模从而把提取出的向量用于点积得到相似度排序，而这里是使用序列模型，对数据进行预处理之后，尝试调用 GRU4Rec 做 next-item 召回。
虽然没有并入最终模型，但其中数据预处理及使用神经网络对序列建模的思想是值得学习的。

#### 2.1 数据预处理

A.对数据进行降维，仅保留train最近一周的数据
B.合并train和test历史，让需要预测的对应session数据也用于训练
C.对时间进行放大(*1e9),避免同一秒多个事件，同时避免小数，RecBole只可以输入整数
D.对原始数据进行排序，因为.inter 文件要求数据按 (USER_ID, ITEM_ID, TIME_FIELD) 排序


### 3. 基础神经网络尝试

基于之前已经有空间神经网络建模提取的两份向量([airbnb](../restart/airbnb.py), [BGE](../restart/BGE.py))作为提取特征的备用数据，但是还没有尝试过序列建模；同时之前我的代码里面，没有用上时间信息，因此下面尝试构建使用了时间信息的简单神经网络序列模型，最终结果呈现在[`my_deep_recall.py`](../restart/my_deep_recall.py)。

#### 3.1 思路阐述
之前已经尝试过在粗排阶段把时间信息用于构建序列衰减模型，对操作进行赋权，直接用于召回。因此这里就不再进行已经尝试过的东西，而是希望把时间信息用于新方面。

在过往的开源项目学习中，作者通常在数据预处理阶段，使用时间信息进行截断，这种思路是可取的，这也是刻画短期兴趣最直接的方式。按照这种思路，可以从每一个session中，提取出短期，中期以及长期兴趣aid 分别用于神经网络的训练。 

本节后续将首先对各个session的aid长度以及ts跨度进行分析，然后再决定是使用event还是ts对序列进行切割。同时，将分割出来的短期和长期aid序列，和所有加购/买单行为提取出来的序列一同，作为输入全连接神经网络的3个维度用于训练向量。

因为本节主要目的是对序列进行不同切割后一起输入神经网络进行尝试，因此对于神经网络选择方面，采用的是mask平均池化+2个MLP层+1层Relu作为激活函数的简单神经网络，不构建复杂模型。

#### 3.2 数据分析和预处理

**3.2.1 数据合并**

将 train 的数据和 test 数据集的 history 进行合并。

**3.2.2 序列切割**

在开始之前，首先对各个 session 的 aid 长度和 ts 跨度进行了分析，具体过程在 [`my_deep_recall_data_test.ipynb`](../restart/my_deep_recall_data_test.ipynb) 中。

因此，根据里面的分析，对数据的短期兴趣和中期兴趣分别以最有一个ts往前回溯前一周、前21天进行提取。

此外，基于event长度的各分位数，设置短期兴趣aid最多为50个，中期最多为100个。具体的，提取数据的时候，序列仅仅保留aid编号信息，其他信息不进行存储。

基于surprise算法的启发，对订购信息特别留意，模型中使用的是最后一个往回溯源的前十个aid信息，此处订购aid包含了加购和买单序列，type信息只有在提取加购/买单序列的时候被使用。

**3.2.3 正/负样本设置**

设置最后一个ts往前24h为正样本范围，负样本和正样本的比例为5:1。同时为了防止信息泄露，基于loss function是由正/负样本和session的相似度构建的模型，此处采取提取的序列里去除正样本的操作。具体操作细节为，24h 窗内多个正样本 按 aid 去重后展开成多行，每一次训练的时候只有每行 1 个正样本 + 随机 5 个负样本。同时，历史序列排除末 24h 防泄漏。

此外，为了负样本的作用是增加随机噪声，防止过拟合，因此对于负样本没有另外构建模型，而是选择了在除了正样本之外的aid里面随机选取。

#### 3.3 损失函数

采用的是经典的使用正负样本刻画交叉熵的形式，具体公式如下：

**总公式**

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \sigma\left( \frac{\text{cos}(h_{u_i}, e_{pos_i})}{\tau} \right) - \frac{1}{B} \sum_{i=1}^{B} \frac{1}{N} \sum_{j=1}^{N} \log \sigma\left( -\frac{\text{cos}(h_{u_i}, e_{neg_{ij}})}{\tau} \right)
$$

| 符号 | 含义 |
|------|------|
| $B$ | 批次大小 |
| $N$ | 每个样本的负采样数量（=5） |
| $h_{u_i}$ | 第 $i$ 个用户的向量表示 |
| $e_{pos_i}$ | 第 $i$ 个正样本商品的向量表示 |
| $e_{neg_{ij}}$ | 第 $i$ 个用户的第 $j$ 个负样本商品向量 |
| $\tau$ | 温度系数 |
| $\sigma(\cdot)$ | Sigmoid 激活函数 |
| $\text{cos}(a, b)$ | 余弦相似度 |

#### 3.4 实现细节

主要使用pytorch进行模型搭建。

**3.3.1 平均池化**

因为各序列有效长度可能和给定的最大值不同，对于确实部分使用从左边开始进行零填充，这部分用mask实现。

在平均池化的时候，为了维度匹配，需要用到.unsqueeze()函数，对mask增加一个维度，从而实现可以和emb进行逐元素相乘，即对mask广播后按有效位求平均.

特别的，求平均的时候，分母需要使用.clamp_min(1.0)，防止mask全部都是0的异常情况出现。

**3.3.2 Loss 中的相似度计算**

因为在实现的时候，虽然正样本有多个，但是输入方式是，读取一个正样本，随机配对5各负样本，每一次训练独立同时进行，因此，计算的具体实现代码如下：

```python
cos_pos = F.cosine_similarity(h_u, e_pos, dim=1) / TEMPERATURE
cos_neg = F.cosine_similarity(h_u.unsqueeze(1), e_neg, dim=2) / TEMPERATURE
```
dim=1和dim=2的区别在于，对于负样本，维度的不匹配用unsqueeze广播机制解决，即逐个计算，得到的cos_neg维度是(Batch number,neg num),但是正样本只有一个，所以是一维。


#### 3.5 参数设置

超参数设置如下：

| 参数 | 取值 | 含义 |
|------|------|------|
| `MAX_SHORT_LEN` | 50 | 短期兴趣序列最大长度 |
| `MAX_MID_LEN` | 100 | 中期兴趣序列最大长度 |
| `CART_LEN` | 10 | 加购/买单序列长度 |
| `NUM_NEG` | 5 | 负采样数量 |
| `EMBEDDING_DIM` | 64 | 嵌入维度 |
| `BATCH_SIZE` | 256 | 批次大小 |
| `EPOCHS` | 10 | 训练轮数 |
| `LR` | 1e-3 | 学习率 |
| `TEMPERATURE` | 1.0 | 温度系数 |

神经网络结构：

```python
self.mlp = nn.Sequential(
    nn.Linear(embedding_dim * 3, 128),
    nn.ReLU(),
    nn.Linear(128, embedding_dim),
)
```

#### 3.6 模型汇总

```
Embedding 共享 → 三段 masked_mean → concat → MLP → h_u → 与正/负 item 算余弦对比损失
```


### 4. [`my_sasrec.py`](../restart/my_sasrec.py)：自注意力序列建模

前面尝试了时间序列用于神经网络得到向量，但其实没有真正步入序列神经网络。对于后者，比较经典的就是 Transformer。为了仅添加其中重要部分进行不同策略结果对照，下面仅对模型添加自注意力机制，最终完成的代码在 [`my_sasrec.py`](../restart/my_sasrec.py) 中。

#### 4.1 数据预处理

**4.1.1 数据拼接**

首先对 train 和 test 的 history 进行合并。

**4.1.2 序列读取**

参考前面基础神经网络模型中对数据进行测试得到的 event 长度均值（62.87），取 SASRec 模型的读入序列长度 `SEQ_LEN = 50`。读取范围为 session 从实际出现的第一个正样本时间戳开始往前回溯。

**4.1.3 正/负样本选取规则**

正样本读取规则与上一个模型相同：从最后一个 aid 对应 ts 往前 24h 的内容；负样本与正样本比例为 5:1。24h 内正样本去重后展开多行，每行 1 个正样本 + 5 个负样本。

#### 4.2 实现细节

使用 PyTorch 进行实现。

**4.2.1 因果 Mask**

Attention 模型的训练规则是：训练时只能基于当前位置之前出现的历史数据进行训练，即训练计算可并行，但靠因果 mask 禁止看未来，效果上像按时间因果。因此必须确保处理位置后面的信息不泄露。

对于 attention 模型，数据矩阵的上三角位置表示未来位置，因此需要设置上三角 mask 进行屏蔽：

```python
return torch.triu(
    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
    diagonal=1,
)
```

`torch.triu` 本身就是上三角矩阵，`diagonal=1` 表示主对角线不需要置零（处理对象本身可被看见）。Attention 使用该 mask 的逻辑是：模型在计算 `Q@K^T` 得到分数后，被 mask 的位置会被替换成 `-inf`，Softmax 后权重变为 0。

**4.2.2 位置编码**

Transformer 的 Attention 机制是**无序的**，只关心相关性，因此需要有位置编码，让模型知道位置和普通的 aid 是不同的。训练时需要将序列的 embedding 和位置 embedding 相加：

```python
x = self.emb(seq) + self.pos_emb(torch.arange(l, device=seq.device))
```

序列 embedding 和位置 embedding 的维度必须相同（均为 `EMBEDDING_DIM`）。在 PyTorch 中，维度相同时，会自动触发广播机制相加，不需要额外处理。

**4.2.3 时间优化**

模型训练时，不返回注意力权重，只返回向量，可以加快计算速度：

```python
self.attn(need_weights=False)
```

**4.2.4 损失函数**

与 DeepRecall 的余弦相似度不同，SASRec 使用**点积**作为相似度度量。

**总公式**

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \sigma\left( \langle h_{u_i}, e_{pos_i} \rangle \right) - \frac{1}{B} \sum_{i=1}^{B} \frac{1}{N} \sum_{j=1}^{N} \log \sigma\left( -\langle h_{u_i}, e_{neg_{ij}} \rangle \right)
$$

| 符号 | 含义 |
|------|------|
| $B$ | 批次大小 |
| $N$ | 每个样本的负采样数量 |
| $h_{u_i}$ | 第 $i$ 个用户的向量表示 |
| $e_{pos_i}$ | 第 $i$ 个正样本商品的向量表示 |
| $e_{neg_{ij}}$ | 第 $i$ 个用户的第 $j$ 个负样本商品向量 |
| $\langle a, b \rangle$ | 点积（内积）相似度 |

**正样本损失**

$$
\mathcal{L}_{pos} = -\frac{1}{B} \sum_{i=1}^{B} \log \sigma\left( \langle h_{u_i}, e_{pos_i} \rangle \right)
$$

**负样本损失**

$$
\mathcal{L}_{neg} = -\frac{1}{B} \sum_{i=1}^{B} \frac{1}{N} \sum_{j=1}^{N} \log \sigma\left( -\langle h_{u_i}, e_{neg_{ij}} \rangle \right)
$$

**代码实现**

```python
pos_score = (h_u * e_pos).sum(dim=1)
neg_score = torch.bmm(e_neg, h_u.unsqueeze(2)).squeeze(2)

pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).mean()

loss = pos_loss + neg_loss
```

此损失函数为**二元交叉熵损失**的向量化形式，是 SASRec 论文中使用的标准损失函数。

#### 4.3 参数设置

| 参数 | 取值 | 含义 |
|------|------|------|
| `SEQ_LEN` | 50 | 序列最大长度 |
| `NUM_NEG` | 5 | 负采样数量 |
| `EMBEDDING_DIM` | 64 | 嵌入维度 |
| `NUM_HEADS` | 2 | 注意力头数 |
| `BATCH_SIZE` | 256 | 批次大小 |
| `EPOCHS` | 10 | 训练轮数 |
| `LR` | 1e-3 | 学习率 |

### 5. 今天的核心认识

1. **序列模型的数据预处理要求更严格**：GRU4Rec 的尝试说明，时间精度（放大 1e9）、数据排序、格式转换对序列建模的成功至关重要，这是序列模型与统计召回在工程实现上的最大区别。

2. **时间分段是刻画短期兴趣的直接方式**：通过短/中/长三段分别池化再拼接，可以在不引入复杂序列模型的前提下，用全连接网络捕捉不同时间尺度下的用户兴趣变化，是深度召回的一条轻量级基线。

3. **因果 mask 是 Transformer 序列建模的关键**：通过 `torch.triu` 生成上三角 mask 禁止未来信息泄露，保证训练时每个位置只能看到自己和之前的信息，这是 Transformer 从 NLP 迁移到推荐序列任务的核心适配点。

4. **两种损失函数的定位不同**：
   - DeepRecall 使用**余弦相似度**，适合衡量向量方向的一致性，对向量模长不敏感；
   - SASRec 使用**点积**，是论文中的标准做法，对向量模长敏感，更适合匹配任务。

5. **两条深度路线已就位**：[`my_deep_recall.py`](../restart/my_deep_recall.py)（时间分段 + MLP）和 [`my_sasrec.py`](../restart/my_sasrec.py)（自注意力）分别提供了两种不同复杂度的序列建模方案，后续均可接入候选表参与排序，形成“统计召回 + 深度特征”的双层架构。
