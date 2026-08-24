## Day04：共现矩阵与向量召回

### 1. 今天做了什么

今天主要读三个方向的代码：

1. `covisit_fe_preprare.py` 和 `covisit_matrix.py`：理解共现矩阵的构建方式（时间/操作双模式、8h 窗口、方向衰减），并对比开源实现与我之前转移矩阵的差异；
2. `w2v_fe_prepare.py` 和 `w2v_sim.py`：理解 Word2Vec 在推荐场景中的应用方式（向量训练 → Annoy 索引 → 在候选表上生成相似度特征列），而不是用向量做端到端召回；
3. 回顾我自己之前写的 [BGE.py](restart/BGE.py) 和 [airbnb.py](restart/airbnb.py)，对照开源 w2v 的思路，决定将它们改造为特征提取工具（`_v2` 版本），去掉预测部分，仅产出向量供后续使用：[BGE_v2](restart/BGE_v2.py)、[airbnb_v2](restart/airbnb_v2.py)。

同时，三路召回（past_event、item_cf、covisit）的拆解学习已全部完成，今天做了一次汇总对比。

---

### 2. `covisit_fe_preprare.py`：共现矩阵的离线构建

目的：学习如何构建离线共现矩阵并保存结果。

#### 2.1 数据处理细节

**2.1.1 NumPy 数组的使用**

代码中的列表几乎都不是 Python 列表，而是使用 `np.array(list)` 转换为 NumPy 数组。NumPy 数组相比 Python 列表的优势在于：连续内存块、存储数据类型相同、索引速度快、内存占用小、支持向量化运算和 Numba JIT 机器码操作。开源项目使用 `np.array` 的目的是因为后续要在 `@nb.jit` 中做索引和数学运算，数组更快且 Numba 兼容。

**2.1.2 循环加速**

`nb.prange(par_n)` 告诉 Numba：这个循环的每一次迭代之间是相互独立的，可以同时交给多个 CPU 核心去并行执行。

**2.1.3 数据降维**

作者使用 `tail=30` 限制每一个 session 在构建共现矩阵时只关注最近的 30 个 event，去除冗余数据。

**2.1.4 数据合并**

共现矩阵的数据是 `train + test` 历史部分合并的，不可以只使用 `train`。

#### 2.2 共现矩阵的建模与实现

**2.2.1 核心思路**

两个物品只有在出现间隔小于 8h 时，才基于彼此更新共现矩阵的边，从而得到共现矩阵。

**2.2.2 时间权重与操作权重**

共现矩阵分为时间衰减模式和操作模式：

- **时间模式**：关注 `start_time` 在数据集总体时间里的位置，越晚比重越高；
- **操作模式**：关注操作类型，反向回溯乘上 0.3，比正向削弱。

**操作权重模式（OP_WEIGHT）**

正向权重（aid_i → aid_j）：

$$
w_{i \rightarrow j} = \text{ops weights}[op_j]
$$

反向权重（aid_j → aid_i）：

$$
w_{j \rightarrow i} = \text{ops weights}[op_i] \times 0.3
$$

其中：

- `op_i`、`op_j` 分别表示 `aid_i`、`aid_j` 对应的操作类型（0=click, 1=cart, 2=order）
- `ops_weights = [1.0, 6.0, 3.0]`

**时间权重模式（TIME_WEIGHT）**

正向权重（aid_i → aid_j）：

$$
w_{i \rightarrow j} = 1 + 3 \times \frac{t_i + start\_time - T_{\min}}{T_{\max} - T_{\min}}
$$

反向权重（aid_j → aid_i）：

$$
w_{j \rightarrow i} = \left(1 + 3 \times \frac{t_j + start\_time - T_{\min}}{T_{\max} - T_{\min}}\right) \times 0.3
$$

其中：

- `t_i`、`t_j` 分别表示 `aid_i`、`aid_j` 在 session 内的相对时间戳
- `start_time` 为当前 session 的起始时间
- `T_min = 1659304800`（数据集最早时间，约 2022-08-01）
- `T_max = 1662328791`（数据集最晚时间，约 2022-09-05）
- 权重范围为 `[1.0, 4.0]`

**总结**

| 模式 | 正向权重 | 反向权重 |
| :--- | :--- | :--- |
| **操作权重** | `ops_weights[op_j]` | `ops_weights[op_i] * 0.3` |
| **时间权重** | `1 + 3 * norm(t_i)` | `(1 + 3 * norm(t_j)) * 0.3` |
| **方向衰减** | — | 反向统一乘以 0.3 |

**方向衰减 0.3 的含义：** 反向传播（目标 → 历史）的语义强度弱于正向传播（历史 → 目标），因此统一打折。

**2.2.3 嵌套字典存储**

使用嵌套字典来存储多个 session 中共现矩阵的叠加。需要注意：Numba 的 `nopython` 模式下，直接写 `{}` 创建空字典是不被允许的，应该写 `{0: 0.0 for _ in range(0)}`。

**2.2.4 共现矩阵排序**

排序时仍使用 `heapq` 库进行堆排序，但也要用嵌套字典实现。堆排序对每个 `aid` 取分数前 100 进行截断。

**2.2.5 共现矩阵存储**

把之前得到的 Numba 类型共现矩阵字典转换为 Python 普通字典，再存成 `pkl` 文件。转换的原因是 Numba 类型无法被 `pickle` 序列化保存到磁盘，而转换后可序列化成 `pkl` 文件，下次可以直接读回，不需要重新计算。

#### 2.3 开源共现矩阵与我之前转移矩阵的对比

| 对比维度 | 开源共现矩阵 | 我之前转移矩阵 |
|---|---|---|
| 数据降维 | 截断每个 session 前 30 个事件，共现窗口 8h | 使用所有 session 内事件 |
| 权重建模 | 时间 + 操作双模式，反向打 0.3 折 | 1:1 建模 |
| 计算效率 | 分批读取 + 多 CPU 并行 | 一次读取 + 单线程 |

开源实现去除了冗余数据，加强了对短时间内共现窗口内数据的使用；权重建模更具有现实意义；计算效率也远高于我之前的实现。

---

### 3. `covisit_matrix.py`：共现矩阵的召回应用

#### 3.1 数据处理细节

前面共现矩阵存储时对每个 `aid` 保留了分数前 100 的邻居，后续计算时只需要前 50；最终召回结果保留前 100 个。

#### 3.2 共现矩阵召回建模

**3.2.1 操作权重调整**

此处对 `clicks`、`carts`、`orders` 的权重比例调整为 `1:3:6`，与之前历史序列召回中的 `1:10:3` 进行区分。

**3.2.2 指数衰减建模**

基于共现矩阵中每个 `aid` 的 Top-K 邻居列表（按相似度从高到低排序），以 0.1 为统一起点，构建指数衰减权重：

$$
\text{weight}[i] = 2^{\text{linspace}(0.1,\ 1,\ K)[i]} - 1
$$

其中：

- `K` = 当前 session 中不重复历史 aid 的数量
- `i` = 在相似度排序列表中的位置（从 0 开始）
- 权重范围为 `[2^0.1 - 1, 1]`，单调递增

**与历史序列召回（past_event）的对比**

| 对比维度 | **历史序列召回（past_event）** | **共现矩阵召回（co_occur）** |
| :--- | :--- | :--- |
| **排序来源** | session 内历史行为的时间倒序 | 共现矩阵中每个 aid 的相似度降序 |
| **区分模型的方式** | 调整衰减起点 `s`（clicks=0.1，carts/orders=0.5） | 选择不同的共现矩阵（时间模型 → clicks，操作模型 → carts/orders） |
| **衰减起点** | 可变（0.1 或 0.5） | **固定为 0.1** |

**共现矩阵与行为类型的对应关系**

- **时间权重共现矩阵** → `clicks`
- **操作权重共现矩阵** → `carts` / `orders`

---

### 4. 三路召回的汇总

至此，开源项目的 3 路召回的拆解学习已全部结束。以下给出简要概括：

| 召回方法 | 核心思路 |
|---|---|
| **item_cf** | 调用 `implicit` 库的 BM25 模型，对物品进行协同过滤 |
| **past_event** | 基于不同操作类型，构建不同的指数衰减权重 |
| **covisit** | 基于时间/操作分别构建共现矩阵，带入指数衰减 |

最后对三者得到的结果进行去重并汇总，得到召回阶段的最终结果。

> **思考**：既然三路召回中有一路明显比其他好，为什么还要三路融合，而不是直接用一路召回然后精排？如果希望优化召回，是否只完善 past_event 就够了？——这个问题留到后续排序阶段再回头验证，目前先完整跑通开源流程。

---

### 5. 小样本验证结果

在固定的小样本数据集上运行共现矩阵召回，得到结果：

| 方法 | Overall | clicks | carts | orders |
|---|---:|---:|---:|---:|
| covisit | 0.012916 | 0.030120 | 0.014358 | 0.009328 |

共现矩阵召回结果明显偏低。猜测可能原因：小样本下共现 key 覆盖不足，且没有历史 fallback。考虑到当前主线是学习算法及其工程处理，这部分先暂时告一段落，后续放大样本再进一步排查。

---

### 6. `w2v_fe_prepare.py`：Word2Vec 向量训练

#### 6.1 内容概括

主要使用 `gensim.models.Word2Vec` 进行调用，这是最基础的 Skip-Gram + 负采样 I2I 模型。

#### 6.2 数据预处理

合并 `train` 和 `test`，按 session 聚合，提取其中的 `aid` 列为 Python 列表，作为训练 Word2Vec 模型的输入。

#### 6.3 参数设置与解释

```python
w2vec = Word2Vec(sentences=sentences, vector_size=64, window=3, negative=8, ns_exponent=0.2, sg=1, min_count=1, workers=15)
```

| 参数 | 值 | 含义 |
|---|---|---|
| `vector_size` | 64 | 嵌入维度，平衡效果与计算量 |
| `window` | 3 | 上下文窗口，相邻行为关联最强 |
| `negative` | 8 | 负采样数量，足以学到区分度 |
| `ns_exponent` | 0.2 | 负采样分布指数，降低后对低频物品更友好 |
| `sg` | 1 | Skip-Gram，对低频物品更友好 |
| `min_count` | 1 | 小样本下全部保留 |
| `workers` | 15 | 多核并行加速 |

#### 6.4 结果存储

训练好的向量存储在 `w2vec.wv` 中：
- `w2vec.wv.vectors` 维度为 `(物品数, 64)`
- `w2vec.wv.index_to_key` 格式为 `[aid1, aid2, ...]`

使用 `AnnoyIndex` 存储最终结果（`.ann` 文件），树结构比普通列表/字典快很多。树内存储的是归一化后的 64 维向量及 idx 索引，归一化的目的是让向量点积直接等于余弦相似度。开源项目中树的数量设置为 50 棵。

---

### 7. `w2v_sim.py`：向量相似度特征提取

#### 7.1 概括

使用 w2v 结果对 `aid` 提取 3 维特征。

#### 7.2 操作细节

作者把 w2v 模型应用在之前 BM25 协同过滤召回得到的结果上：对召回结果中的 `aid` 与 test 历史物品进行合并，使用 w2v 计算相似度，为每个 `aid` 存储**最高相似度、最低相似度、平均相似度**三个值。

具体流程：按 session 聚合 → 与历史 aid 合并 → 提取相似度 → `explode` 回原始行数。对历史数据提取为空的特殊情况，用 `-1` 填补缺失值。

最终结果呈现为一个 session 对应一个物品、3 个相似度值的列表。

---

### 8. BGE 与 Airbnb：从端到端预测改造为特征提取

#### 8.1 改造背景

我原本的 [BGE.py](restart/BGE.py) 和 [airbnb.py](restart/airbnb.py) 是早期学习算法时尝试写的端到端预测代码。由于全量数据时间复杂度过大，两份代码均未完整运行过。

通过学习开源 w2v 后发现：在竞赛中向量通常不是用来扫全库的，而是用于候选表上的特征。因此决定将这两份代码改造为仅用于特征提取的工具（`_v2` 版本），去除预测部分：[BGE_v2](restart/BGE_v2.py)、[airbnb_v2](restart/airbnb_v2.py)。

#### 8.2 BGE.py 原本框架

**8.2.1 共现边统计**

遍历所有 train session，在窗口内统计有方向性的共现情况。

**8.2.2 训练数据打包**

使用 PyTorch 的 `Dataset` 类，方便更换数据集时只需重构 `Dataset` 并导入 `DataLoader`。数据需通过 `torch.tensor` 从 Python 格式转换为 PyTorch 张量。

**8.2.3 PyTorch 相关**

- `Dataset` 必须包含 `__len__` 和 `__getitem__`，前者必须返回整数
- `DataLoader` 调用：`DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)`
- 继承 `nn.Module` 时必须调用父类初始化；`__init__` 和 `forward` 是必须函数，`forward` 必须返回 `torch.Tensor`

**损失函数**

正样本损失：

$$
L_{\text{pos}} = - \frac{1}{N} \sum_{i=1}^{N} \log \left( \sigma(\mathbf{c}_i \cdot \mathbf{p}_i) + \epsilon \right)
$$

负样本损失：

$$
L_{\text{neg}} = - \frac{1}{N} \sum_{i=1}^{N} \frac{1}{K} \sum_{j=1}^{K} \log \left( \sigma(-\mathbf{n}_{ij} \cdot \mathbf{c}_i) + \epsilon \right)
$$

总损失：

$$
L = L_{\text{pos}} + L_{\text{neg}}
$$

**参数设置**：`WINDOW_SIZE=3`，`NUM_NEG_SAMPLES=5`，`EMBEDDING_DIM=64`，`EPOCHS=3`

**8.2.4 改造后（BGE_v2）**

仅保存向量，不做端到端预测。保存方式为 `.npy + id_mapping.json`，前者为 `(num_items, 64)` 浮点矩阵，按 `pair_id` 顺序存储；后者记录 `pair_id -> (aid, type)` 映射关系。

#### 8.3 Airbnb.py 原本框架

基于 Airbnb 论文思路的改进版 Skip-Gram，在原版 BGE 基础上增加了全局上下文建模。

**8.3.1 最后一个订购行为**

在记录共现矩阵时，增加记录每个 session 的最后一个 `order` 行为（如果有）。具体做法：对 events 逆向遍历，选择第一个 `type` 为 `orders` 的。

**8.3.2 损失函数**

Skip-Gram 损失（局部上下文）：

$$
\mathcal{L}_{\text{skipgram}} = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma(\mathbf{c}_i \cdot \mathbf{p}_i) - \frac{1}{N} \sum_{i=1}^{N} \frac{1}{K} \sum_{j=1}^{K} \log \sigma(-\mathbf{c}_i \cdot \mathbf{n}_{ij})
$$

全局上下文损失（只对有下单行为的 session）：

$$
\mathcal{L}_{\text{global}} = -\frac{1}{M} \sum_{i \in \mathcal{S}} \log \sigma(\mathbf{c}_i \cdot \mathbf{g}_i)
$$

总损失：

$$
\mathcal{L} = \mathcal{L}_{\text{skipgram}} + \lambda \cdot \mathcal{L}_{\text{global}}
$$

其中 `lambda = 0.1`（`LAMBDA_GLOBAL`）

**参数设置**：`WINDOW_SIZE=3`，`NUM_NEG_SAMPLES=5`，`EMBEDDING_DIM=64`，`BATCH_SIZE=512`，`EPOCHS=2`，`LAMBDA_GLOBAL=0.1`

**8.3.3 改造后（Airbnb_v2）**

同样仅保存向量，存储方式与 BGE_v2 一致。

#### 8.4 BGE、Airbnb、w2v 对比

| 对比维度 | 开源 w2v | BGE_v2 | airbnb_v2 |
|----------|----------|--------|-----------|
| **训练库** | `gensim` 一行调用 | PyTorch 自写 Skip-Gram | PyTorch + 全局上下文 |
| **序列单位** | 仅 `aid` | `(aid, type)` pair | `(aid, type)` pair |
| **额外信号** | 无 | 无 | 每个 session 最后一个 `order` |
| **本章产物** | `.model` + `.ann` | `.npy` + `.json` | `.npy` + `.json` |
| **竞赛用途** | `w2v_sim` 特征列 | 备用 embedding 特征 | 备用 embedding 特征 |
| **本章是否端到端召回** | 否 | 否（v2 已去掉预测） | 否（v2 已去掉预测） |

**补充说明**：
- 开源 w2v 是「一行调库」，快速产出向量；BGE/airbnb 是「自己写 PyTorch 训练」，可定制损失和信号，但实现成本更高。
- BGE 和 airbnb 的单位都是 `(aid, type)` pair，和 itemCF / covisit 保持一致，方便后续在候选表上按 type 做相似度计算；w2v 只用 `aid`，不区分行为类型。
- airbnb 在 BGE 基础上增加了全局上下文，利用 session 最后一个 `order` 作为额外信号，让向量不仅学习局部窗口内的共现，也向最终下单行为靠拢。

---

### 9. 今天的核心认识

1. **共现矩阵的构建比转移矩阵更精细**：时间/操作双模式、8h 窗口、方向衰减、并行加速，每个环节都在为更准确和更高效服务。

2. **向量在竞赛中通常不作为端到端召回工具**：开源 w2v 的用法是训练向量 → 建立 Annoy 索引 → 在候选表上生成相似度特征列，而不是扫全库。这个思路也引导了我对 BGE 和 Airbnb 的改造方向。

3. **三路召回的融合是工程上的稳健策略**：即使其中一路明显更强，保留多路召回可以在后续排序阶段提供更丰富的候选集和特征来源。具体取舍留到排序阶段再验证。

4. **小样本下共现矩阵效果受限**：covisit 在小样本上 Recall 明显偏低，可能因为共现 key 覆盖不足。后续放大样本后需要重新评估。

下一步将进入特征工程阶段，开始阅读 `user_fe.py`、`item_fe.py`、`user_item_fe.py` 等特征构建文件。
