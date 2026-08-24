
# Day03：第一次算法改进与开源召回框架阅读

## 1. 今天做了什么

今天主要做两件事：

1. 对我自己之前写的 [itemCF](../restart/itemCF.py) 和 [surprise](../restart/surprise.py) 代码进行第一次改进，加入时间衰减系数和行为类型权重，分别生成 [itemCF_v2](../restart/itemCF_v2.py) 和 [surprise_v2](../restart/surprise_v2.py)，在小样本验证集上测试效果；
2. 开始阅读开源项目的 `itemcf_data_preprare.py` 和 `item_cf.py`，理解作者如何实现基于物品的协同过滤召回，并与我自己的实现进行对比。


## 2. 初版代码以及对应改进

前一天学习了开源项目的基于用户时间序列建模，而这部分时间信息在我之前的模型构建是完全缺失的。因此在学习了开源思路后，我尝试将建模思路用于我原本的 itemCF 和 surprise 模型，并对这两份代码进行一些修改。在此之前，下面将首先回顾我在拆解项目前写的两份文件。

---

### 2.1 itemCF.py 文件阅读

#### 2.1.1 思路概括

初次对 itemCF 标准算法进行尝试的时候，因为在减小训练 session 的情况下仍然内存爆炸，所以在当时对于降维没有思路的时候，选择了将共现矩阵简化为**基于相邻转移共现计算相似度**，减少计算量，而后直接取 TopK 作为预测结果。

#### 2.1.2 数据预处理

将每一次 event 记为 `(aid, type)` 的元组，通过稀疏矩阵，以有向边的形式，统计相邻转移次数。具体地，稀疏矩阵使用 `sparse.coo_matrix()` 构建，而不通过高维度矩阵直接存储，减少内存。

#### 2.1.3 结果计算

遍历每一个 pair 的转移次数向量和其余 pair 对应向量进行点积刻画相似度，通过 `np.argpartition()` 对相似度进行部分排序，相比全排序减少计算量。同时每一个 pair 保留前 20 个最大的 pair。

最后遍历测试集 history 的 event 的 top20 pair，同时遍历 type，如果 type 对应，就对这个 aid 的分数加上相似度，最后结果为每一个 type 的对应分数前 20 的 aid。此时取前 20 使用的是 `sorted()`。

---

### 2.2 surprise.py 文件阅读

#### 2.2.1 思路概括

在从 [Datawhale 推荐系统教程](https://datawhalechina.github.io/fun-rec/) 学习基础推荐算法时，读到里面提到了 Swing 论文里的 Surprise 算法，重点加入时间顺序和方向性，通过聚类把稀疏的共现关系聚合到聚类级别，在类别层面进行相关性计算，我因此受到了启发。

因为之前的 itemCF 丢失了时间信息，但真实用户行为是有顺序的。因此，在 OTTO 项目提供了 `ts` 标签的前提下，我想看看加入时间方向后，模型能不能更准，因此就有了从 itemCF 修改出的 surprise 版本。

但是之前建模的时候，通过聚类把稀疏的共现关系聚合到聚类级别因为复杂度太高，内存爆炸，因此修改后**仅仅是每个 aid 作为一类**，把 pair 转移变成了 aid 转移，计算 Swing 分数，对类别层面和商品层面的 Surprise 都做了简化处理，聚类层面更是直接没有做，对模型大大进行了简化。

可以说，我只是基于这个启发，加上了时间和方向性，同时试图用每一个 aid 作为一类，替代里面的聚类，暂时对模型加上时间，进行一个尝试。

#### 2.2.2 数据预处理

在 itemCF 的数据处理基础上，增添了 aid 转移矩阵，同时对于每一个 pair 的时间进行了记录，用于刻画时间权重。

#### 2.2.3 数学模型对比

**我的 Surprise 实现公式**

**一、类别层面（Aid-Level Transition Probability）**

在我的简化实现中，每个 `aid` 自身作为一个"类别"。类别层面的相关性由**有向转移概率**定义：

$$
\theta_{a_i, a_j} = P(a_j \mid a_i) = \frac{N(a_i \rightarrow a_j)}{N(a_i)}
$$

其中：

- $a_i$：某个商品 ID（同时充当"类别"角色）
- $a_j$：另一个商品 ID（$a_i \neq a_j$）
- $N(a_i \rightarrow a_j)$：在全部训练 session 的相邻转移中，先出现 `(aid=a_i, type=*)` 随后出现 `(aid=a_j, type=*)` 的累计次数
- $N(a_i)$：`aid=a_i` 在所有 `(aid, type)` pair 中出现的总次数

该概率刻画了"给定用户刚刚交互过 $a_i$，下一个交互对象是 $a_j$"的经验概率。

**二、商品层面（Pair-Level Surprise Score）**

在类别层面过滤的基础上，对每个具体的 `(aid, type)` pair 计算互补分数。

**a. 时间权重**

对于 pair $p_i = (a_i, t_i)$ 与 $p_j = (a_j, t_j)$，首先计算两者的平均时间间隔：

$$
\Delta \bar{\tau} = \left| \bar{\tau}_{p_i} - \bar{\tau}_{p_j} \right|
$$

其中：

- $\bar{\tau}_{p_i}$：pair $p_i$ 在所有出现记录中时间戳的均值
- $\bar{\tau}_{p_j}$：pair $p_j$ 在所有出现记录中时间戳的均值

进而定义时间衰减权重：

$$
w_{\text{time}}(p_i, p_j) = \frac{1}{1 + \log\left(1 + \dfrac{\Delta \bar{\tau}}{1000}\right)}
$$

$\dfrac{\Delta \bar{\tau}}{1000}$ 将毫秒转换为秒；对数函数用于压缩长间隔的差异。

**b. 商品层面分数**

$$
s_1(p_i, p_j) =
\begin{cases}
\dfrac{w_{\text{time}}(p_i, p_j)}{N(p_i) \cdot N(p_j)}, & \text{if } a_j \in \text{Rel}(a_i) \\[10pt]
\dfrac{N(p_i \rightarrow p_j)}{N(p_i)}, & \text{otherwise}
\end{cases}
$$

其中：

- $N(p_i)$：pair $p_i$ 的总出现次数
- $N(p_j)$：pair $p_j$ 的总出现次数
- $N(p_i \rightarrow p_j)$：$p_i$ 之后相邻出现 $p_j$ 的次数
- $\text{Rel}(a_i)$：与 $a_i$ 相关的商品集合，定义为 $\{a_j \mid \theta_{a_i,a_j} > 0\}$

满足 $\theta_{a_i,a_j} > 0$ 即认为两个 aid 之间存在有向转移证据，从而进入带时间权重的商品层面计算；否则退化为简单的转移概率。

**三、最终相似度与预测**

所有 pair 的相似度计算完成后，按分数降序排序，每个 pair 仅保留 Top-20 邻居：

$$
\text{TopK}(p_i) = \arg\max_{p_j}^{(20)} s_1(p_i, p_j)
$$

预测阶段，对于一个测试 session 中已出现的 pair 集合 $S_{\text{seen}}$，目标类型 $t$ 的候选 aid 分数为：

$$
\text{score}(a) = \sum_{p_i \in S_{\text{seen}}} \sum_{p_j \in \text{TopK}(p_i)} s_1(p_i, p_j) \cdot \mathbb{1}[a_j = a \land t_j = t]
$$

即对每个 seen pair 的 Top-20 邻居按目标类型进行等权累加，最终取分数最高的 20 个 aid 作为预测结果。

---

**官方 Surprise 公式**

**一、类别层面**

$$
\theta_{c_i, c_j} = P(c_{i,j} \mid c_j) = \frac{N(c_{i,j})}{N(c_j)}
$$

其中：

- $c_i, c_j$：商品类别
- $N(c_{i,j})$：先购买类别 $c_i$ 后又购买类别 $c_j$ 的次数
- $N(c_j)$：购买类别 $c_j$ 的总次数

通过最大相对落点阈值筛选相关类别。

**二、商品层面**

$$
s_1(i, j) = \frac{\sum_{u \in U_i} \dfrac{1}{1 + |\tau_{ui} - \tau_{uj}|}}{\|U_i\| \times \|U_j\|}
$$

其中：

- $i, j$：商品
- $U_i, U_j$：购买过商品 $i$、$j$ 的用户集合
- $\tau_{ui}, \tau_{uj}$：用户 $u$ 购买商品 $i$、$j$ 的时间
- 求和仅在 $j$ 属于 $i$ 的相关类别且购买时间晚于 $i$ 时计入

**三、聚类层面**

$$
s_2(i, j) = s_1(L(i), L(j))
$$

其中 $L(i)$ 是商品 $i$ 的聚类标签，由 Swing 分数作为边权的标签传播算法得到。

**四、线性组合**

$$
s(i, j) = \omega \cdot s_1(i, j) + (1 - \omega) \cdot s_2(i, j)
$$

其中 $\omega$ 是权重超参数。

---

**对比总结**

我的实现与官方 Surprise 的核心差异集中在**类别定义**和**用户维度**上。

官方算法假设存在商品到类别的映射，并在同一用户的购买记录中计算时间差和方向性；而我的数据没有用户维度和类目标注，因此把每个 `aid` 当作独立类别，把 `(aid, type)` pair 作为计算单元，用相邻转移代替用户内购买序列。在时间建模上，官方对每对 `(user, item)` 逐一计算时间差求和，我则用 pair 的平均时间戳差异作为简化近似。聚类层面完全未实现。整体而言，我的版本保留了"方向性转移"和"时间衰减"两个核心思想，但在类别粒度和用户信息利用上做了大幅简化。




### 2.3 具体改动

因为之前学习了开源文件 `past_event.py` 中基于历史用户行为召回的思路，其中重点策略为**时间序列衰减模型**和**操作权重**的应用。因此，虽然共现矩阵有很大的进步空间，但目前主要是针对时间衰减系数的使用测试以及代码熟悉，因此共现转移矩阵暂时不变，只是在预测阶段尝试加入指数时间衰减系数，看看和原本的对比是否有提升。

我原本的做法：对 train 数据计算相似度，匹配 valid 数据集最后一个 event，直接提取相似度 top20。

指数时间衰减系数和 type 权重：与 `ts`、`type` 有关。

因此设计如下改进：

遍历历史 aid，记每一个历史 aid 得到指数时间衰减系数和 type 权重乘积为 `w`。对于历史 aid 所对应 pair 的 top20 里的 aid，如果有，所有 aid 添加 `sim / sum(sim) * w` 的分数，其中 `sum(sim)` 为 top20 的相似度之和；而对于历史 aid 本身，则添加 `w` 本身。

撰写代码如下：

```python
sequence_weight = defaultdict(float)
aid_scores = {t: defaultdict(float) for t in TARGET_TYPES}
for target_type in TARGET_TYPES:
    seq_weight = 0.1 if target_type == 'clicks' else 0.5
    sequence_weight[target_type] = np.power(2, np.linspace(seq_weight, 1, len(events))) - 1

    for idx, event in enumerate(events):
        pid = get_pair_id(event['aid'], event['type'])
        mix_weight = type_to_weight[event['type']] * sequence_weight[target_type][idx]
        neighbors = top20_similarity.get(pid) or []
        if neighbors:
            sum_sim = sum(sim_s for _, sim_s in neighbors)
            for id, sim_s in neighbors:
                aid = id_to_pair[id][0]
                aid_scores[target_type][aid] += mix_weight * sim_s / sum_sim
        aid_scores[target_type][event['aid']] += mix_weight
    scored = aid_scores[target_type]
    top20 = heapq.nlargest(20, scored.items(), key=lambda x: x[1])
    preds = " ".join(str(aid) for aid, _ in top20)
    submission_rows.append({"session_type": f"{session_id}_{target_type}", "labels": preds})
```

同样是使用堆让 `aid_scores` 保留 top20 个，不过此处调用 `heapq.nlargest()` 完成。也可以按照开源作者那样，自定义 heapq 相关函数达到相同目的。后续练习代码能力的时候，可以单独把这部分拿出来写，把 `heapq.nlargest()` 替换成自己写的代码。

---

### 2.4 测试结果

对我自己的 itemCF 和 surprise 都做了上述调整，修改后的代码为 v2，按照固定小样本验证集测试，得到结果如下：

| 方法 | Overall Recall | clicks | carts | orders |
|---|---:|---:|---:|---:|
| [itemCF](../restart/itemCF.py)（原版） | 0.057971 | 0.134873 | 0.071788 | 0.038246 |
| [itemCF_v2](../restart/itemCF_v2.py) | 0.391581 | 0.358434 | 0.198851 | 0.493470 |
| [surprise](../restart/surprise.py)（原版） | 0.059086 | 0.145582 | 0.077531 | 0.035448 |
| [surprise_v2](../restart/surprise_v2.py) | 0.383695 | 0.347390 | 0.194903 | 0.484142 |

作为对比，开源 `past_event` 在小样本上的分数为：

| 召回方法 | Overall | clicks | carts | orders |
|---|---:|---:|---:|---:|
| past_event | 0.394910 | 0.340696 | 0.202800 | 0.500000 |

**简要评价**：加入时间衰减和行为权重后，两个算法的整体 Recall 均提升约 6 倍，改进后的分数已接近开源 `past_event` 的 0.394910，但略弱于开源方法。这说明我原先基于相邻转移的共现矩阵构建方式在计算过程中丢失了大量有效信息，且数据组织方式存在冗余。后续优化时，可优先考虑去除转移矩阵的构建思路，转向在保留有效降维手段的前提下，尝试其他共现矩阵构建方式。


## 3. 开源代码阅读：`itemcf_data_preprare.py`

### 3.1 数据预处理策略

作者只保留最近 7 天的数据，避免 `user × item` 矩阵过于稀疏。对于粗排召回阶段，保留全部历史的性价比确实不高，这是一个合理的取舍。

### 3.2 并行加速

作者使用 `pandarallel` 进行多核并行处理：

```python
pandarallel.initialize(progress_bar=True, nb_workers=15)
```

与 Numba JIT 编译不同，`pandarallel` 适合处理大量重复且互相独立的子任务（如对每个 session 独立处理），让多个 CPU 核心同时工作。

### 3.3 最终数据格式

预处理后生成稀疏矩阵，表示每个 session 中各个 `aid` 出现的次数，供后续 `item_cf` 调用。

---

## 4. 开源代码阅读：`item_cf.py`

### 4.1 模型调用

作者使用 `implicit` 库中的 `BM25Recommender` 进行基于物品的协同过滤：

```python
bm25_model = implicit.nearest_neighbours.BM25Recommender(K=50, num_threads=15)
bm25_model.fit(user_items_train)
```

`K=50` 表示对每个物品保留最相似的 50 个邻居。此处`num_threads=15`是并行运算参数，代表使用15个CPU同时进行。

### 4.2 预测与去重

```python
test['labels'] = test['aid'].parallel_apply(lambda x: bm25_model.similar_items(x, N=51)[0][1:])
```

`similar_items(x, N=51)` 返回 51 个相似物品，其中第一个是自己，因此需要注意细节，从索引 1 开始取。

得到召回结果后，与其他召回来源合并去重，使用 `np.setdiff1d()` 取差集。

### 4.3 与我之前的实现对比

| 对比维度 | 我的实现 | 开源实现 |
|---|---|---|
| 相似度计算 | 自己构建的转移共现矩阵 | 调用 `implicit` 库的 BM25 |
| 降维方式 | 简化相似矩阵的计算方法 | 限制历史数据范围（7天）+ 缩小候选集 |
| 预测依据 | 基于 session 所有历史行为的加权 | 基于最后一个 `aid` 的相似物品 |

核心差异在于：开源作者没有简化算法逻辑，而是通过**工程手段控制数据规模**，保留了完整的相似度计算流程。

### 4.4 小样本测试结果

在我固定的小样本集合上运行开源代码，与 `past_event` 对比：

| 召回方法 | Overall | clicks | carts | orders |
|---|---:|---:|---:|---:|
| past_event | 0.394910 | 0.340696 | 0.202800 | 0.500000 |
| item_cf | 0.188468 | 0.201138 | 0.104092 | 0.228545 |

**简要评价**：在小样本上，`item_cf` 的召回效果明显弱于 `past_event`，三个维度的 Recall 均偏低。这可能是因为协同过滤在小规模数据上难以构建足够的共现关系，而历史序列召回对 session 内行为利用更直接有效。

---

## 5. 今天的核心认识

1. **算法效果高度依赖数据组织方式**：我的原版 itemCF 和 surprise 与开源 past_event 在核心思想上并不差多少（都是基于历史行为的加权聚合），但分数从 0.05 到 0.38 的差距说明：共现矩阵的计算方式决定了信息保留的质量。相邻转移虽然简单，但丢弃了太多信息。

2. **开源工程代码的每一层都在做时间/空间上的优化**：`past_event` 中看到的分批推理、JIT 加速、Heap TopK，以及 `item_cf` 中的 BM25、并行 apply、7天数据截断——这些不是在改变算法，而是让算法能够在大规模数据上运行。算法公式只占 20%，剩下 80% 是让它跑起来。

3. **不同召回策略在不同数据规模下表现不同**：在小样本上 past_event 优于 item_cf，但并不意味着 item_cf 本身更差——它只是需要更多数据来构建稳定的共现关系。后续随着数据量扩大，这个结论可能会变化。

下一步继续阅读开源项目的其他召回方式（如 covisit），重点关注它与 past_event、item_cf 在工程实现上的差异，以及在相同小样本上的表现对比。
