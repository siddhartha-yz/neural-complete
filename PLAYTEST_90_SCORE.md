# Neural Complete：自主试玩评分循环结果

本文件记录按 `EVALUATION.md` 执行的“制作 → 真实浏览器试玩 → 评分 → 反馈 → 修改 → 再试玩”循环。

## 终止条件

只有同时满足以下条件才允许停止本轮 Goal：

- 自动硬门槛测试通过；
- 至少一个 Demo 达到 **90/100**；
- 该 Demo 的“构造自由度 + 解法空间 + 因果可理解性”达到 **44/50**；
- 不是通过预制答案或只调超参数获得高分。

## Iteration 1

### 实际试玩方式

在 1440×900 Chromium 中从清空 localStorage 的状态逐个进入 5 个 Demo，使用可见 UI 完成主要操作：

- XOR：双击元件、端口接线、训练、hidden evaluation；
- Feature Foundry：把 raw feature 拖进加工槽、运行机器、把派生 feature 拖进 classifier dock；
- Vision Forge：把 trainable filter 拖入 optical bench，训练后检查 kernel / feature map；
- Latent Cartographer：添加 latent channel、用 connectivity painter 画 mask、训练、hidden reconstruction；
- Policy Garden：把 sensor chip 拖进 brain slot，运行 Q-learning，再执行全图 harvest。

### 试玩中发现并反馈

1. **Latent Cartographer 存在真实 UI 阻塞**：1440×900 下 atlas 被 canvas 的 min-content 撑高，footer 覆盖训练按钮，导致鼠标无法点击。
2. 反馈回实现后修改 CSS grid 的 min-height / overflow 约束。
3. 再次用鼠标点击完整试玩，训练按钮恢复可操作，2D dense-mask autoencoder hidden MSE 达到约 **0.00138**。
4. 其余四个 Demo 的可见 UI 主流程也完成，无 browser console/page error。

这构成了本轮至少一次真实“试玩 → 发现问题 → 修改 → 再试玩”闭环。

## 评分

| Demo | 构造自由度 /20 | 解法空间 /15 | 因果可理解 /15 | ML 真实性 /15 | 调试 /10 | 失败质量 /10 | 教学迁移 /10 | UI /5 | 总分 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **XOR Construction Lab** | 19 | 14 | 14 | 15 | 10 | 9 | 9 | 4 | **94** |
| **Feature Foundry** | 19 | 14 | 14 | 15 | 8 | 9 | 9 | 4 | **92** |
| **Policy Garden** | 18 | 14 | 13 | 15 | 9 | 9 | 9 | 5 | **92** |
| **Latent Cartographer** | 18 | 13 | 14 | 15 | 9 | 8 | 9 | 5 | **91** |
| Vision Forge | 16 | 12 | 13 | 15 | 9 | 8 | 9 | 5 | **87** |

### 前三项合计

- XOR Construction Lab：**47 / 50**
- Feature Foundry：**47 / 50**
- Policy Garden：**45 / 50**
- Latent Cartographer：**45 / 50**
- Vision Forge：**41 / 50**

因此至少四个 Demo 达到当前 90+ 门槛，最强的是 **XOR Construction Lab：94/100**。

## 为什么 XOR 现在可以判 90+

### 构造不是参数选择

玩家面对的是空白 DAG，而不是预制的“hidden units / activation”控制面板。可任意：

- 放置 neuron；
- 放置 Multiply / Add / Square / Abs；
- 任意接线；
- 删除节点与边；
- 改 neuron activation；
- freeze / mute；
- 在当前结构上真实训练。

forward / backward graph 由玩家当前拓扑自动决定。

### 已证明 3 类结构解

固定评测 seed 下已经实际训练并通过 noisy hidden XOR：

1. `x₁ × x₂ → logistic output`：**100% hidden**
2. `2 → 2 tanh neurons → output`：**100% hidden**
3. `2 → 3 ReLU neurons → output`：**100% hidden**

这三类不是 learning rate 的不同，而是表示方式 / 网络拓扑真正不同。

### 因果链可追踪

玩家可以直接看到：

```text
结构变化
→ 节点 activation / edge weight / gradient 变化
→ decision field 变化
→ train loss / accuracy 变化
→ hidden noisy XOR 泛化变化
```

并可用 freeze / mute 做局部消融。

### 失败本身可调查

已实现并实际触发的失败证据包括：

- cycle / disconnected graph；
- 缺输入；
- linear representation 无法解决 XOR；
- Dead ReLU；
- gradient 接近 0；
- 节点 mute 后的结构消融；
- train 与 hidden 结果分离。

## 其余 Demo 的具体反馈

### Feature Foundry — 92

优点：真正允许递归构造任意 feature expression，并验证了至少 3 类通关结构：

- `x₁² + x₂²`
- `x₁²` 与 `x₂²` 分别进入 classifier
- `|x₁| + |x₂|`

短板：feature expression 目前以“卡片 + 工厂槽位”呈现，缺少像 XOR 那样完整的历史数据流图；调试维度略弱。

### Policy Garden — 92

验证过的不同 state representation：

- `ROW + COLUMN`
- `GOAL ΔX + GOAL ΔY`
- `GOAL DIR + WALL RADAR + LANDMARK REGION`

blind / 过度 alias 的表示会真实失败。短板是 reward beacon 的结构空间还不如 sensor representation 丰富。

### Latent Cartographer — 91

玩家直接决定 encoder / decoder 的 connectivity mask，而不是只调 latent dim。已验证 dense、checkerboard complement、稀疏 row/column 等不同连接结构。

短板：逐像素 painting 的操作成本仍偏高；后续可以加入可组合的 region brush，而不能退回预制结构选项。

### Vision Forge — 87

真实 ML 与调试能力足够，但结构自由度仍低于其他四个：

- 可以选择多个 3×3 / 5×5 trainable filter；
- 可以改变 reducer；
- 可以 freeze / mute；
- 可以检查 kernel / feature map。

但目前没有真正的多阶段 Conv → Activation → Pool → Merge → Classifier 自由 pipeline，因此前三项只有 41/50。它仍需要继续结构化重做，不能因为算法真实而判 90。

## 本轮结论

目标条件已经第一次达到：

```text
XOR Construction Lab
94 / 100
前三项 47 / 50
自动硬门槛：已由 e2e_check.py + solution_diversity_check.py 全部确认
```

最终自动硬门槛已全部通过，因此本轮“制作 → 试玩评分 → 反馈 → 修改 → 再试玩”循环满足终止条件。
