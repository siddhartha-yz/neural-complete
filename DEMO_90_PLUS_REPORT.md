# Neural Complete：90+ Demo 迭代评估报告

目标：不是做“更精致的 ML 可视化实验”，而是做出一个按当前 `EVALUATION.md` 评分能稳定达到 **90/100 以上** 的 Demo。

## 当前试玩结果

已实际通过浏览器试玩 5 个 Demo，并测试错误配置、替代解法与隐藏评测：

| Demo | 总分 |
|---|---:|
| Q-Lab | 75 |
| XOR Workshop | 73 |
| Conv Forge | 71 |
| Boundary Foundry | 63 |
| Latent Vault | 62 |

当前版本已经解决“没有真实机器学习”的问题。所有 Demo 都存在真实训练、真实失败、hidden/test evaluation、模型内部状态可视化。

但它们仍然没有达到《图灵完备》式游戏体验。

核心结构性问题：

> **目前是“模型结构由开发者写死，玩家负责调参数”；目标应该变成“基础元件由开发者提供，模型结构由玩家亲手构造”。**

不能再主要依赖 slider / select / Train 按钮形成玩法。

---

## 90+ 目标评分

必须按 `EVALUATION.md` 的 100 分制设计：

| 维度 | 权重 | 90+ Demo 的要求 |
|---|---:|---:|
| 构造自由度 | 20 | ≥18 |
| 解法空间 | 15 | ≥13 |
| 因果可理解性 | 15 | ≥13 |
| ML 真实性 | 15 | ≥14 |
| 调试能力 | 10 | ≥9 |
| 失败质量 | 10 | ≥9 |
| 教学迁移 | 10 | ≥9 |
| 操作 / UI | 5 | ≥4 |

目标不是总分勉强达到 90，而是首先保证：

```text
构造自由度 + 解法空间 + 因果可理解性 ≥ 44 / 50
```

这是当前所有 Demo 最大的短板。

---

## 必须保留的优点

现有版本已经做对的部分不要丢：

1. 参数必须真的通过 gradient / reward / data 更新。
2. 错误设计必须真实失败。
3. 必须有独立 hidden/test evaluation。
4. 玩家必须能看到模型内部状态。
5. 训练不能是预制动画。
6. 固定 seed 可复现。
7. 失败不能通过“多训练几个 epoch”万能解决。
8. 最终评测不能只测训练集。
9. 浏览器 console / numerical state 必须稳定。
10. 存档、重置、重试必须可靠。

这些属于硬门槛，不是加分项。

---

## 必须彻底改变的玩法结构

禁止继续以这种循环作为核心玩法：

```text
选择 activation
→ 调 hidden units
→ 调 learning rate
→ 点击 Train
→ 看 accuracy
```

这仍然只是 ML Playground。

目标主循环应当是：

```text
理解任务
→ 从元件库选组件
→ 搭建数据流 / 模型结构
→ 运行训练
→ 观察内部状态
→ 定位失败原因
→ 修改结构
→ 再训练
→ 通过隐藏评测
```

玩家主要操作对象必须是**结构**，而不是超参数。

---

## 推荐的元件体系

应提供一组真正可组合的 ML primitive，例如：

```text
Input
Feature
Linear
Neuron
Add
Multiply
Square
ReLU
Tanh
Sigmoid
Conv
Pool
Normalize
Loss
MSE
BCE
Optimizer
Memory
Reward
State
Action
```

不要求首个 Demo 一次实现全部元件，但至少需要一组能产生明显组合空间的 primitive。

关键要求：

> **开发者提供积木，但不能提前提供完整正确模型。**

例如 Boundary 不能再直接提供：

```text
Raw
Cross
Radial
```

而应提供：

```text
x₁
x₂
Multiply
Square
Add
Linear
Sigmoid
```

让玩家自己发现：

```text
x₁ → Square ┐
            Add → Linear → Sigmoid
x₂ → Square ┘
```

这才是目标体验。

---

## 解法空间要求

每个关卡必须提前证明存在至少 **3 类结构性不同** 的通关方案。

注意，是“结构不同”，不是：

```text
lr=.1
lr=.2
lr=.3
```

这种参数变化。

例如 XOR 可以允许：

方案 A：

```text
2 → 2 tanh → 1
```

方案 B：

```text
2 → 3 ReLU → 1
```

方案 C：

```text
手工组合 nonlinear feature → linear classifier
```

不同方案应在以下方面产生真实 trade-off：

- 参数量；
- 收敛速度；
- 稳定性；
- 泛化能力；
- 可解释性。

最好允许“能通关，但不是最优”，而不是只有唯一正确结构。

---

## 失败必须成为游戏内容

高分 Demo 不能只告诉玩家：

```text
FAILED
hidden accuracy 72%
```

失败本身必须具有可调查性，例如：

```text
梯度接近 0
某个 neuron 永远不激活
两个 feature 高度冗余
train acc 100% / hidden acc 61%
kernel 只响应固定位置
latent 两个维度塌缩
Q-value 在局部环路中不断增大
```

玩家应当通过这些迹象自己推断下一步怎么改。

不要直接提示：

> “请增加 ReLU。”

而应该提供足够证据，让玩家自己得出：

> “当前结构是线性的，无法把这些区域分开。”

---

## 调试系统必须成为核心玩法

至少应支持多种内部探针：

```text
查看某节点输出
查看某条边的数值
查看 neuron activation
查看 gradient
查看 parameter
查看 loss contribution
查看 feature map
查看错误样本
查看 hidden/test 分布差异
冻结节点
禁用连接
单步训练
```

最好支持类似《图灵完备》的：

> 单步运行 / 暂停 / probe / 局部观察。

调试不是附属 UI，而应直接参与解谜。

---

## 不要再做“五个 Demo 一个 UI 模板”

不同任务应拥有不同的交互空间。

例如：

- 神经网络：节点图编辑器；
- CNN：空间 feature-map pipeline；
- RL：环境 + policy/reward editor；
- Autoencoder：latent workspace；
- feature engineering：数据流图。

不能只是相同三栏布局更换标题和颜色。

---

## 推荐的第一个 90+ 原型

### XOR Construction Lab

这是最适合作为首个 90+ Demo 的方向。

任务：

> 不直接选择预制网络，而是从基础组件构建一个能解决 noisy XOR 的模型。

元件：

```text
Input x₁
Input x₂
Linear neuron
ReLU
Tanh
Sigmoid
Add
Multiply
Output
BCE Loss
```

玩家能够：

- 拖入 neuron；
- 创建连接；
- 删除连接；
- 改 activation；
- 调整层和拓扑；
- 查看每个 neuron activation；
- 查看 gradient；
- freeze / mute 节点；
- 单步训练；
- 在 decision field 上 probe。

系统根据玩家当前图结构自动生成 forward / backward computation。

隐藏评测：

- 使用新的 noisy XOR 点；
- 不允许读取 hidden labels；
- 至少要求 hidden accuracy ≥ 94%。

额外评价：

```text
通关状态
参数数量
训练步数
hidden accuracy
结构复杂度
```

这样自然会产生多种有效解。

核心要求：

> 玩家必须真的能搭出一个开发者没有手写成选项的网络。

---

## 90+ Demo 应产生的体验

第一次进入：

> 我不知道正确结构是什么。

尝试错误结构：

> 我能看到为什么失败。

观察内部状态：

> 我产生了一个自己的假设。

修改结构：

> 结果按我的预期发生变化。

最后通关：

> 这是我搭出来的模型，不是我猜中了正确选项。

如果没有这种体验，即使算法再真实、UI 再漂亮，也不应判定为 90+。

---

## 最终验收

完成 Demo 后必须同时进行两类测试。

### 自动测试

- forward / backward 数值正确；
- 真实训练；
- hidden evaluation；
- 多种错误结构真实失败；
- 至少 3 类不同结构可成功；
- persistence 正常；
- console error = 0；
- NaN / Inf = 0。

### 人工试玩

- 不看源码，从零开始能否理解规则；
- 是否真的需要搭结构；
- 是否存在探索过程；
- 是否有“发现机制”的瞬间；
- 是否可以用不同思路通关；
- 失败是否能指导下一步实验；
- 是否有明显复玩价值。

只有同时满足：

```text
自动测试通过
+
EVALUATION.md ≥ 90 / 100
+
前三项 ≥ 44 / 50
```

才算完成。

---

## 北极星问题

评估任何方案时都必须先回答：

> **玩家是在寻找一种自己构造出来的模型，还是在寻找开发者预先藏好的正确参数组合？**

如果仍然是后者，就继续重做，不要因为算法真实或 UI 完整而停止迭代。
