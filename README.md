# Neural Complete

Neural Complete 是一个以“亲手构造机器学习结构”为核心的教学游戏原型。

当前版本已经放弃“预制模型 + slider + Train”的 Playground 方向，改成 5 个结构优先的实验。共同规则只有一个：

> 开发者提供基础元件，玩家自己决定结构；训练、梯度、表示与策略都由当前结构真实产生。

## 当前五个 Demo

### 01 · XOR Construction Lab

自由神经图编辑器。

玩家可以：

- 放置 Neuron / Multiply / Add / Square / Abs；
- 任意接线与删线；
- 改 activation；
- freeze / mute 节点；
- 查看节点 activation、gradient、weight；
- 单步或批量训练；
- 在 decision field 上移动 probe；
- 用新的 noisy XOR hidden set 做最终评测。

forward / backward graph 根据玩家当前 DAG 自动生成。

已验证的结构性通关方案包括：

- `x₁ × x₂ → logistic output`
- `2 → 2 tanh neurons → output`
- `2 → 3 ReLU neurons → output`

### 02 · Feature Foundry

特征工程工厂。

`x₁ / x₂` 是原料，Square / Abs / Add / Multiply / Subtract 是加工机器。玩家可以递归生产任意派生 feature，再把自己生产的 feature 拖进 Logistic classifier dock。

已验证的不同方案：

- `x₁² + x₂²`
- `x₁²` 与 `x₂²` 分别进入 classifier
- `|x₁| + |x₂|`

系统显示 feature class separation、redundancy、训练边界与 hidden generalization。

### 03 · Vision Forge

CNN 光学工作台。

玩家把真正可训练的 3×3 / 5×5 filter 放入 filter bank，并决定每个 feature map 如何聚合。

可观察：

- learned kernel；
- feature map；
- pooled response；
- output weight；
- kernel gradient；
- freeze / mute 消融结果。

卷积核从随机参数开始，由最终 BCE 分类误差真实反向传播更新。

### 04 · Latent Cartographer

Masked Autoencoder 潜空间制图室。

玩家不是选择 latent dim 后点 Train，而是直接“画”信息连接：

- 每个 latent channel 能听哪些输入像素；
- 每个 latent channel 能向哪些重建像素写回。

支持：

- 多 latent channel；
- connectivity painter；
- freeze；
- reconstruction；
- latent scatter；
- interpolation；
- variance / correlation / collapse diagnostics；
- hidden reconstruction evaluation。

### 05 · Policy Garden

强化学习策略花园。

玩家不能直接移动 agent。玩家构造的是 agent 的“感知世界”：

- ROW
- COLUMN
- GOAL ΔX / ΔY
- GOAL DIR
- WALL RADAR
- DANGER
- LANDMARK REGION

最多把 3 个 sensor chip 装进 brain slots，还可以在环境中放有限的 reward beacon。

Q-table 从 0 开始，真实执行 TD update。改变 sensor 组合会直接改变 state representation 和 Q-table 的地址空间。

## 90+ 评估

正式评分标准：

- [EVALUATION.md](./EVALUATION.md)

上一版试玩反馈与 90+ 设计合同：

- [DEMO_90_PLUS_REPORT.md](./DEMO_90_PLUS_REPORT.md)

当前自主“制作 → 试玩 → 评分 → 反馈 → 修改 → 再试玩”结果：

- [PLAYTEST_90_SCORE.md](./PLAYTEST_90_SCORE.md)

当前最高分：

~~~text
XOR Construction Lab
94 / 100
前三项：47 / 50
~~~

## 运行

~~~bash
cd /home/ubuntu/workspace/neural-complete
python3 -m http.server 4173 --bind 127.0.0.1
~~~

打开：

~~~text
http://127.0.0.1:4173
~~~

项目为纯前端 ES Modules，不依赖外部 ML library。学习算法直接在浏览器内执行。

## 自动验收

真实 UI 端到端试玩：

~~~bash
LD_LIBRARY_PATH=/home/ubuntu/workspace/neural-complete/.browser-libs/root/usr/lib/x86_64-linux-gnu \
/home/ubuntu/local-shell-mcp/.venv/bin/python e2e_check.py
~~~

结构解法多样性：

~~~bash
LD_LIBRARY_PATH=/home/ubuntu/workspace/neural-complete/.browser-libs/root/usr/lib/x86_64-linux-gnu \
/home/ubuntu/local-shell-mcp/.venv/bin/python solution_diversity_check.py
~~~

验收覆盖：

- 可见 UI 构造，而不是只调用内部接口；
- 真实训练；
- hidden/test 分离；
- 五个 Demo 的成功流程；
- 至少 3 类结构性不同的有效解；
- 已知弱结构真实失败；
- persistence / reload；
- console / page error；
- NaN / Inf 防护；
- 固定 seed 下的可复现实验。

## 设计原则

1. 玩家主要操作对象必须是结构，不是超参数。
2. 参数必须由 data / gradient / reward 真实更新。
3. 失败必须留下可调查的内部证据。
4. train 与 hidden/test 必须分开。
5. 同一任务至少允许 3 类结构性不同的有效解。
6. 不同任务应拥有不同的交互空间，不复用统一 Dashboard。
7. 自动测试只证明“没有作弊”；是否达到 90+ 仍由 `EVALUATION.md` 的人工试玩评分决定。
