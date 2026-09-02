# Neural Complete

一个以“真的让模型学习”为核心的机器学习交互实验项目。

当前版本已经完全放弃之前的符号电路 / 手工公式拼装方向，改成 5 个互不相同的成品级 Demo。每个 Demo 都包含真实数值学习过程，参数会由数据或奖励更新，不是预制动画。

## 五个 Demo

### 01 · Boundary Foundry
**主题：监督学习 / Logistic Regression**

玩家决定模型能看到什么特征、使用多少训练样本和什么学习率，然后用真实 BCE 梯度下降训练分类器。

关键现象：
- 特征表示决定可学习边界的形状；
- raw features 无法表达环形分类；
- radial feature 可以；
- 训练集准确率和隐藏集泛化不是一回事。

### 02 · XOR Workshop
**主题：MLP / Backpropagation / Nonlinearity**

玩家配置隐藏单元数量、激活函数和学习率，让一个小型 MLP 通过真实反向传播学习带抖动的 XOR。

关键现象：
- 多层线性网络仍然等价于线性变换；
- 非线性隐藏表示是解决 XOR 的核心；
- 隐藏单元激活可以直接探测；
- 决策场会随着 backprop 实时变化。

### 03 · Conv Forge
**主题：CNN / Learned Kernels / Feature Maps**

网络从随机 3×3 卷积核开始，用分类误差反向训练 kernel、bias 和输出分类器。

关键现象：
- 卷积核不是人工写好的 Sobel 模板；
- kernel 会从数据中形成方向偏好；
- feature map 可以直接观察；
- pooling 改变位置不变性的方式；
- 隐藏集使用新的噪声和线条位置。

### 04 · Latent Vault
**主题：Autoencoder / Unsupervised Learning**

不给类别标签，只要求网络把 6×6 图像压入低维 bottleneck 后重建。

关键现象：
- target 就是 input 本身；
- 1D bottleneck 无法稳定保存两个独立生成因素；
- 2D latent 可以显著降低隐藏重构误差；
- 3D/4D 虽更容易，但不符合压缩目标；
- 支持 latent interpolation 和潜空间可视化。

### 05 · Q-Lab
**主题：Reinforcement Learning / Q-Learning**

玩家不能直接移动 agent，只能控制 reward、epsilon、alpha、gamma，让策略从经验中学习。

关键现象：
- Q-table 从全 0 开始；
- 每次 transition 都执行 TD update；
- epsilon 控制 exploration / exploitation；
- step penalty 会影响路径长度；
- 最终评测遍历全图安全起点，而不是只检查一条固定路线。

## 运行

~~~bash
cd /home/ubuntu/workspace/neural-complete
python3 -m http.server 4173 --bind 127.0.0.1
~~~

打开：

~~~text
http://127.0.0.1:4173
~~~

项目为纯前端 ES Modules，不依赖 TensorFlow.js、PyTorch.js 或其他外部 ML 库。所有学习算法都直接在浏览器中实现。

## 自动验收

~~~bash
LD_LIBRARY_PATH=/home/ubuntu/workspace/neural-complete/.browser-libs/root/usr/lib/x86_64-linux-gnu \
/home/ubuntu/local-shell-mcp/.venv/bin/python e2e_check.py
~~~

该测试会在真实 Chromium 中验证：

- 主页存在 5 个独立 Demo；
- Boundary Foundry：
  - 弱特征表示失败；
  - radial feature + 足够样本通过隐藏评测；
- XOR Workshop：
  - linear hidden stack 失败；
  - nonlinear MLP 通过；
- Conv Forge：
  - 随机 kernel 失败；
  - 训练后的 kernel 通过新噪声 / 新位置隐藏集；
- Latent Vault：
  - 1D bottleneck 失败；
  - 2D bottleneck 通过；
- Q-Lab：
  - 初始 Q-table 失败；
  - 训练后全图策略通过；
- 五个 Demo 的完成状态可持久化；
- 页面刷新后完成状态仍然存在；
- 多种桌面 / 窄屏尺寸下无基础布局崩坏；
- 浏览器 console / page error 为 0。

## 设计原则

1. **必须发生学习**
   参数、表示或策略必须由训练数据 / reward 更新。

2. **错误配置必须可能失败**
   不是按一次按钮就永远 PASS。

3. **训练集不是最终答案**
   每个实验都有独立隐藏评测或全局策略评测。

4. **不同 Demo 不允许只是换皮**
   五个实验分别覆盖：
   - 线性监督学习；
   - 非线性表示学习；
   - 卷积特征学习；
   - 无监督表示学习；
   - 强化学习。

5. **模型内部状态可观察**
   包括 learned weights、hidden activations、feature maps、latent coordinates、Q-values / policy。
