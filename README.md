# Neural Complete — 第一关 Demo

静态网页成品，无外部依赖。

## 运行

```bash
cd /home/ubuntu/workspace/neural-complete
python3 -m http.server 4173 --bind 127.0.0.1
```

打开 `http://127.0.0.1:4173`。

## 第一关：边界初生

完整流程：
1. 连接 Linear → Sigmoid → BCE Loss → SGD。
2. 手动执行 Forward → Loss → Backward → Update，至少 3 轮。
3. 解锁连续训练，并使 Accuracy ≥ 94%、Loss ≤ 0.25。
4. 通关后进入自由实验，可调整学习率、切换观测样本。

所有训练计算都在浏览器本地真实执行，不是预制动画。
