import sys
sys.exit("DEPRECATED (2026-07-09): build_all.py 内容已与单课 build_NN.py 分叉,单课脚本才是源头。请运行 build_NN.py。")
"""Build notebooks 04-10 with understanding-first priority.

Strategy:
- 04 CFG: interview must-know → keep guidance scale code demo
- 05 Score Matching: pure understanding (bridge concept), no new code
- 06 Flow Matching: 🔥 MATURE — heavy explanation + minimal CFM loss code
- 07 Rectified Flow: 🔥 MATURE — explanation + reflow concept, minimal code
- 08 Consistency/LCM: explanation-heavy + consistency constraint demo
- 09 DiT: 🔥 CRITICAL — explanation + adaLN-Zero key code
- 10 Mini Diffusion 总装: integration demo DDPM vs Flow Matching
"""
import json
import os

os.chdir("/path/to/code/learning-diffusion")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.split("\n"),
    }


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(fn, cells):
    with open(fn, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"wrote {fn} ({len(cells)} cells)")


# =====================================================================
# 04 CFG — interview must-know, keep code demo
# =====================================================================
cells_04 = [
    md("""# 04 · Classifier-Free Guidance (CFG)

## 一句话直觉

**训练时偷偷教模型两种本事**（看条件生成 / 不看条件生成），**推理时把两者相减放大**，就能让生成结果"更像条件描述的那样"。

## 类比

想象你让画家画"戴帽子的猫"：
- **无条件画家**：随便画只猫
- **有条件画家**：画戴帽子的猫

CFG 的操作：先请**有条件画家**画一张，再请**无条件画家**画一张，然后沿着**"有条件 - 无条件"**这个方向**放大 3-7 倍**，得到"帽子特征极度强化的猫"。

这就是为什么 CFG scale=7 比 scale=1 更"听话"——你把条件信号放大了 7 倍。"""),

    md("""## Motivation：为什么需要 CFG

### 朴素条件生成的痛

把条件 c（文本/类别）直接喂给网络：`ε_θ(x_t, t, c)`。
**问题**：训练数据里条件 c 和图像 x 的对应关系噪声大，网络学出来的条件响应很弱，生成结果"不太像"描述。

### 早期方案：Classifier Guidance（2021）

额外训练一个分类器 `p(c|x)`，用它的梯度把生成"推"向条件：
```
ε_guided = ε_θ(x_t, t) - s·√(1-ᾱ)·∇_x log p(c|x)
```

**痛点**：要额外训一个分类器，而且分类器必须能处理**带噪图像**，工程麻烦。

### CFG（2022，Ho & Salimans）

**一个模型搞定**：训练时以 10-20% 概率**丢掉条件**（用 null token 替换），让模型**同时学会** conditional 和 unconditional 生成。

推理时：
```
ε̂ = ε_θ(x_t, t, ∅) + s · [ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅)]
```

- s=1：纯 conditional
- s=0：纯 unconditional
- **s=7**：条件方向放大 7 倍（SD 默认）

**为什么能成？** 数学上等价于从 `p(x|c)^s · p(x)^(1-s)` 采样——把条件分布"锐化"了 s 倍。"""),

    md("""## 核心算法一页纸

### 训练（改动极小）
```python
for x, c in loader:
    t = sample_t()
    noise = torch.randn_like(x)
    x_t = add_noise(x, t, noise)

    # 关键：10% 概率丢条件
    mask = torch.rand(B) < 0.1
    c_input = torch.where(mask, null_token, c)

    pred = model(x_t, t, c_input)
    loss = MSE(pred, noise)
```

### 推理
```python
# 一次 forward 同时算两个分支（batch 拼起来更快）
pred_cond = model(x_t, t, c)
pred_uncond = model(x_t, t, null)
pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
```

**工程 trick**：把 `[x_t; x_t]` 和 `[c; null]` 拼成 batch 一次前向，省一半时间。"""),

    code("""# CFG 在简单 toy 数据上的效果演示
# 用 2D 高斯混合做"条件生成"——两个类别分别是上下两团
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)

device = 'cpu'

# 构造数据：class 0 = (0, 2) 附近；class 1 = (0, -2) 附近
def make_data(n=2000):
    c = torch.randint(0, 2, (n,))
    mean = torch.stack([torch.zeros(n), (1 - 2*c).float() * 2.0], dim=1)
    return mean + 0.3*torch.randn(n, 2), c

x_data, c_data = make_data()

# schedule
T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, 0)

# 小 MLP，同时支持 class 条件 + null token (idx=2)
class CondMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_c = nn.Embedding(3, 32)  # 0,1,null
        self.emb_t = nn.Embedding(T, 32)
        self.net = nn.Sequential(
            nn.Linear(2+32+32, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2))
    def forward(self, x, t, c):
        h = torch.cat([x, self.emb_t(t), self.emb_c(c)], dim=-1)
        return self.net(h)

model = CondMLP()
opt = torch.optim.Adam(model.parameters(), 3e-3)
NULL = 2

# 训练：10% drop condition
for step in range(3000):
    idx = torch.randint(0, len(x_data), (256,))
    x0, c = x_data[idx], c_data[idx]
    t = torch.randint(0, T, (256,))
    noise = torch.randn_like(x0)
    ab = alpha_bars[t].unsqueeze(-1)
    xt = ab.sqrt()*x0 + (1-ab).sqrt()*noise
    drop = torch.rand(256) < 0.1
    c_in = torch.where(drop, torch.full_like(c, NULL), c)
    pred = model(xt, t, c_in)
    loss = F.mse_loss(pred, noise)
    opt.zero_grad(); loss.backward(); opt.step()
print(f'final loss: {loss.item():.4f}')"""),

    code("""# DDIM 采样 + CFG
@torch.no_grad()
def sample(model, n, cls, cfg_scale, steps=50):
    ts = torch.linspace(T-1, 0, steps+1).long()
    x = torch.randn(n, 2)
    c_cond = torch.full((n,), cls)
    c_null = torch.full((n,), NULL)
    for i in range(steps):
        t_cur, t_next = ts[i], ts[i+1]
        t_vec = torch.full((n,), t_cur.item(), dtype=torch.long)
        # CFG 核心：两个 forward 加权
        eps_c = model(x, t_vec, c_cond)
        eps_u = model(x, t_vec, c_null)
        eps = eps_u + cfg_scale * (eps_c - eps_u)

        ab_cur = alpha_bars[t_cur]; ab_next = alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0)
        x0_pred = (x - (1-ab_cur).sqrt()*eps) / ab_cur.sqrt()
        x = ab_next.sqrt()*x0_pred + (1-ab_next).sqrt()*eps
    return x

# 扫描 CFG scale，看生成"类别 0"（上团）的效果
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, s in zip(axes, [0.0, 1.0, 3.0, 7.0]):
    samples = sample(model, 500, cls=0, cfg_scale=s)
    ax.scatter(x_data[:,0], x_data[:,1], s=2, alpha=0.2, label='data')
    ax.scatter(samples[:,0], samples[:,1], s=4, c='red', label=f'CFG={s}')
    ax.axhline(0, color='gray', lw=0.5); ax.set_xlim(-4,4); ax.set_ylim(-4,4)
    ax.set_title(f'CFG scale = {s}'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print('观察：CFG=0 两团都有（无条件），CFG=1 有条件但有串味，CFG=3/7 越来越锐化到目标类别')"""),

    md("""## 面试题

**Q1：CFG 和 Classifier Guidance 的本质区别？**
CFG 用一个网络同时训 conditional+unconditional，推理时用两者差值放大；Classifier Guidance 需要额外训一个**能处理带噪图像**的分类器。CFG 工程简单、效果更好，是当前主流。

**Q2：训练时 drop 概率为什么是 10%？**
经验值。太低（<5%）uncond 分支学不好；太高（>30%）cond 分支质量下降。10-20% 是甜点。SD 用 10%。

**Q3：CFG scale 越大越好吗？**
不是。scale 过大（>15）会出现 **saturation/over-saturation**：颜色过饱和、细节扭曲、多样性崩塌。SD 默认 7-7.5，SDXL 5-7。

**Q4：CFG 的代价是什么？**
**推理慢一倍**。每步要 forward 两次（cond + uncond）。实际工程把两者拼 batch 一次前向，但显存翻倍。

**Q5：CFG 在 Flow Matching 里还用吗？**
**用，而且效果更好**。SD3/FLUX 都用 CFG，因为它本质是在分数/速度场上的线性外推，和算法无关。

**Q6：为什么 CFG 能 work（直觉解释）？**
从 `[ε_c - ε_u]` 方向走相当于沿着**"最像条件"的方向**加速。数学上等价于锐化条件分布 `p(x|c)^s`。"""),
]

save("04_cfg.ipynb", cells_04)


# =====================================================================
# 05 Score Matching — pure understanding, no code needed
# =====================================================================
cells_05 = [
    md("""# 05 · Score Matching 视角（理解向，无代码）

## 一句话直觉

**预测噪声** ε 和 **预测分数** ∇log p(x) 本质是同一件事，只差一个系数。理解这个等价关系，能让你在 Diffusion / Flow Matching / Langevin 之间自由切换。

## 为什么这一节很重要（但不需要撕代码）

面试官问："你能用两句话解释 Diffusion 和 Score Matching 的关系吗？" —— 能答出来的人不到 20%。

这是**概念打通**的节点，不是手撕点。"""),

    md("""## 什么是 Score？

对于一个概率分布 p(x)：

$$s(x) = \\nabla_x \\log p(x)$$

**几何直觉**：分数 = 指向"数据更密集方向"的箭头。
- 在数据点附近，分数指向数据中心
- 远离数据的地方，分数非常大（指引你"往回走"）
- 数据众数（mode）处，分数 = 0

### 类比：山谷里的水流

把数据分布想象成山谷的地形：
- **谷底** = 高概率数据点
- **山顶** = 低概率区域
- **score** = 重力方向（永远指向最近的谷底）

只要你拿到 score，就能用 **Langevin 动力学**：

$$x_{t+1} = x_t + \\epsilon \\cdot s(x_t) + \\sqrt{2\\epsilon} \\cdot z$$

从任何点出发，跟着 score 走，就能"沿着山势滚到谷底"，即采样到高概率数据点。"""),

    md("""## 关键等式：ε-prediction ↔ score

Diffusion 前向公式：
$$x_t = \\sqrt{\\bar\\alpha_t} \\cdot x_0 + \\sqrt{1 - \\bar\\alpha_t} \\cdot \\epsilon$$

两边关于 x_t 求导（利用高斯分布性质）：

$$\\nabla_{x_t} \\log p(x_t | x_0) = -\\frac{\\epsilon}{\\sqrt{1-\\bar\\alpha_t}}$$

**翻译成人话**：

| 物理量 | 符号 | 关系 |
|--------|------|------|
| 加进去的噪声 | ε | DDPM 直接预测 |
| 数据分布的分数 | s = ∇log p | Score Matching 直接预测 |
| **两者关系** | | **s = -ε / √(1-ᾱ)** |

**所以**：
- DDPM 训练 `MSE(ε_pred, ε_true)` → 等价于训练 score network
- DDPM 的反向采样 → 等价于 **reverse SDE 解方程**
- DDIM 的确定性采样 → 等价于 **probability flow ODE**

一套数学两种语言。"""),

    md("""## 家族地图：从 Score 出发看整个生成模型世界

```
                    Score s(x) = ∇log p(x)
                           │
            ┌──────────────┼──────────────┐
            │              │              │
    Reverse SDE      Probability      Langevin
    (DDPM 的本体)     Flow ODE          (MCMC)
                   (DDIM 等)
            │              │              │
      随机轨迹        确定性轨迹      可以 mix
```

### 三种视角对同一件事的描述

**视角 1（DDPM 原论文）**：学去噪任务 `ε_θ(x_t, t) ≈ ε`
**视角 2（Score Matching / Song & Ermon）**：学 score network `s_θ(x_t, t) ≈ ∇log p_t(x_t)`
**视角 3（Diffusion SDE / Song 2021）**：前向 SDE 的时间反转给出一个 reverse SDE，里面出现了 score

**它们训出来的网络完全等价**，只差一个变量替换。但**视角决定思考方式**：

| 视角 | 思考方式 | 帮你推出 |
|------|----------|----------|
| ε-pred | "我在去噪" | DDPM 采样公式 |
| score | "我在估计密度梯度" | Langevin MCMC、Flow Matching |
| SDE | "我在解随机微分方程" | DDIM 推导、SDE solvers |"""),

    md("""## 为什么 2022 后主流转向 score / flow 视角

1. **统一框架**：score + ODE 框架能同时解释 Diffusion、Flow Matching、Consistency Models
2. **求解器通用**：现成的 ODE solver（Euler、Heun、DPM-Solver）可以直接复用
3. **Flow Matching 的跳板**：Flow Matching 就是"不走 score 这条弯路，直接学一个更好的速度场"——只有在 score 视角下才能看清它是怎么来的

### Diffusion → Flow Matching 的视角跳跃

| | Diffusion | Flow Matching |
|---|-----------|---------------|
| 学什么 | score / noise | velocity field v |
| 路径 | 随机 SDE | 确定性 ODE |
| 采样 | 50-1000 步 | 4-50 步 |
| 和 score 关系 | 本体 | 退化形式（直线路径下等价） |"""),

    md("""## 面试题

**Q1：DDPM 训的是 ε-prediction，为什么叫 "score-based model"？**
ε 和 score 只差系数：s = -ε / √(1-ᾱ)。预测 ε 等价于预测 score，只是参数化不同。Song 2021 统一了这一切到 SDE 框架下。

**Q2：DDIM 确定性采样的数学本质是什么？**
反向 SDE 去掉随机项 → **Probability Flow ODE**。它和 reverse SDE 共享同一个边际分布 p_t(x)，但轨迹是确定的，所以可以用 Euler/Heun 等 ODE solver 加速。

**Q3：score 能不能直接拿来做 Langevin 采样？为什么 Diffusion 不直接用？**
可以。但**纯 Langevin 需要高密度区域的 score 足够准**，而数据稀疏区域 score 估计极不准。Diffusion 加噪的精妙之处：通过大量噪声把数据"糊开"，让每个 t 下的 score 都好估计，然后从 T 逐步走回 0。

**Q4：一句话说清楚 Flow Matching 怎么从 Score 视角推出来？**
Flow Matching 放弃"通过 score 推 velocity"的间接路径，直接**参数化 velocity field v(x,t)**，用一个**比 score matching 更简单的 L2 回归**训。在直线路径（Rectified Flow）下，它等价于 score matching 的一种特殊参数化，但训练和采样都更好。"""),
]

save("05_score_matching.ipynb", cells_05)


# =====================================================================
# 06 Flow Matching — 🔥 mature tech, heavy explanation + core code
# =====================================================================
cells_06 = [
    md("""# 06 · Flow Matching 基础 🔥

## 一句话直觉

**不学加噪去噪那套绕路**，直接学一个**向量场**：告诉每个点"现在该往哪儿走"。沿着箭头直线走，就到目的地。

**这是 SD3 / FLUX / Sora 的当前底座。2026 年做生成默认起手式。**"""),

    md("""## 类比：湖面树叶

想象一片湖面（噪声分布 p_0）和湖底的宝藏（数据分布 p_1）。

- **Diffusion**：扔一片树叶，让它**随机漂**，每一步稍微修正方向，**50-1000 步**漂到目标。走的是**布朗运动**。
- **Flow Matching**：在湖里装一套**水流系统**（向量场），树叶顺着水流**直线**过去，**4-8 步**到家。走的是**确定性 ODE**。

水流每一点的箭头怎么定？—— 就是 Flow Matching 要学的东西。"""),

    md("""## Motivation：为什么要抛弃 Diffusion

### Diffusion 三大痛

1. **采样步数多**：50-1000 步，每步都要 NN forward，实时场景（数字人、游戏 NPC）根本扛不住
2. **路径弯**：DDPM 反向是带随机项的 SDE，走的是 Z 字形路径，必然需要多步才能补偿
3. **训练/推理不一致**：训时在扰动的边际分布上，推时累积误差，需要 schedule 精调

### Flow Matching 的思想

"我不需要 score 这个中间变量。**我直接参数化速度场** `v(x, t)`，用最简单的 L2 回归学它。"

**目标**：学一个向量场 v_θ(x, t)，使得从 noise 沿着 v 积分 T 次，就能到 data：

$$\\frac{dx}{dt} = v_\\theta(x, t), \\quad x(0) = noise, \\quad x(1) = data$$"""),

    md("""## 核心算法一页纸

### 1. 定义路径（条件概率路径）

最简单选择：**线性插值**（Rectified Flow 风格）

$$x_t = (1-t) \\cdot x_0 + t \\cdot x_1, \\quad t \\in [0, 1]$$

其中 x_0 ~ N(0, I)（噪声），x_1 ~ data。

对 t 求导，这条路径上每个点的**真实速度**：

$$v_{true}(x_t, t) = \\frac{dx_t}{dt} = x_1 - x_0$$

### 2. CFM Loss（Conditional Flow Matching）

**不需要知道边际速度场**（那是个积分，算不出来），只需要训条件速度场：

$$\\mathcal{L}_{CFM} = \\mathbb{E}_{t, x_0, x_1} \\left[ \\| v_\\theta(x_t, t) - (x_1 - x_0) \\|^2 \\right]$$

**一行代码**：
```python
t = torch.rand(B, 1)
x0 = torch.randn_like(x1)
xt = (1-t)*x0 + t*x1
v_pred = model(xt, t)
loss = F.mse_loss(v_pred, x1 - x0)   # 就这么简单
```

### 3. 采样（纯 ODE）

```python
x = randn()                          # 从噪声出发
for t in linspace(0, 1, N):         # N 可以很小，4-50
    x = x + v_theta(x, t) * dt      # Euler 积分
return x
```

**对比 DDPM**：没有 α_bar、没有噪声项、没有 cumprod。就是**一个 Euler ODE**。"""),

    md("""## Flow Matching vs Diffusion 对比表

| 维度 | DDPM | Flow Matching |
|------|------|---------------|
| **训练目标** | MSE(ε_pred, ε) | MSE(v_pred, x1 - x0) |
| **schedule** | 需要 β 调参 | 线性路径，无需调参 |
| **路径** | SDE 随机 | ODE 确定 |
| **采样步数** | 50-1000 | 4-50 |
| **求解器** | 特制（DDIM/DPM-Solver） | 通用 ODE solver |
| **训练稳定性** | 对 schedule 敏感 | 稳 |
| **scaling law** | 有天花板 | 更清晰 |
| **代表模型** | SD 1.5 / SDXL | SD3 / FLUX / Sora |

### SD3 论文的关键发现（2024）

在**大规模训练**下，Flow Matching + **Logit-Normal 时间采样**显著优于 Diffusion 的 v-prediction。

**Logit-Normal**：采样 t 时用 `t = sigmoid(normal(0, 1))` 而不是均匀，让训练更关注中间时刻（噪声和数据的过渡区）。"""),

    code("""# 最小 Flow Matching demo：2D 双月亮 → 单高斯
# 训练一个 2D 向量场，4 步采样能到目标
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
torch.manual_seed(0)

# 双月亮数据
def two_moons(n=2000):
    from sklearn.datasets import make_moons
    x, _ = make_moons(n, noise=0.05)
    return torch.tensor(x, dtype=torch.float32) * 2

data = two_moons()

# 速度场 MLP
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.SiLU(),  # 2(x) + 1(t)
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2))
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

model = VectorField()
opt = torch.optim.Adam(model.parameters(), 3e-3)

# 核心训练循环（就是 CFM loss，简单到离谱）
for step in range(3000):
    idx = torch.randint(0, len(data), (256,))
    x1 = data[idx]
    x0 = torch.randn_like(x1)
    t = torch.rand(256, 1)
    xt = (1-t)*x0 + t*x1          # 线性路径
    v_true = x1 - x0              # 真实速度（直线 → 常数）
    v_pred = model(xt, t)
    loss = F.mse_loss(v_pred, v_true)
    opt.zero_grad(); loss.backward(); opt.step()
print(f'final CFM loss: {loss.item():.4f}')""",),

    code("""# Euler ODE 采样：4 步 vs 50 步对比
@torch.no_grad()
def sample_fm(model, n, steps):
    x = torch.randn(n, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i*dt)
        x = x + model(x, t) * dt
    return x

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, steps in zip(axes, [4, 10, 50]):
    samples = sample_fm(model, 1000, steps)
    ax.scatter(data[:,0], data[:,1], s=3, alpha=0.3, label='data')
    ax.scatter(samples[:,0], samples[:,1], s=3, c='red', alpha=0.5, label=f'FM {steps} steps')
    ax.set_title(f'Flow Matching, {steps} steps'); ax.legend()
plt.tight_layout(); plt.show()
print('观察：4 步就能看到双月亮形状，50 步已经很接近 data。')
print('对比 DDPM 50 步常见的模糊，Flow Matching 细节更清楚。')"""),

    md("""## 面试题

**Q1：Flow Matching 比 Diffusion 快在哪？**
**路径直，求解器好**。Diffusion 反向是随机 SDE，必须多步走；Flow Matching 学确定性 ODE，可以用 Euler/Heun 等高阶 solver 几步到位。训练 loss 也更稳定。

**Q2：CFM loss 为什么不需要边际速度场？**
边际速度场是个条件期望，计算复杂。关键定理：**训练条件速度场 v(x_t, t | x_0, x_1) = x_1 - x_0 的 L2 loss**，最优解等价于边际 loss。这是 Flow Matching 论文的核心贡献。

**Q3：Flow Matching 和 Score Matching 的关系？**
Rectified Flow 的直线路径下，v 和 score 可以互相换算。但 Flow Matching 训练更简单（不需要噪声 schedule），推理更快（ODE 直接走）。

**Q4：为什么 SD3 / FLUX / Sora 都转向 Flow Matching？**
大规模训练下 FM **scaling 更好、调参更少、采样更快**。SD3 论文证实 FM + Logit-Normal 时间采样显著优于 Diffusion v-prediction。工业部署还能蒸馏到 1-4 步（DMD2 / LCM）。

**Q5：Flow Matching 还能用 CFG 吗？**
能。`v_guided = v_uncond + s·(v_cond - v_uncond)`，和 Diffusion 的 CFG 完全同构。FLUX 默认用 CFG。

**Q6：Logit-Normal 时间采样是干嘛的？**
SD3 引入：`t = sigmoid(N(0,1))`，让 t 集中在 0.5 附近。中间时刻最难学（既不是纯噪声也不是纯数据），多采样一些让训练更聚焦。"""),
]

save("06_flow_matching.ipynb", cells_06)


# =====================================================================
# 07 Rectified Flow + Reflow — explanation + minimal code
# =====================================================================
cells_07 = [
    md("""# 07 · Rectified Flow + Reflow 🔥

## 一句话直觉

**第一次训的 Flow Matching 路径还有点弯**，用模型自己生成的 (noise, image) 配对**再训一遍**，路径会变直。**越直 → 采样步数越少 → 极限 1 步出图**。

## 核心升级

**Rectified Flow** = Flow Matching 的**最直版本**，专门追求"**每条路径都是直线**"。是 FLUX、SD3 的直接前身。"""),

    md("""## 类比：修高速公路

第一代：你在一片山地上要建从 A（噪声）到 B（数据）的路。
- **Flow Matching v1**：按地形修路，绕了几个弯，车开过去要减速转向（多步采样）
- **Reflow**：把第一版的轨迹记下来 → 重新规划一条"**把所有弯都拉直**"的版本 → 车可以一脚油门到底（少步采样）

**极限情况**：路径完全是直线 → 一步跨过去。这就是 1-step 生成的数学基础。"""),

    md("""## 算法：Reflow 的三步

### Step 1：训第一版 Flow Matching（v1）

```python
x_0 ~ N(0, I)     # 任意噪声
x_1 ~ data         # 真实数据
x_t = (1-t)*x_0 + t*x_1
v1 = train(MSE(v_pred, x_1 - x_0))
```

**问题**：采样时，从不同 x_0 出发的轨迹可能**交叉**（两条路径在某个点速度方向不一致）。模型学到的是"平均速度"，实际路径不是直线。

### Step 2：用 v1 生成"对齐"的配对

```python
for x_0 in randn(N):
    x_1_hat = ODE_solve(v1, from=x_0, to=t=1)   # 跑完整个 ODE
    save (x_0, x_1_hat)                          # 记录首尾配对
```

关键洞察：这些配对是 **"v1 模型自己认可的轨迹终点"**，没有交叉冲突。

### Step 3：在新配对上重训 v2（Reflow）

```python
x_0, x_1_hat = sample_pair()
x_t = (1-t)*x_0 + t*x_1_hat
v2 = train(MSE(v_pred, x_1_hat - x_0))
```

**v2 学到的路径比 v1 更直**，因为训练数据本身就是 v1 的"一条龙"结果。

### 可以 reflow 多次

reflow 2 次 → 3 次 → N 次，路径越来越直，最终趋近于"一步跨到位"的 shortcut。但边际收益递减，实际 1 次 reflow 就够。"""),

    md("""## Rectified Flow vs 普通 Flow Matching

| 维度 | Vanilla FM | Rectified Flow |
|------|-----------|----------------|
| 路径 | 条件直线，但整体可能弯 | 追求**全局**最直 |
| 训练轮次 | 1 次 | 1 次训 + 1-2 次 reflow |
| 采样步数 | 10-50 | 4-10（1 步也能用） |
| 1 步质量 | 差 | 可接受 |
| 工程成本 | 低 | 多跑一轮 reflow |

## 工业应用

- **FLUX.1**（Black Forest Labs, 2024）：Rectified Flow + DiT，开源最强开源文生图模型之一
- **SD3**：Rectified Flow + MMDiT
- **InstaFlow**（2023）：SD1.5 + reflow → **1 步出图**，质量接近原 50 步"""),

    md("""## Reflow 为什么能拉直路径？（直觉解释）

**轨迹交叉 = 弯路的根源**

想象两条路径：
- 路径 A：(0, 0) → (0, 1) → (1, 1)
- 路径 B：(1, 0) → (1, 1) → (0, 1)

它们在 (0.5, 1) 附近交叉。Flow Matching v1 在交叉点学到的是"平均方向" → 预测的 v 指向模糊位置 → 实际采样轨迹是弯的。

**Reflow 做的事**：
用 v1 从 (0, 0) 跑一遍，得到真实终点 (1, 1)，记录配对 **(0, 0) → (1, 1)**。
用 v1 从 (1, 0) 跑一遍，得到终点 (0, 1)，记录配对 **(1, 0) → (0, 1)**。

现在配对 **不再交叉**（v1 保证了不交叉的 coupling），重新训 v2 时每个点只有一个方向，路径自然变直。"""),

    code("""# 极小 Reflow demo：2D 高斯 → 双月亮，观察路径直度
# 为节省时间只跑关键对比
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
torch.manual_seed(0)

x1_data = torch.tensor(make_moons(2000, noise=0.05)[0], dtype=torch.float32) * 2

class VF(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3,128), nn.SiLU(),
                                 nn.Linear(128,128), nn.SiLU(),
                                 nn.Linear(128,2))
    def forward(self, x, t): return self.net(torch.cat([x,t],-1))

def train_fm(pairs_x0, pairs_x1, steps=2000):
    m = VF(); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(steps):
        idx = torch.randint(0, len(pairs_x0), (256,))
        x0, x1 = pairs_x0[idx], pairs_x1[idx]
        t = torch.rand(256, 1)
        xt = (1-t)*x0 + t*x1
        loss = F.mse_loss(m(xt,t), x1-x0)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

# v1：随机配对（纯 FM）
x0_rand = torch.randn_like(x1_data)
v1 = train_fm(x0_rand, x1_data)
print('v1 trained (random coupling)')

# v1 生成配对 → reflow
@torch.no_grad()
def ode_sample(m, x0, steps=50):
    x = x0.clone()
    dt = 1/steps
    for i in range(steps):
        t = torch.full((len(x),1), i*dt)
        x = x + m(x,t)*dt
    return x

x0_new = torch.randn(2000, 2)
x1_hat = ode_sample(v1, x0_new)
v2 = train_fm(x0_new, x1_hat)           # reflow
print('v2 trained (reflow coupling)')"""),

    code("""# 对比：v1 vs v2 在少步数下的采样质量
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for col, steps in enumerate([1, 4, 20]):
    for row, (name, m) in enumerate([('v1 (FM)', v1), ('v2 (Reflow)', v2)]):
        ax = axes[row, col]
        s = ode_sample(m, torch.randn(500,2), steps=steps)
        ax.scatter(x1_data[:,0], x1_data[:,1], s=3, alpha=0.3)
        ax.scatter(s[:,0], s[:,1], s=3, c='red', alpha=0.5)
        ax.set_title(f'{name}, {steps} step(s)')
        ax.set_xlim(-3,4); ax.set_ylim(-2,3)
plt.tight_layout(); plt.show()
print('观察：v2 (Reflow) 在 1-4 步下明显比 v1 质量更好，路径被拉直了。')"""),

    md("""## 面试题

**Q1：Rectified Flow 比普通 Flow Matching 多做了什么？**
多做了一次 **Reflow**：用训好的 v1 生成 (x0, x1_hat) 配对，在这个新配对上重训 v2。关键是新配对**不交叉**，v2 学到的路径因此更直。

**Q2：Reflow 的理论依据？**
Flow Matching 的配对是**随机**的（任意 x0 对任意 x1），导致路径交叉。Reflow 用 ODE 解出的配对是**确定性 coupling**（类似 optimal transport 的弱化版），几何上不交叉，重训后路径自然拉直。

**Q3：InstaFlow 怎么做到 1 步出图？**
在 SD1.5 上做 2 次 Reflow + 蒸馏（K-Rectified Flow）。路径几乎完全拉直后，Euler 1 步足以到达数据分布。质量接近原 50 步 DDIM。

**Q4：FLUX 用的是 Rectified Flow 还是 DMD？**
**Rectified Flow 是架构基础**，部署时可能额外蒸馏到少步。FLUX.1-schnell 是 4 步版本，就是在 dev（标准多步）基础上蒸馏得到的。

**Q5：Reflow 有坏处吗？**
两点：① 需要多跑一轮训练（成本翻倍+）；② v1 如果有偏差，Reflow 会**放大偏差**（v2 学的是 v1 的生成物）。实际里 1 次 Reflow 是甜点。"""),
]

save("07_rectified_flow.ipynb", cells_07)

print("--- batch 1 done ---")
