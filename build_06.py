"""06 Flow Matching - 当前工业主流（SD3/FLUX/Movie Gen 都换了）。理解为主 + 核心代码只撕最关键的。"""
import json

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})

md("""# 06 - Flow Matching（现代生成的主流范式）

> **这一篇极重要**。2024-2026 年新出的主流大生成模型（SD3 / FLUX.1 / Movie Gen，Sora 未公开、社区推测同路线）**纷纷从传统 Diffusion 转向 Flow Matching**。面试问"最新生成技术"答不上这个，基本凉凉。

**一句话直觉**：别再玩"加噪→去噪"那套绕弯路的游戏了。**直接学一个向量场**，告诉空间里每一点"现在该往哪走"，沿着箭头走到数据分布。

**类比 — 湖面和湖底**：
- **Diffusion 视角**：湖面扔一片树叶，让它随机漂（SDE），但每一步修正一下方向。能到湖底但绕弯。
- **Flow Matching 视角**：在湖里装一套水流系统（向量场 $v$）。树叶顺着水流走，**走直线**到湖底。

**数学对偶（核心）**：
- DDPM/DDIM 学的是 $\\epsilon_\\theta$ 或 score $s_\\theta$
- Flow Matching 学的是 **速度场 $v_\\theta(x_t, t)$**
- 采样 = 解 ODE：$\\dot{x}_t = v_\\theta(x_t, t)$

### 本课符号表（06/07/08 通用）

| 符号 | 含义 | 直觉 |
|------|------|------|
| $x_0$ | 纯噪声 $\\sim \\mathcal{N}(0,I)$ | 出发点 |
| $x_1$ | 真实数据 | 终点 |
| $t \\in [0,1]$ | 连续时间 | 走了几成路程 |
| $x_t$ | 插值点 $(1-t)x_0 + t x_1$ | 路上的位置 |
| $v_\\theta(x_t, t)$ | 网络学的速度场 | 水流：这一点该往哪走 |
| $u_t(x)$ | 真实（边际）速度场 | 理想答案，直接算不出来 |

> ⚠️ **方向警告（全系列最大混淆源）**：本课约定 **$t=0$ 是噪声、$t=1$ 是数据**，与 01-05 DDPM 的"$t$ 越大越噪"**正好相反**。后面看到任何 $t$，先想这句。""")

md("""## Part 0 — 为什么要换？Diffusion 的三个硬伤

| 问题 | Diffusion 的痛 | FM 怎么解决 |
|------|----------------|-------------|
| **路径弯曲** | 反向 SDE 的轨迹曲折，50-1000 步才够 | 学"直线路径"，4-50 步就够 |
| **噪声 schedule 调参黑魔法** | linear / cosine / v-pred 一堆花活，调不好就崩 | 路径设计直接由公式给死 |
| **理论框架割裂** | 训练用 $\\epsilon$-pred，采样用 SDE/ODE，CFG 又回到 score… 四五套符号互相翻译 | 一个视角（ODE + 向量场）搞定训练 + 采样 + 条件 + guidance |

**SD3 论文（Esser et al., 2024）的实证**：在相同计算预算下，Flow Matching + Rectified Flow 在 FID、CLIP score、人类评价三个维度都**显著优于**原 DDPM/v-pred 配方。这是工业界集体换赛道的直接原因。

**FLUX.1（Black Forest Labs, 2024）**：SD 原团队出走后的新作，从零选型就用 Flow Matching + DiT + Rectified Flow。

**视频生成（Movie Gen 等）**：时序维度让每一步推理都极贵，步数需要大幅压缩（从上百步压到几步~几十步）—— Flow Matching 这种"路径可拉直"的范式天然适合做这件事。""")

md("""## Part 1 — Flow Matching 算法一页纸

### 核心对象：**概率路径** $p_t(x)$

想象从 $t=0$（纯噪声 $p_0 = \\mathcal{N}(0, I)$）到 $t=1$（数据 $p_1 = p_{data}$）的一族分布演化。FM 的目标就是**找一个向量场 $v_t(x)$ 把 $p_0$ 连续推到 $p_1$**。

接回开头的湖面类比：$p_t$ 是"第 $t$ 秒全湖树叶的分布"，$u_t$ 是"真实的水流"，我们要装的那套水流系统就是 $v_\\theta$。

### 训练目标（梦想版）：**Flow Matching Loss**

$$
\\mathcal{L}_{FM} = \\mathbb{E}_{t,\\, x \\sim p_t} \\Big[ \\| v_\\theta(x, t) - u_t(x) \\|^2 \\Big]
$$

其中 $u_t(x)$ 是"真实的速度场"。**痛点**：$u_t$ 我们不知道，$p_t$ 的积分形式也不可计算。

**白话拆解**：这个 loss 就是个回归——网络输出 $v_\\theta$，标准答案 $u_t$，MSE 逼近。问题全在"标准答案"上：边际速度 $u_t(x)$ 要把"所有可能生成 $x$ 的数据点"的方向加权平均（分母还带一个算不出的 $p_t(x)$）。就像问"此刻路过天安门的所有人平均要去哪"——你枚举不完所有人。CFM 的破局思路：每次只盯一个人（条件在具体的 $x_1$ 上），他要去哪是明确的直线方向，直接可算。

### 训练目标（落地版）：**Conditional Flow Matching (CFM)**

Lipman 等人 2023 的关键发现：把 $u_t$ 拆成**条件速度场** $u_t(x | x_1)$，其中 $x_1$ 是某个具体的数据点。然后：

$$
\\boxed{\\; \\mathcal{L}_{CFM} = \\mathbb{E}_{t,\\, x_1 \\sim p_{data},\\, x_0 \\sim \\mathcal{N}(0,I)} \\Big[ \\| v_\\theta(x_t, t) - (x_1 - x_0) \\|^2 \\Big] \\;}
$$

**读式子**：期望里有三个随机源 $t$ / $x_1$ / $x_0$，训练时各采一次；回归目标 $(x_1 - x_0)$ 是**常数向量**——不含 $\\theta$、不含 $t$；极端代入检查：$t=0$ 时 $x_t = x_0$（噪声端）、$t=1$ 时 $x_t = x_1$（数据端）✓。

**核心 trick**：选最简单的**直线路径**
$$
x_t = (1-t) \\cdot x_0 + t \\cdot x_1, \\qquad u_t(x_t | x_0, x_1) = x_1 - x_0
$$

**面试必答点 — 为什么这个 loss 能用来替代原 FM loss**：Lipman 证明了 $\\nabla_\\theta \\mathcal{L}_{FM} = \\nabla_\\theta \\mathcal{L}_{CFM}$。两个 loss 梯度相同，所以优化 CFM 就等于优化 FM。这是 Flow Matching 工业可用的关键证明。

<details><summary>为什么梯度相等（选读，30 秒版）</summary>

把两个 loss 的平方项都展开成 $\\|v_\\theta\\|^2 - 2\\langle v_\\theta, \\text{target}\\rangle + \\|\\text{target}\\|^2$。第一项两边相同；第二项交换期望顺序后相等（条件速度对 $x_1$ 取期望恰好是边际速度）；第三项不含 $\\theta$，求导直接消掉。所以 $\\mathcal{L}_{FM}$ 和 $\\mathcal{L}_{CFM}$ 只差一个与 $\\theta$ 无关的常数项，梯度完全相同。

</details>

**记住：是梯度相等，不是 loss 值相等。**

### 采样：**Euler ODE**

训练完后，给 $x_0 \\sim \\mathcal{N}(0, I)$，按 Euler 方法解 ODE：
$$
x_{t+\\Delta t} = x_t + \\Delta t \\cdot v_\\theta(x_t, t)
$$

10-50 步就够。没有随机性（默认情况），完全确定。""")

md("""## Part 2 — FM 和 Diffusion 的对应关系（面试重点）

很多面试官会问"FM 是不是 DDPM 的马甲"，答案是"**它是 DDPM 的连续+直线化版本**"，但两者不等价。

**怎么读下面这张表**：逐行读，每一行都是面试追问点；最重要的是第 2 行（**路径几何**），其余行基本都是它的推论。

| 维度 | DDPM (ε-pred) | Flow Matching (v-pred) |
|------|---------------|------------------------|
| **学习目标** | 噪声 $\\epsilon$ | 速度 $v = x_1 - x_0$ |
| **路径** | $x_t = \\sqrt{\\bar\\alpha_t}x_0 + \\sqrt{1-\\bar\\alpha_t}\\epsilon$（弯） | $x_t = (1-t)x_0 + t x_1$（直） |
| **noise schedule** | 一堆花哨参数（β, ᾱ, σ）要调 | 只有 $t \\in [0,1]$ |
| **采样** | DDPM 是 SDE，DDIM 是离散 ODE | 标准 ODE，任意 solver |
| **条件/CFG** | 用 score trick 推导 | 直接在 $v$ 上做差（更干净） |
| **步数** | 20-50 步（DDIM） | **4-50 步** |

**精妙对应（v-pred 之谜的统一解释）**：SD2.1 用过的 "v-prediction" loss，和 FM 在**高斯路径下是仿射等价的亲缘关系**（不是同一个东西，但可互相换算）。Salimans & Ho 定义的 $v = \\alpha_t \\epsilon - \\sigma_t x_0$（符号落位：这里的 $\\alpha_t, \\sigma_t$ 就是 01 课 $\\sqrt{\\bar\\alpha_t}, \\sqrt{1-\\bar\\alpha_t}$ 的简写）和 FM 的 $v = x_1 - x_0$ 在高斯路径下可精确换算。这就是为什么 SD3 说"我们从 v-pred 迁移到 Rectified Flow 非常丝滑"——**它们本来就是亲戚**。""")

md("""## Part 3 — 代码手撕（只撕面试必考的 2 段）

按新方针，这一节**不做完整训练**（训练循环和 02 DDPM 几乎一样，只是 loss 和采样换了）。只把**面试现场最可能让你手写**的两段剥出来：

1. **CFM loss 一步计算**（考察你懂不懂训练目标）
2. **Euler ODE 采样循环**（考察你懂不懂推理）

> 如果面试官让你"从零写一个 Flow Matching"，这两段背下来就够了。""")

code("""import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 面试手撕 #1：Conditional Flow Matching loss（训练端核心）
# ============================================================
def cfm_loss(model, x1, device='cpu'):
    \"\"\"
    给定一个 batch 的真实数据 x1，算一次 CFM loss。
    面试时能白板默写这 5 行 = 懂 FM 训练。

    数学对应：
        x_0 ~ N(0, I)
        t   ~ U(0, 1)
        x_t = (1-t)·x_0 + t·x_1
        v_target = x_1 - x_0   (直线路径的速度 = 终点-起点)
        loss = || v_theta(x_t, t) - v_target ||^2
    \"\"\"
    B = x1.shape[0]
    x0 = torch.randn_like(x1)                  # 起点：纯噪声
    t = torch.rand(B, device=device)           # t ~ U(0,1), 逐样本一个 t
    # 广播到 x 的形状（batch, ...）
    t_shape = (B,) + (1,) * (x1.ndim - 1)
    t_b = t.view(t_shape)

    x_t = (1 - t_b) * x0 + t_b * x1            # 直线插值
    v_target = x1 - x0                         # 真实速度（常数向量）
    v_pred = model(x_t, t)                     # 网络预测速度
    return F.mse_loss(v_pred, v_target)

# ============================================================
# 面试手撕 #2：Euler ODE 采样（推理端核心）
# ============================================================
@torch.no_grad()
def euler_sample(model, shape, steps=50, device='cpu'):
    \"\"\"从纯噪声跑到数据分布。面试现场 10 行以内写完。
    核心：x_{t+dt} = x_t + dt · v_theta(x_t, t)
    \"\"\"
    x = torch.randn(shape, device=device)      # t=0: 噪声
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        v = model(x, t)
        x = x + dt * v                         # Euler 一步
    return x                                    # t=1: 数据

# ============================================================
# 验证这两个函数能跑通（用一个哑 model 即可，不真训）
# ============================================================
class DummyVelocityNet(nn.Module):
    \"\"\"只为了验证 API 通顺，不学任何东西。\"\"\"
    def __init__(self, dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+1, 64), nn.SiLU(), nn.Linear(64, dim))
    def forward(self, x, t):
        if x.ndim > 2: x = x.flatten(1)
        t = t.view(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t], dim=-1))

torch.manual_seed(0)
model = DummyVelocityNet(dim=16)
x1 = torch.randn(8, 16)
loss = cfm_loss(model, x1)
samples = euler_sample(model, (4, 16), steps=20)
print(f'CFM loss    : {loss.item():.4f}')
print(f'采样 shape   : {samples.shape}')
print(f'采样均值/方差 : {samples.mean().item():.3f} / {samples.std().item():.3f}')
print('API 通顺 ✓ （没真训，所以分布无意义，但代码骨架对）')""")

md("""## Part 4 — Flow Matching 的工业现状（2026 年视角）

### 主流生成模型阵营（按路线划分）

| 阵营 | 代表 | 范式 |
|------|------|------|
| **传统 Diffusion** | SD 1.5 / SDXL / Stable Video Diffusion (EDM) | DDPM/EDM + U-Net |
| **Flow Matching 新生代** ⭐ | SD3 / FLUX.1 / Movie Gen / AuraFlow（Sora 未公开，社区推测同路线） | **CFM/Rectified Flow + DiT** |
| **一步生成** | FLUX-schnell / SDXL-Turbo / LCM-SDXL | 从 FM/Diffusion 蒸馏 1-4 步 |

（注：Midjourney 架构从未公开，不列入对比。）

### 关键细节（面试加分点）

**1. SD3 的 Logit-Normal 时间采样**
训练时 $t$ 不均匀采，而是从 `logit-normal(0, 1)` 采样（中间 $t$ 多、两端少）。因为中间 $t$ 的 loss 方差大、信号多，训练更高效。

**2. SD3 的 Timestep Shift + Logit-Normal 加权**
高分辨率下同一个 $t$ 的信噪比更高，SD3 对 $t$ 分布做平移（timestep shift），配合 Logit-Normal 采样等效于给不同 $t$ 不同的 loss 权重 —— 这是 FM 的灵活性所在（Diffusion 时代换 schedule 很痛苦）。

**3. 条件注入方式**
SD3 / FLUX 都把文本条件从 U-Net 的 cross-attention 换成了 **MM-DiT 的 concat + self-attn**（文本 token 和图像 token 一起进 transformer）。这也是 FM 路线更自然接入 DiT 的原因（见 09 DiT）。""")

md("""## Part 5 — 对你（金融机构数字人 / AIGC 面试）的直接关联

**1. 视频数字人（EMO / Hallo / Sonic 这些路线）**
现在的视频数字人还在用 Diffusion（UNet3D），推理 25-50 步，1 秒视频要算几秒。**下一代必然换 Flow Matching**——因为视频的时序维度让步数成本爆炸，FM 的"少步+直线"是唯一出路。

**2. 信创部署**
FLUX / SD3 的权重已公开，DiT + FM 架构在 Ascend 910b 上是**标准 Transformer 算子**（不像 U-Net 的卷积要做很多 kernel fusion）。信创落地比 U-Net 更友好。

**3. 面试必问话题**
- "说说 Flow Matching 和 Diffusion 的区别" → 走本 notebook Part 2 的表格
- "SD3 为什么选 Rectified Flow" → 见 07 notebook
- "FLUX.1 的技术栈" → DiT + Rectified Flow + CFG，能说清每一块的作用就是强候选""")

md("""## 面试题连问

### Q1：一句话解释 Flow Matching。

<details><summary>参考答案</summary>

训练一个**速度场** $v_\\theta(x,t)$，采样时从噪声 $x_0$ 出发，沿 ODE $\\dot x_t = v_\\theta(x_t, t)$ 积分到 $x_1$。相比 Diffusion 的"加噪→去噪"SDE，FM 直接在连续 ODE 框架下工作，路径可以是直线，步数更少，实证效果更好（SD3 / FLUX 都选它）。
</details>

### Q2：Conditional Flow Matching (CFM) 相比原始 FM loss 的核心 trick 是什么？

<details><summary>参考答案</summary>

原始 FM loss 要计算**边际速度场** $u_t(x)$，这个量依赖于 $p_t$ 的积分，没法直接算。CFM 的 trick 是给 loss **添加一个条件变量** $x_1$（具体数据点），这样条件速度场 $u_t(x|x_1) = x_1 - x_0$ 变成**常数向量**，可以直接算。

Lipman (2023) 证明：**$\\nabla_\\theta \\mathcal{L}_{FM} = \\nabla_\\theta \\mathcal{L}_{CFM}$**（两者梯度期望相同），所以优化 CFM 等价于优化原 FM。这是整个范式落地的关键理论支撑。
</details>

### Q3：Flow Matching 和 DDPM 的训练目标有什么本质区别？

<details><summary>参考答案</summary>

- **DDPM**：预测噪声 $\\epsilon$。输入带噪图，输出噪声，loss = MSE。
- **FM**：预测速度 $v = x_1 - x_0$。输入插值图，输出向量场，loss = MSE。

看似都是 MSE，但**路径几何**完全不同：
- DDPM 的 $x_t$ 是**高斯扩散轨迹**（弯曲）
- FM 的 $x_t$ 是**直线插值**（直）

直线路径 → ODE 离散化误差小 → 少步就够。这就是 FM 推理更快的根本原因。
</details>

### Q4：为什么 FM 的采样可以只要 4-8 步，而 DDPM 要 50-1000 步？

<details><summary>参考答案</summary>

本质是**路径的几何曲率**不同。
- DDPM 的反向 SDE 轨迹是强曲率的 —— Euler/RK 离散化误差大，必须小步长
- FM 的路径是**直线**（至少 Rectified Flow 后接近直线）—— 用 Euler 一步能跨一大段而不失真

进一步，**Rectified Flow** 可以通过 reflow 把路径逼近到严格直线，理论上 1 步就够（见 07 notebook）。
</details>

### Q5：SD3 为什么从 v-pred / ε-pred 迁移到 Flow Matching？

<details><summary>参考答案</summary>

SD3 论文（Esser et al., 2024）消融实验结论：在相同 FLOPs / 数据 / 架构下，**Rectified Flow + Logit-Normal 时间采样** 的 FID / CLIP score / 人类偏好全面优于 ε-pred 和 v-pred。

三个核心原因：
1. **路径直** → 少步采样时精度损失小
2. **噪声权重 schedule 设计更简单** → 消融空间小，更容易训好
3. **与 DiT 架构耦合更自然** → Transformer 对 $v$ 场的建模友好（不受卷积的 U-Net 感受野限制）

这也是 FLUX、Movie Gen 接着选 FM 的原因（Sora 未公开，社区推测同路线）。
</details>

### Q6：Flow Matching 的采样可以用哪些 ODE solver？

<details><summary>参考答案</summary>

- **Euler**（一阶，最简单）：4-50 步
- **Heun / Midpoint**（二阶）：步数砍半，精度相似
- **RK4**（四阶）：精度最高但每步要 4 次前向，不划算
- **DPM-Solver（原本给 Diffusion 用的）**：也能用于 FM，社区有改造版

**工业现状**：FLUX / SD3 的推理代码默认用 Euler（简单稳定），做加速用 Heun。RK4 基本不用（计算成本高不值得）。
</details>""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

with open('06_flow_matching.ipynb', 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'wrote 06 with {len(cells)} cells')
