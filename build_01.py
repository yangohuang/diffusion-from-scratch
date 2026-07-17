# ⚠️ STALE (2026-07-09): 01_forward_diffusion.ipynb 已于 2026-05-06 手工重写,
# 与本脚本内容完全分叉。ipynb 是唯一源头,本脚本仅存档,不要再运行它覆盖 ipynb。
"""Build 01_forward_diffusion.ipynb"""
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ============================================================
# 开篇导语
# ============================================================
md("""# 01 - Forward Diffusion 手撕

> **从这个 notebook 开始，我们进入生成模型的世界。**
> 先不急着写代码，我们先问一个根本问题：**为什么 Diffusion 能干掉 VAE 和 GAN？它凭什么？**

**一句话直觉**：Diffusion 的核心不是别的，就是把"一步生成一张图"这个困难问题，拆成"一千步里每步去一点点噪声"的简单问题。**化难为易，以慢制胜。**""")

# ============================================================
# Why Diffusion
# ============================================================
md("""## 为什么要有 Diffusion？—— VAE 和 GAN 的 3 个痛

生成模型的目标都是同一个：**学到数据分布 p(x)，然后从里面采样新样本**。

在 Diffusion 之前，两大玩家 VAE 和 GAN 各有各的坑。

### 痛点 1：VAE 的图糊 😶‍🌫️

VAE 用 **变分下界 (ELBO)** 优化，损失包含一个 **KL 项** 让 latent 空间接近高斯。

- 这个 KL 项会"压平" latent 空间 → 解码器看到的 z 变模糊
- 再加上 pixel-wise MSE 损失 → 输出是**平均值**而不是某个具体样本
- **后果**：VAE 生成的图出名地糊，像蒙了一层雾

### 痛点 2：GAN 的训练灾难 💣

GAN 用 min-max 博弈：G 骗 D，D 抓 G。

| 症状 | 表现 |
|------|------|
| Mode collapse | G 只会生成几种样本（比如猫永远是橘猫） |
| 梯度消失 | D 太强 → G 学不动 |
| 训练不稳定 | 需要大量 tricks（spectral norm / R1 penalty / progressive ...） |
| 没有 likelihood | 无法评估、无法比较模型好坏 |

**GAN 能做到的极致很高（StyleGAN 2020 年的人脸确实恐怖），但普通人训不起来。**

### 痛点 3：Autoregressive 太慢 🐢

PixelRNN / PixelCNN 一个一个像素往外吐。
- 1024×1024 图 = 100 万步 forward → **生成一张图几分钟**
- 做不了大分辨率

### Diffusion 怎么解？

- **图糊** → 每一步只学小扰动，不做 pixel MSE 平均 → **锐利**
- **训练崩** → 单个回归目标（预测噪声），损失**稳如狗**
- **没 likelihood** → ELBO 可算，能比较
- **太慢** → 并行训练，推理虽然要 1000 步，但有 DDIM/Flow Matching/Consistency 能加速到 1-50 步

**代价**：推理慢。这是 Diffusion 的**原罪**，也是后面 Flow Matching / Consistency Models 要解决的核心问题。""")

# ============================================================
# 家族地图
# ============================================================
md("""## 生成模型家族地图

Diffusion 也不是凭空出现，它是**踩着前人的坑**走出来的。

```
                    生成模型
                       |
    +------------------+------------------+
    |         |        |         |        |
  Explicit  Implicit   Score   Energy   Diffusion ← 本系列
  Density   Density    Based   Based
    |         |          |       |        |
  VAE       GAN    SBM(Song)  EBM   DDPM/DDIM/Flow/Consistency
  Flow      Autoregressive
```

| 类别 | 代表 | 核心思想 | 致命伤 |
|------|------|----------|--------|
| **VAE** | VAE, VQ-VAE | 学 encoder/decoder，优化 ELBO | 图糊、latent 空间受限 |
| **GAN** | StyleGAN, BigGAN | 对抗博弈 | 训练难、mode collapse、无 likelihood |
| **Normalizing Flow** | RealNVP, Glow | 可逆变换、精确 likelihood | 架构受限（必须可逆） |
| **Autoregressive** | PixelCNN, PixelRNN | 像素逐个生成 | 推理太慢 |
| **Score-based** | NCSN (Song 2019) | 学 ∇log p(x) | 训练 tricks 多 |
| **Diffusion** | DDPM (Ho 2020), DDIM | **前向加噪 + 反向去噪**，转化为 1000 个简单回归问题 | **推理慢** |

**Diffusion 赢在哪？** → **训练稳、生成锐利、可扩展到 10B+ 参数**。Stable Diffusion、SD3、FLUX、Sora、Imagen 全都是它的变种。""")

# ============================================================
# 算法一页纸
# ============================================================
md(r"""## Forward Diffusion 算法一页纸

**前向过程**（不需要学习，纯数学）：给一张真实图 $x_0$，一步步往里加高斯噪声，做 $T$ 步（通常 1000）。

$$
x_0 \xrightarrow{\text{加噪}} x_1 \xrightarrow{\text{加噪}} x_2 \xrightarrow{\text{加噪}} \cdots \xrightarrow{\text{加噪}} x_T \approx \mathcal{N}(0, I)
$$

**单步加噪公式**（马尔可夫链）：

$$
q(x_t | x_{t-1}) = \mathcal{N}\bigl(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\bigr)
$$

**关键设计**（每条都要能答出 why）：

| 设计 | 做法 | 为什么 |
|------|------|--------|
| 均值缩放 $\sqrt{1-\beta_t}$ | 不是 1，也不是 0 | 保证 $T\to\infty$ 时 $x_T$ 收敛到标准高斯（方差守恒） |
| 方差 $\beta_t$ | 每步加一点点噪声 | 单步扰动要小，反向才好学 |
| $\beta_t$ schedule | linear / cosine / sigmoid | 开头慢加（保细节），末尾快加（到纯噪声） |
| 步数 $T$ | 通常 1000 | 太少反向学不会；太多浪费 |

**核心 trick - 重参数化一步到位**（Step 3 会详细推导）：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

其中 $\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$。

**这个公式是 Diffusion 的命根子**：给定 $x_0$ 和任意 $t$，一步就能算出 $x_t$，不用真的递推 1000 次。训练时随机采样 $t$ 做 mini-batch 才可行。

---

下面按 Step 0 → Step 7 依次手撕。""")

# ============================================================
# Imports & setup
# ============================================================
code("""import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

torch.manual_seed(42)
np.random.seed(42)

print('torch:', torch.__version__)
print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')""")

# ============================================================
# Step 0
# ============================================================
md(r"""## Step 0: 为什么不能一步到位？—— 两本账

先不谈 Diffusion，先算清楚**直接学生成的代价**，才能理解 Diffusion 为什么要绕 1000 步这么迂回的路。

### 账本 1：高维数据分布太复杂

一张 256×256 RGB 图在 $\mathbb{R}^{196608}$ 空间里。真实图像分布 $p(x)$ 只是这个高维空间里一个**极薄的流形**。

- 想直接学 $p(x)$ ：模型要在 196608 维空间里精确勾勒这个流形 → 参数爆炸、训练崩
- GAN 用对抗博弈绕开了显式 $p(x)$，但付出了**训练不稳**的代价
- VAE 用 latent bottleneck，但付出了**图糊**的代价

### 账本 2：Diffusion 的妙招 —— 分而治之

**把"从噪声到图"这一个难题，拆成 1000 个"去一点噪声"的简单回归问题**。

| 方案 | 每步难度 | 1000 步后效果 |
|------|----------|---------------|
| 直接 $z \to x$ | 难如登天 | 模型学不会 |
| $x_t \to x_{t-1}$ | 小扰动、梯度平滑 | 叠起来也是"噪声 → 图" |

打个比方：
- 直接让你从 1000 米外一枪打中苹果 → 几乎不可能
- 让你每次往前走 1 米，每次只需要微调枪口 → 每步都简单，总和等价

下面用代码算一下这笔账。""")

code("""# 账本 1 demo：真实图像分布在高维空间里多稀疏？
# 我们造一个简单的"类图像流形"：16x16 的手写数字风格合成图

def make_synthetic_digit(shape='cross'):
    '''合成一个 16x16 的假'数字'图，数值在 [-1, 1]'''
    img = -torch.ones(16, 16)  # 背景黑
    if shape == 'cross':
        img[7:9, 2:14] = 1.0   # 横
        img[2:14, 7:9] = 1.0   # 竖
    elif shape == 'circle':
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
        r = torch.sqrt((yy - 7.5)**2 + (xx - 7.5)**2)
        img = torch.where((r > 4) & (r < 6), torch.tensor(1.0), torch.tensor(-1.0))
    elif shape == 'checker':
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing='ij')
        img = torch.where((yy // 4 + xx // 4) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
    return img

x0 = make_synthetic_digit('cross')

# 看看这张"真实图"在 256 维空间里的样子
print(f'真实图维度: {x0.numel()} 维')
print(f'像素均值: {x0.mean():.3f}, 标准差: {x0.std():.3f}')
print(f'像素范围: [{x0.min():.2f}, {x0.max():.2f}]')

# 对比：同维度的纯高斯噪声
noise = torch.randn_like(x0)
print(f'\\n标准高斯噪声的均值: {noise.mean():.3f}, 标准差: {noise.std():.3f}')

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(x0, cmap='gray', vmin=-1, vmax=1); axes[0].set_title('x_0 (真实图)')
axes[1].imshow(noise, cmap='gray', vmin=-2, vmax=2); axes[1].set_title('x_T (标准高斯)')
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()""")

md("""**关键洞察**：从右图的纯噪声，一步跳到左图的结构，**模型需要在 256 维空间里精确定位流形**。太难了。

Diffusion 的做法 → 把这个跳跃**拆成 1000 个连续的小任务**，每一步只从"稍微糊一点的图"去一点点噪声。每一步都是一个简单回归题。""")

# ============================================================
# Step 1
# ============================================================
md(r"""## Step 1: 马尔可夫链 —— 单步加噪公式

先把"加一步噪声"这件事写成数学：

$$
q(x_t | x_{t-1}) = \mathcal{N}\bigl(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\bigr)
$$

用**重参数化**写成可采样形式：

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0, I)
$$

**为什么是 $\sqrt{1-\beta_t}$ 而不是 $1$？**

我们希望整个过程**方差守恒**（variance-preserving, VP）。假设 $\text{Var}(x_{t-1}) = 1$：

$$
\text{Var}(x_t) = (1-\beta_t) \cdot \text{Var}(x_{t-1}) + \beta_t \cdot \text{Var}(\epsilon_t) = (1-\beta_t) + \beta_t = 1 \;\checkmark
$$

如果用 $1$ 作为系数，方差会**爆炸**到无穷（variance-exploding, VE，另一条技术路线）。""")

code("""# 验证单步加噪的方差守恒
beta = 0.01  # 一个小的 β
x_prev = torch.randn(10000)  # 假设 x_{t-1} ~ N(0, 1)
eps = torch.randn(10000)

x_next = math.sqrt(1 - beta) * x_prev + math.sqrt(beta) * eps

print(f'x_{{t-1}} 方差: {x_prev.var().item():.4f}')
print(f'x_t     方差: {x_next.var().item():.4f}  ← 应该 ≈ 1（方差守恒）')
print(f'\\n如果用系数 1（VE 变体）：')
x_ve = 1.0 * x_prev + math.sqrt(beta) * eps
print(f'x_t     方差: {x_ve.var().item():.4f}  ← 略大于 1，累积 1000 步会爆炸')""")

# ============================================================
# Step 2
# ============================================================
md(r"""## Step 2: 递推 1000 步（朴素实现）

按 Step 1 的公式，硬递推 1000 次：

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t
$$""")

code("""T = 1000
betas = torch.linspace(1e-4, 0.02, T)   # linear schedule（DDPM 原版）

x = make_synthetic_digit('cross').flatten()  # 从 x_0 出发
trajectory = [x.clone()]

for t in range(T):
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - betas[t]) * x + torch.sqrt(betas[t]) * eps
    trajectory.append(x.clone())

# 抽 8 个时间点看
checkpoints = [0, 50, 100, 200, 400, 700, 900, 1000]
fig, axes = plt.subplots(1, 8, figsize=(16, 2.2))
for ax, t in zip(axes, checkpoints):
    img = trajectory[t].reshape(16, 16)
    ax.imshow(img, cmap='gray', vmin=-2, vmax=2)
    ax.set_title(f't={t}'); ax.axis('off')
plt.suptitle('朴素递推 1000 步：十字 → 噪声', y=1.05)
plt.tight_layout(); plt.show()

print(f'x_0   方差: {trajectory[0].var().item():.4f}')
print(f'x_500 方差: {trajectory[500].var().item():.4f}')
print(f'x_T   方差: {trajectory[T].var().item():.4f}  ← ≈ 1 ✓')""")

md("""**观察**：到 t=1000 时，图像已经完全变成了不可识别的噪声，方差稳定在 1。**前向过程本身完全不需要神经网络，纯数学公式。**

**但这种朴素实现有问题**：训练时我们要随机采样 t，如果每次都要递推 t 次（最坏 1000 次），一个 epoch 要跑烂。

下一步，我们推导**一步到位**的公式。""")

# ============================================================
# Step 3
# ============================================================
md(r"""## Step 3: 重参数化魔法 —— 一步到位

**核心问题**：给定 $x_0$ 和任意 $t$，能不能**一个公式直接算出** $x_t$，不用递推？

**答案可以，推导如下**。

记 $\alpha_t = 1 - \beta_t$。从 Step 1：

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_t
$$

代入 $x_{t-1} = \sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_{t-1}}\, \epsilon_{t-1}$：

$$
x_t = \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\, \epsilon_{t-1} + \sqrt{1-\alpha_t}\, \epsilon_t
$$

后两项都是独立高斯，合并（**两个独立高斯相加，方差相加**）：

$$
\sqrt{\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)}\, \epsilon = \sqrt{1 - \alpha_t \alpha_{t-1}}\, \epsilon
$$

所以：

$$
x_t = \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\, \epsilon
$$

归纳到 $x_0$：

$$
\boxed{x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon},\quad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s
$$

**这就是 Diffusion 训练能跑通的关键**。

| 版本 | 复杂度 | batch 能否并行 |
|------|--------|----------------|
| 朴素递推 | O(T) per sample | ❌ 每个样本要 1000 步 |
| 重参数化 | O(1) per sample | ✅ 一个矩阵乘完事 |""")

code("""# 实现 & 验证：重参数化 vs 朴素递推
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)   # ᾱ_t = ∏ α_s

def q_sample(x0, t, eps):
    '''一步到位：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε'''
    ab = alpha_bars[t]
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps

# 验证：固定 ε，重参数化结果应该和递推结果方差一致（不可能完全相等，因为递推每步 ε 独立）
# 所以我们用"跑多次求统计量"来验
x0 = make_synthetic_digit('cross').flatten()
t_check = 500

# 方法 A：重参数化，跑 1000 次看统计
xs_A = torch.stack([q_sample(x0, t_check, torch.randn_like(x0)) for _ in range(1000)])

# 方法 B：朴素递推，跑 1000 次看统计
xs_B = []
for _ in range(1000):
    x = x0.clone()
    for tt in range(t_check + 1):
        x = torch.sqrt(alphas[tt]) * x + torch.sqrt(betas[tt]) * torch.randn_like(x)
    xs_B.append(x)
xs_B = torch.stack(xs_B)

print(f'在 t={t_check} 处，1000 次采样的统计量：')
print(f'重参数化: 均值 {xs_A.mean():.4f}, 方差 {xs_A.var():.4f}')
print(f'朴素递推: 均值 {xs_B.mean():.4f}, 方差 {xs_B.var():.4f}')
print(f'\\n两者分布一致 ✓（均值应接近 √ᾱ_t·x0 的均值，方差应接近 1）')

print(f'\\n√ᾱ_{t_check}   = {torch.sqrt(alpha_bars[t_check]):.4f}   ← 原图被压缩的比例')
print(f'√(1-ᾱ_{t_check}) = {torch.sqrt(1-alpha_bars[t_check]):.4f} ← 噪声混入的比例')""")

md("""**关键理解**：`√ᾱ_t` 和 `√(1-ᾱ_t)` 是一对**权重**，t 小时前者大（图清晰）、t 大时后者大（噪声主导）。

这一对权重决定了 **信噪比 (SNR)**，这是 Step 6 要讲的。""")

# ============================================================
# Step 4
# ============================================================
md(r"""## Step 4: β schedule 实现 —— linear vs cosine

**β schedule**（加噪时间表）决定了每一步加多少噪声。不同 schedule 差异巨大。

### Schedule A：Linear (DDPM 原版 Ho 2020)

$$
\beta_t = \beta_{\min} + \frac{t}{T} (\beta_{\max} - \beta_{\min}), \quad \beta_{\min}=10^{-4},\ \beta_{\max}=0.02
$$

**问题**：对小分辨率 OK，但高分辨率时**末期破坏过快** —— 噪声在 t=500 就基本淹没图像，后 500 步"浪费"了。

### Schedule B：Cosine (Nichol & Dhariwal 2021, improved DDPM)

直接设计 $\bar\alpha_t$ 按余弦下降：

$$
\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)
$$

其中 $s = 0.008$ 是个小 offset 防 $t=0$ 处 $\beta$ 过大。

**好处**：中段加噪节奏均匀，图像信息**循序渐进地被噪声覆盖**，训练收敛更稳。""")

code("""def make_linear_schedule(T=1000, beta_min=1e-4, beta_max=0.02):
    betas = torch.linspace(beta_min, beta_max, T)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, 0)
    return betas, alphas, alpha_bars

def make_cosine_schedule(T=1000, s=0.008):
    '''Improved DDPM cosine schedule (Nichol & Dhariwal 2021)'''
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alpha_bars = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    # 从 ᾱ 反推 β
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = torch.clamp(betas, 0, 0.999)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, 0)
    return betas, alphas, alpha_bars

b_lin, a_lin, ab_lin = make_linear_schedule()
b_cos, a_cos, ab_cos = make_cosine_schedule()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(b_lin, label='linear', color='C0')
axes[0].plot(b_cos, label='cosine', color='C1')
axes[0].set_title(r'$\\beta_t$ schedule'); axes[0].set_xlabel('t'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ab_lin, label='linear', color='C0')
axes[1].plot(ab_cos, label='cosine', color='C1')
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.4, label='SNR 信号半衰点')
axes[1].set_title(r'$\\bar\\alpha_t$ (剩余信号比例)'); axes[1].set_xlabel('t'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

# 找 ᾱ_t = 0.5 的位置（信号刚好剩一半，噪声一半）
half_lin = (ab_lin < 0.5).nonzero()[0].item()
half_cos = (ab_cos < 0.5).nonzero()[0].item()
print(f'\\n信号半衰点 (ᾱ_t ≈ 0.5)：')
print(f'  linear 在 t={half_lin:>4}   ← 只走了 {half_lin/T:.0%}')
print(f'  cosine 在 t={half_cos:>4}   ← 走了 {half_cos/T:.0%}')
print('\\n→ cosine 把"信号衰减一半"这件事推迟到中段，给模型更多缓冲步骤。')""")

# ============================================================
# Step 5
# ============================================================
md("""## Step 5: 真实图像加噪可视化

把 Step 3 的重参数化公式用到三张合成图上，看 linear / cosine 的差异。""")

code("""# 准备 3 张合成图（当作小 mini-batch）
x0_batch = torch.stack([
    make_synthetic_digit('cross').flatten(),
    make_synthetic_digit('circle').flatten(),
    make_synthetic_digit('checker').flatten(),
])  # [3, 256]

def q_sample_ab(x0, t_idx, alpha_bars):
    ab = alpha_bars[t_idx]
    eps = torch.randn_like(x0)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps, eps

checkpoints = [0, 100, 300, 500, 700, 900, 999]

fig, axes = plt.subplots(6, len(checkpoints), figsize=(14, 10))
for row, (sched_name, ab) in enumerate([('linear', ab_lin), ('cosine', ab_cos)]):
    for col, t in enumerate(checkpoints):
        for img_idx, (img_row_base) in enumerate([0, 1, 2]):
            x_t, _ = q_sample_ab(x0_batch[img_idx], t, ab)
            axes[row*3 + img_idx, col].imshow(x_t.reshape(16, 16), cmap='gray', vmin=-2, vmax=2)
            axes[row*3 + img_idx, col].axis('off')
            if col == 0:
                axes[row*3 + img_idx, col].set_ylabel(f'{sched_name}\\n{["cross","circle","checker"][img_idx]}', rotation=0, labelpad=30, va='center')
        if row == 0:
            axes[img_idx, col].set_title(f't={t}')
    for col, t in enumerate(checkpoints):
        axes[row*3, col].set_title(f'{sched_name} t={t}')

plt.tight_layout(); plt.show()""")

md("""**对比观察**：
- **linear**：在 t=500 时结构已经几乎被噪声淹没（图形边缘模糊难辨）
- **cosine**：在 t=500 时原始结构仍清晰可见，要到 t=800+ 才彻底变噪声

**为什么这对训练重要？**

模型在每个 t 都要学"从 $x_t$ 推 $x_0$ 的噪声"。如果 linear schedule 让 t=500~999 这 500 步都是"纯噪声"，模型在这段几乎学不到信息 → **50% 的训练步浪费了**。cosine 把信息衰减拉平，让每个 t 都带一些有效信号。""")

# ============================================================
# Step 6
# ============================================================
md(r"""## Step 6: 从 SNR 角度理解 ᾱ_t

**信噪比 (Signal-to-Noise Ratio)** 的定义：

$$
\text{SNR}(t) = \frac{\text{信号方差}}{\text{噪声方差}} = \frac{\bar\alpha_t}{1-\bar\alpha_t}
$$

- $t=0$：SNR → ∞（全是信号）
- $t=T$：SNR → 0（全是噪声）
- $\bar\alpha_t = 0.5$：SNR = 1（信号和噪声各半）

**面试高频**：现代 Diffusion 论文都用 `log(SNR)` 或 `γ = log(SNR)` 作为时间的替代参数，因为它在 log 空间线性下降更均匀。Flow Matching 用的 `t ∈ [0,1]` 也可以通过 schedule 映射到 log-SNR。""")

code("""log_snr_lin = torch.log(ab_lin / (1 - ab_lin + 1e-8))
log_snr_cos = torch.log(ab_cos / (1 - ab_cos + 1e-8))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(log_snr_lin, label='linear', color='C0')
ax.plot(log_snr_cos, label='cosine', color='C1')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='SNR = 1')
ax.set_xlabel('t'); ax.set_ylabel('log SNR'); ax.set_title('log SNR 随 t 的变化')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print('关键点：')
print(f'  linear 在 t={T-1}: log SNR = {log_snr_lin[-1]:.2f}（SNR = {log_snr_lin[-1].exp():.4f}，几乎纯噪声）')
print(f'  cosine 在 t={T-1}: log SNR = {log_snr_cos[-1]:.2f}（SNR = {log_snr_cos[-1].exp():.4f}）')
print(f'\\n  log SNR 的斜率越陡 → 信息衰减越快 → 模型在那一段越难学')""")

# ============================================================
# Step 7
# ============================================================
md("""## Step 7: 训练时的前向代码（生产版）

把 Step 3 的 `q_sample` 写成**训练时真正在用的样子** —— batch 维度、时间索引 gather、GPU 友好。这就是 DDPM 训练循环里**最里层那几行**。""")

code("""class ForwardDiffusion:
    '''前向扩散：训练时用的 q_sample 封装'''
    def __init__(self, T=1000, schedule='cosine', device='cpu'):
        self.T = T
        if schedule == 'linear':
            betas, alphas, alpha_bars = make_linear_schedule(T)
        elif schedule == 'cosine':
            betas, alphas, alpha_bars = make_cosine_schedule(T)
        else:
            raise ValueError(schedule)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alpha_bars = alpha_bars.to(device)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars).to(device)

    def q_sample(self, x0, t, noise=None):
        '''
        x0:    [B, ...]  原始图
        t:     [B]       每个样本采样到的时间步 (整数)
        noise: [B, ...]  可选，显式给定噪声（训练时一般随机采）
        返回:  (x_t, noise) —— x_t 和实际用到的噪声
        '''
        if noise is None:
            noise = torch.randn_like(x0)

        # 按 batch 的 t gather 出每个样本对应的系数
        sab = self.sqrt_alpha_bars[t]               # [B]
        somab = self.sqrt_one_minus_alpha_bars[t]   # [B]
        # 对齐 x0 的维度
        while sab.dim() < x0.dim():
            sab = sab.unsqueeze(-1)
            somab = somab.unsqueeze(-1)

        x_t = sab * x0 + somab * noise
        return x_t, noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.T, (batch_size,))


# 演示：一个真实训练步骤里发生了什么
fd = ForwardDiffusion(T=1000, schedule='cosine')

batch_size = 4
x0 = torch.stack([make_synthetic_digit(s).flatten() for s in ['cross','circle','checker','cross']])
t = fd.sample_timesteps(batch_size)     # 每个样本采一个随机 t
x_t, noise = fd.q_sample(x0, t)

print(f'x0 shape:    {list(x0.shape)}')
print(f't:           {t.tolist()}  ← 每个样本一个随机时间步')
print(f'x_t shape:   {list(x_t.shape)}')
print(f'noise shape: {list(noise.shape)}')
print('\\n这就是 DDPM 训练 loop 的最里层 —— 下一个 notebook 的反向过程会学：')
print('  给定 x_t 和 t，预测当时加的 noise 是什么。')""")

# ============================================================
# 收束
# ============================================================
md(r"""## 一页纸收束

### 本章重点

| 概念 | 公式 | 一句话 |
|------|------|--------|
| 单步加噪 | $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\epsilon$ | 每步加一点高斯噪声，方差守恒 |
| 一步到位 | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | **训练能跑的关键** |
| β schedule | linear / cosine | cosine 信息衰减更均匀，训练更稳 |
| SNR | $\bar\alpha_t / (1-\bar\alpha_t)$ | 信噪比，现代论文用 log SNR 代替 t |

### 前向 vs 反向 分工

| 阶段 | 学不学 | 复杂度 |
|------|--------|--------|
| 前向加噪 | **不学**，纯数学 | O(1) per sample |
| 反向去噪 | **要学**，靠神经网络（02 讲） | 1000 次 forward pass |

### 和后续 notebook 的关系

- **02 DDPM 反向 + 训练** —— 学一个网络 $\epsilon_\theta(x_t, t)$ 来预测 Step 7 里的 noise
- **06 Flow Matching** —— 不再用 $\sqrt{\bar\alpha}$ 这种间接 schedule，直接学直线路径 $x_t = (1-t)x_0 + t \epsilon$
- **09 DiT** —— 把 U-Net 换成 Transformer，让前向/反向架构 scaling 到 10B+""")

# ============================================================
# 面试题
# ============================================================
md("""## 面试题 7 连问

---

### Q1: 描述 Diffusion 前向过程的加噪公式，并解释每个符号的含义。

<details><summary>参考答案</summary>

单步：$q(x_t|x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t}\\, x_{t-1}, \\beta_t I)$，即 $x_t = \\sqrt{1-\\beta_t}\\, x_{t-1} + \\sqrt{\\beta_t}\\, \\epsilon$，$\\epsilon \\sim \\mathcal{N}(0,I)$。

- $\\beta_t$：第 $t$ 步的噪声强度（schedule 定义）
- $\\alpha_t = 1-\\beta_t$：第 $t$ 步保留的信号比例
- $\\bar\\alpha_t = \\prod_{s=1}^t \\alpha_s$：到 $t$ 步累计保留的信号
- 一步到位公式：$x_t = \\sqrt{\\bar\\alpha_t}\\, x_0 + \\sqrt{1-\\bar\\alpha_t}\\, \\epsilon$
</details>

---

### Q2: 为什么单步的系数是 √(1-βₜ) 而不是 1？这么设计的意义是什么？

<details><summary>参考答案</summary>

为了**方差守恒 (variance-preserving)**。若 Var(x_{t-1}) = 1，则 Var(x_t) = (1-β_t)·1 + β_t·1 = 1，不会爆炸。

对比：如果系数用 1，就是 **variance-exploding (VE)** 路线（NCSN、Song 的 SBM），方差会累积到无穷大，需要不同的反向采样公式。

DDPM 用 VP，NCSN 用 VE，EDM (Karras 2022) 提出了统一框架。
</details>

---

### Q3: 一步到位公式 $x_t = \\sqrt{\\bar\\alpha_t}x_0 + \\sqrt{1-\\bar\\alpha_t}\\epsilon$ 怎么推导的？

<details><summary>参考答案</summary>

递推代入 + 两个独立高斯相加（方差相加）。

$x_t = \\sqrt{\\alpha_t}\\, x_{t-1} + \\sqrt{1-\\alpha_t}\\, \\epsilon_t$，再代 $x_{t-1}$：

$x_t = \\sqrt{\\alpha_t \\alpha_{t-1}}\\, x_{t-2} + \\sqrt{\\alpha_t(1-\\alpha_{t-1})}\\, \\epsilon_{t-1} + \\sqrt{1-\\alpha_t}\\, \\epsilon_t$

后两项独立高斯合并：方差相加得 $\\alpha_t(1-\\alpha_{t-1}) + (1-\\alpha_t) = 1 - \\alpha_t \\alpha_{t-1}$。

归纳 → $x_t = \\sqrt{\\bar\\alpha_t}\\, x_0 + \\sqrt{1-\\bar\\alpha_t}\\, \\epsilon$。

**为什么重要**：训练时随机采样 t 做 mini-batch，必须 O(1) 采样 $x_t$，不能递推 1000 次。
</details>

---

### Q4: linear vs cosine schedule 有什么区别？为什么 cosine 更好？

<details><summary>参考答案</summary>

- **linear**：β_t 从 1e-4 线性涨到 0.02。问题：**信号半衰点在 t ≈ 200**，后 800 步几乎全是纯噪声，训练浪费。
- **cosine**：直接让 $\\bar\\alpha_t$ 按 $\\cos^2$ 下降。信号半衰点推到 t ≈ 500，信息衰减更均匀。

**核心好处**：cosine 让每个 t 都带一些有效信号 → 训练信号充分 → 收敛更稳、最终质量更高。Improved DDPM (Nichol 2021) 实证 cosine 在 ImageNet 256×256 有显著提升。
</details>

---

### Q5: 为什么 T 通常取 1000？200 行不行？10000 呢？

<details><summary>参考答案</summary>

- **T 太小（如 200）**：相邻 $x_t$ 差异大 → 反向每步跨度大 → 模型难学 → 质量差
- **T 太大（如 10000）**：相邻 $x_t$ 几乎一样 → 信息浪费 → 训练/推理都慢
- **1000 是经验平衡点**，DDPM 首创，后续大部分工作沿用

推理阶段可用 DDIM 跳步到 50 步甚至更少（03 讲），但训练通常保持 T=1000 以获得精细的 schedule。

Flow Matching 和 Consistency Models 则从根本上换了范式，不再受 T 约束。
</details>

---

### Q6: SNR 是什么？log-SNR 在现代论文里为什么重要？

<details><summary>参考答案</summary>

$\\text{SNR}(t) = \\bar\\alpha_t / (1-\\bar\\alpha_t)$，信号方差 / 噪声方差。

- log-SNR 在 log 空间线性分布更均匀 → 适合作为时间的**统一参数化**
- EDM (Karras 2022)、SD3、FLUX 都用 $\\sigma$ 或 log-SNR 作为时间变量，而非离散 t
- 好处：不同 schedule 可以在 log-SNR 空间对齐比较
</details>

---

### Q7: 前向过程需要训练吗？和反向过程的关系是什么？

<details><summary>参考答案</summary>

**不需要训练**。前向就是 $x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1-\\bar\\alpha_t} \\epsilon$，纯数学公式，schedule 一旦定好就不动。

**反向过程才是要学的**：训练一个网络 $\\epsilon_\\theta(x_t, t)$，输入带噪 $x_t$ 和时间 $t$，输出当时加的噪声 $\\epsilon$。

训练目标（DDPM）：$\\mathcal{L} = \\mathbb{E}_{x_0, t, \\epsilon} \\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2$

推理时从 $x_T \\sim \\mathcal{N}(0, I)$ 出发，反向走 T 步去噪得到 $x_0$。这就是 02 的内容。
</details>

---

### 🎯 加分题：如果面试官问"Diffusion 和 VAE 能不能结合？"

Stable Diffusion 就是这么做的：**VAE encoder 把图压到 latent → 在 latent 上跑 Diffusion → VAE decoder 解码回像素**。这叫 **Latent Diffusion**，计算量降 10x，是现代大图生成的标配。

这就是为什么 SD / SDXL / SD3 / FLUX 都能在消费级 GPU 上跑。""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"}
}

with open('/home/yg/yg/code/learning_everything/diffusion_scratch/01_forward_diffusion.ipynb', 'w') as f:
    nbf.write(nb, f)

print('done, cells:', len(cells))
