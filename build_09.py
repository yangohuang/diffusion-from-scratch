"""09 DiT 架构 - Sora/SD3/FLUX 底座。重点展开，最重要的一篇。"""
import json

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})

md("""# 09 - DiT 架构（Sora / SD3 / FLUX 的共同底座）⭐️

> **一句话直觉**：把 Diffusion 里用的 U-Net **换成 Transformer**。不是换个皮肤，是整个范式从"图像先验内嵌"变成"纯数据驱动 + scaling"。

**为什么这一篇最重要（面试 Top 1 考点）**：
- **Sora / SD3 / FLUX.1 / PixArt / Hunyuan-DiT / Movie Gen** 全是 DiT（注意 Stable Video Diffusion 是 U-Net，别搞混）
- 任何招生成方向的公司面试，**DiT 原理必考**，adaLN-Zero 是高频细节考点
- 懂了 DiT 就懂了 **2024 之后所有生成模型的架构选型逻辑**

**核心断言（Peebles & Xie, 2023 "Scalable Diffusion Models with Transformers"）**：
> Diffusion 也能像 LLM 一样吃 scaling law。U-Net 的归纳偏置是瓶颈，Transformer 解锁规模化。

> 📎 本课是**浓缩版**：patchify / adaLN-Zero / DiT Block / MM-DiT 的逐层手撕版见同仓库 `../dit_scratch/` 01-05/08。""")

md("""## Part 0 — U-Net 有什么问题？

先看 U-Net（DDPM/SD1/SD2 都用它）：

```
Input 图像
   ↓ Conv Down
   ↓ Conv Down
   ↓ Conv Down    ← 中间 bottleneck + attention
   ↑ Conv Up + skip
   ↑ Conv Up + skip
   ↑ Conv Up + skip
Output 噪声预测
```

**优点**：
- 卷积 + 多尺度，图像任务天然契合
- Skip connection 保证细节恢复

**硬伤**（2022 年开始暴露）：
1. **不 scaling 友好**：继续加大 U-Net 参数量，改善明显递减（社区经验共识，DiT 论文的动机之一）
2. **注意力塞得奇怪**：只在中低分辨率层（如 32/16/8）加 attention（为了省算力），空间全局建模弱
3. **文本条件注入手段有限**：只能通过 cross-attention 在几个特定层，文本和图像交互不充分
4. **架构手工调参重**：哪几层加 attention、下采样几次、通道数怎么分配——超参工程很重

**对比 GPT 路线**：LLM 架构就一种（Decoder-only Transformer），scaling law 完美。**U-Net 在图像生成上是 pre-scaling era 的产物**。""")

md("""## Part 1 — DiT 的核心改动（面试必会）

### 改动 1：Patchify（把图变 token）
- 图像 $H \\times W \\times C$ 切成 $P \\times P$ 的 patch（典型 $P=2$）
- 每个 patch 线性投影成一个 token
- 得到序列长度 $HW/P^2$，用 Transformer 处理

（**这一步和 ViT 完全一样**。ViT 是"理解"，DiT 是"生成"，但 tokenize 方法相同。）

### 改动 2：把 U-Net 全部替换为 DiT Block
N 个 DiT Block 堆叠。每个 Block 就是标准 Transformer 的三件套：
- Multi-Head Self-Attention
- FFN（MLP 2 层）
- 残差连接

### 改动 3：条件怎么注入？（核心创新点）
Diffusion 需要两种条件：
- **时间步 $t$**：告诉模型"现在去噪到第几步"
- **标签 / 文本 $c$**：告诉模型"要生成什么"

Peebles & Xie 论文比较了 **4 种注入方式**：

| 方案 | 怎么做 | 效果 |
|------|--------|------|
| **In-context** | 把 $t, c$ 当作额外 token 拼到序列前面 | 差 |
| **Cross-Attention** | 另开 cross-attn 层让 token 关注 $t, c$ | 中 |
| **adaLN** | 把 $t, c$ 输入一个 MLP，输出 LayerNorm 的 scale/shift | 好 |
| **adaLN-Zero** | adaLN 最后一层 **零初始化** | **最好** |

**结论**：**adaLN-Zero 完胜**，FID 最低、训练最稳。SD3/FLUX/Sora 全部用这个。""")

md("""## Part 2 — adaLN-Zero：面试 Top 1 细节考点

下面三个公式是**同一个公式改两笔**：标准 LN → adaLN 只动 $\\gamma, \\beta$ 的**来源**（可学习参数 → 条件算出来）；adaLN → adaLN-Zero 只**多一个 $\\alpha$ 门 + 零初始化**。抓住"每步只改一处"，就不会背混。

### 标准 LayerNorm
$$
y = \\gamma \\cdot \\frac{x - \\mu}{\\sigma} + \\beta
$$
其中 $\\gamma, \\beta$ 是可学习参数，**每个通道一个**，和输入无关。

### adaLN（Adaptive LayerNorm）
$\\gamma, \\beta$ 不再是可学习参数，而是**条件向量算出来的**：
$$
\\gamma, \\beta = \\text{MLP}(t, c)
$$
每次前向 $t, c$ 不同 → $\\gamma, \\beta$ 不同 → LayerNorm 行为被条件调制。

### adaLN-Zero（核心 trick）
更进一步，DiT Block 里还有个 **gate** $\\alpha$：
$$
\\text{output} = x + \\alpha \\cdot \\text{Block}(\\text{adaLN}(x))
$$

**初始化时让 $\\alpha = 0$**（通过让生成 $\\alpha$ 的 MLP 末层零初始化）。

**为什么要零初始化？**
- 初始状态下 $\\alpha=0$ → 每个 Block 输出 = 输入（恒等映射）——**$\\alpha=0$ 代入公式，output = $x$，一眼看出为什么稳**
- 整个 DiT 初始化后**完全不变换输入**
- 训练一开始等价于一个"什么都不做"的恒等网络
- 参数逐渐学起来，**稳定性极好**，不容易一开始就炸梯度

这个 trick 源自 ResNet 的 zero-$\\gamma$ 初始化（2018）/ Fixup（2019）/ ReZero（2020）一脉；DiT 是在 Diffusion 上系统化使用它的代表。

### 一个 DiT Block 完整结构
```
条件 (t, c)
  ↓ MLP → [γ₁, β₁, α₁, γ₂, β₂, α₂]  (6 个调制参数, α₁α₂ 零初始化)

x
  ↓ LayerNorm → scale(γ₁) → shift(β₁)
  ↓ Self-Attention
  ↓ × α₁                              ← 零初始化门
  ↓ Residual Add
  ↓ LayerNorm → scale(γ₂) → shift(β₂)
  ↓ FFN (MLP)
  ↓ × α₂                              ← 零初始化门
  ↓ Residual Add
  output
```
**一个 DiT Block 输出 6 个调制参数**（$\\gamma_1, \\beta_1, \\alpha_1, \\gamma_2, \\beta_2, \\alpha_2$）。这个"6"是面试官常挖的细节。""")

md("""## Part 3 — 从 DiT 到 MM-DiT（SD3 的升级）

SD3 用的不是原版 DiT，是 **MM-DiT (Multimodal DiT)**。改动：

| 原 DiT | MM-DiT (SD3) |
|--------|-------------|
| 图像 token 做 self-attention；文本只通过 adaLN 调制 | **图像 token + 文本 token 拼起来做联合 self-attention** |
| 文本是"全局条件" | 文本是**序列级**参与者，可以做 token-level 交互 |

**为什么要改**：原 DiT 文本信息从 $c$ 全局广播进来，粒度太粗。复杂文本描述（"戴红帽子的柯基站在蓝色桌子上"）时文本 token 和图像 token 需要精细对应。

**FLUX.1 更进一步**：在 MM-DiT 基础上加了 **Single Stream Block**（文本 + 图像彻底融合走完整个 block）。

### 架构演进时间线
```
ViT (2020, 视觉理解)
  ↓
DiT (2022, 生成, adaLN-Zero)
  ↓
PixArt-α (2023, 第一个文生图 DiT)
  ↓
SD3 (2024, MM-DiT)
  ↓
FLUX.1 (2024, MM-DiT + Single Stream)
  ↓
Sora (2024, DiT + 时空 patch)
  ↓
Movie Gen / Hunyuan-DiT (2024-2025)
```
一路走下来，核心就两件事：**adaLN-Zero 是发动机，patch tokenize 是轮子**。""")

md("""## Part 4 — DiT 为什么 scaling 友好？

DiT 论文的核心贡献不是"用了 Transformer"，而是**实证 Diffusion 也有 scaling law**：

- 参数量从 33M → 675M → 训练 FLOPs × 10 → FID 单调下降
- 架构变"胖"（hidden dim ↑）或"深"（layer ↑）都有效
- 和 LLM 的 Chinchilla 定律逻辑相同

**业务含义**：
- **U-Net 时代**：做图像生成的公司比拼数据质量 + 花式调 trick
- **DiT 时代**：比拼算力规模 + scaling 工程能力

这直接解释了为什么 Sora、Movie Gen 这种大厂级生成模型能出来——他们有 LLM 规模的训练基础设施。

**对你们业务（数字人）的启示**：
- 信创 910B 推理友好：DiT 是标准 Transformer 算子，Ascend 支持好，不像 U-Net 的某些算子要单独适配
- 小模型起步路线：用 PixArt-α 这种 600M 参数的 DiT 做 baseline 够了
- 和 LLM 复用基础设施：模型并行、Flash Attention、量化这些 LLM 经验全部迁移""")

md("""## Part 5 — 代码：最小可跑 DiT Block（必背）

不跑训练（要 GPU 几小时），只撕 **DiT Block 的核心代码** — 这是面试手撕题常考。

**逐行对应 Part 2 的 ASCII 图**：`adaLN_modulation` = 图顶部的条件 MLP；`chunk(6)` = 六个调制参数 $[\\gamma_1, \\beta_1, \\alpha_1, \\gamma_2, \\beta_2, \\alpha_2]$；两处 `g * h` = 两个零初始化门。""")

code("""import torch
import torch.nn as nn
import torch.nn.functional as F

class DiTBlock(nn.Module):
    '''DiT Block: adaLN-Zero 调制的 Transformer Block'''
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        # 核心：条件向量 -> 6 个调制参数 (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # Zero-init 关键！
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        '''x: [B, N, D] tokens;  c: [B, D] condition (t 或 t+label)'''
        # 6 个调制参数一次算出来
        s1, b1, g1, s2, b2, g2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        # [B, D] -> [B, 1, D] 方便广播到 [B, N, D]
        s1, b1, g1 = s1.unsqueeze(1), b1.unsqueeze(1), g1.unsqueeze(1)
        s2, b2, g2 = s2.unsqueeze(1), b2.unsqueeze(1), g2.unsqueeze(1)

        # ----- Attention 分支 -----
        h = self.norm1(x)
        h = h * (1 + s1) + b1  # adaLN: scale + shift
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1 * h  # adaLN-Zero: gate 起初为 0，等价恒等映射

        # ----- MLP 分支 -----
        h = self.norm2(x)
        h = h * (1 + s2) + b2
        h = self.mlp(h)
        x = x + g2 * h
        return x

# Smoke test
block = DiTBlock(dim=256, n_heads=8)
x = torch.randn(2, 196, 256)  # batch=2, 14*14 patches, dim=256
c = torch.randn(2, 256)  # condition vector
out = block(x, c)
print(f'input: {x.shape}, output: {out.shape}')
print(f'初始化后 output 是否等于 input (因为 zero-init):')
print(f'  diff norm = {(out - x).norm().item():.6f}  （门 g1=g2=0，残差分支被完全关掉，应严格为 0）')""")

code("""# ===== 完整 mini DiT: patchify + N blocks + unpatchify =====
class MiniDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_ch=3, dim=256, depth=6, n_heads=8, n_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # Patchify: Conv 2d 等价 patch embed
        self.patch_embed = nn.Conv2d(in_ch, dim, patch_size, patch_size)
        # 位置编码 (学习式)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        # 时间步和类别嵌入
        self.t_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.c_embed = nn.Embedding(n_classes + 1, dim)  # 留个 null 给 CFG
        # Blocks
        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads) for _ in range(depth)])
        # Final norm + head
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2*dim))
        nn.init.zeros_(self.final_mod[-1].weight); nn.init.zeros_(self.final_mod[-1].bias)
        self.final_proj = nn.Linear(dim, patch_size * patch_size * in_ch)
        nn.init.zeros_(self.final_proj.weight); nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x):
        B, N, D = x.shape
        P = self.patch_size
        H = W = int(N ** 0.5)
        C = D // (P*P)
        x = x.reshape(B, H, W, P, P, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H*P, W*P)
        return x

    def forward(self, x, t, y):
        '''x: [B, C, H, W]   t: [B]   y: [B] (class label)'''
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed

        t_emb = self.t_embed(t.unsqueeze(-1).float())
        y_emb = self.c_embed(y)
        c = t_emb + y_emb  # 合并条件

        for block in self.blocks:
            x = block(x, c)

        # Final layer also uses adaLN-Zero
        s, b = self.final_mod(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + s.unsqueeze(1)) + b.unsqueeze(1)
        x = self.final_proj(x)  # [B, N, patch_size^2 * C]
        x = self.unpatchify(x)
        return x

model = MiniDiT()
n_params = sum(p.numel() for p in model.parameters())
print(f'MiniDiT 参数量: {n_params/1e6:.2f}M')

# Smoke test
x = torch.randn(2, 3, 32, 32)
t = torch.rand(2)
y = torch.randint(0, 10, (2,))
out = model(x, t, y)
print(f'input: {x.shape}, output: {out.shape}')""")

code("""# ===== 参数量扫描: 验证 DiT 的 "scaling 友好性" =====
# 配置为 DiT 官方论文数值 (Peebles & Xie 2023, Table 1)
# 官方参数量参考: S=33M, B=130M, L=458M, XL=675M
# （我们的 MiniDiT 是 pixel 空间玩具版，绝对数值和官方 latent 版略有出入，看趋势）
configs = [
    ('DiT-S',  dict(dim=384,  depth=12, n_heads=6)),
    ('DiT-B',  dict(dim=768,  depth=12, n_heads=12)),
    ('DiT-L',  dict(dim=1024, depth=24, n_heads=16)),
    ('DiT-XL', dict(dim=1152, depth=28, n_heads=16)),
]

print(f"{'Name':<10} {'dim':>6} {'depth':>7} {'heads':>6} {'Params(M)':>12}")
for name, cfg in configs:
    m = MiniDiT(dim=cfg['dim'], depth=cfg['depth'], n_heads=cfg['n_heads'])
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{name:<10} {cfg['dim']:>6} {cfg['depth']:>7} {cfg['n_heads']:>6} {n:>10.1f} M")
print('\\n对比: GPT-2 Medium=350M, GPT-2 XL=1.5B → DiT-XL (675M) 和 GPT-2 中等规模相当')""")

md("""## Part 6 — 面试题（必考 Top 5）

**Q1：DiT vs U-Net 核心区别？**
- 架构：DiT 是纯 Transformer（patchify + self-attention），U-Net 是卷积 + 多尺度下采样
- 条件注入：U-Net 靠 cross-attention，DiT 靠 adaLN-Zero
- Scaling：DiT 符合 scaling law，U-Net 加参数收益递减
- 工业现状：2024 后新模型几乎全部换 DiT（SD3/FLUX/Sora）

**Q2：adaLN-Zero 的 "Zero" 指什么？为什么重要？**
- 指**输出调制参数 $\\alpha$ 的 MLP 末层零初始化**。初始化时 $\\alpha=0$，每个 Block 等价恒等映射，整个 DiT 初始就是 identity。训练极稳定，不会一开始炸梯度。这是 DiT 能 scale 到几十亿参数的关键工程技巧。

**Q3：一个 DiT Block 需要几个调制参数？都是干什么的？**
- **6 个**：$\\gamma_1, \\beta_1, \\alpha_1$（attention 分支的 scale/shift/gate）+ $\\gamma_2, \\beta_2, \\alpha_2$（MLP 分支的 scale/shift/gate）。$\\gamma, \\beta$ 调制 LayerNorm 输出，$\\alpha$ 是残差门。

**Q4：SD3 / FLUX 的 MM-DiT 对比原 DiT 改了什么？**
- 文本 token **和**图像 token **拼接**做联合 self-attention（原 DiT 文本只通过 adaLN 作为全局条件）。这样文本 token 和图像 token 能 token-level 交互，复杂 prompt 对齐效果显著更好。

**Q5：如果做一个中文实时数字人文生视频，用 DiT 怎么选型？**
- Backbone：DiT（MM-DiT）级别 1-3B 参数
- 训练：Flow Matching + Rectified Flow 路径（SD3 配方）
- 部署蒸馏：DMD 蒸到 1-4 步
- 条件：文本编码器可参考 SD3 的**三编码器**方案（CLIP-L/14 + OpenCLIP bigG + T5-XXL）+ adaLN-Zero 注入
- 推理：Ascend 910B 上 Transformer 算子原生支持，量化到 FP8 / INT8

**Q6（加分）：DiT 为什么 scaling 比 U-Net 友好？**
- Transformer 没有手工的空间归纳偏置（如卷积的局部性），表达能力上限更高
- 架构统一，只有 dim / depth / heads 三个旋钮，超参少
- 算子简单（LayerNorm + Attention + MLP），和 LLM infra 完全复用，scaling 工程成熟
- U-Net 的 skip connection、多尺度下采样是手工设计的"图像先验"，在大数据量下反而是瓶颈""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
with open("/path/to/yg/code/learning_everything/diffusion_scratch/09_dit_architecture.ipynb", "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("09 written")
