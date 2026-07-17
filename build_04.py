"""04 CFG - classifier-free guidance. 重理解，代码只撕核心公式。"""
import json, os

cells = []

def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})

md("""# 04 - Classifier-Free Guidance (CFG)

> **前情提要**：02/03 我们能"无条件"生成（从噪声生成任意图），但真实场景需要"生成一只戴皇冠的猫"——**条件生成**。CFG 是当前所有文生图/文生视频的标配。

**一句话直觉**：训练一个模型**既会看条件又会无视条件**，推理时让它"听条件的话"那一支减去"无视条件"那一支，再把差距放大 `w` 倍 —— 条件信号就被"放大声喊"。

**类比**：两个声音混音
- 有条件的 $\\epsilon_{cond}$（小声喊："画一只猫"）
- 无条件的 $\\epsilon_{uncond}$（噪声基线）
- CFG 推理：$\\epsilon = \\epsilon_{uncond} + w \\cdot (\\epsilon_{cond} - \\epsilon_{uncond})$

`w` 叫 guidance scale，SD 默认 7.5。`w=1` 是普通条件生成，`w>1` 是"加强条件"。

---

**符号提醒**：$x_t$ / $\\bar\\alpha_t$ / $\\epsilon_\\theta$ 忘了？回 01 课符号表和 02 课新增表。本课新增：

| 新符号 | 含义 |
|--------|------|
| $c$ | 条件（文本 embedding / 类别标签等） |
| $\\varnothing$ | 空条件（null embedding / 空 prompt） |
| $w$ | guidance scale：条件方向的放大倍数 |""")

md("""## 为什么需要 CFG？演进路线

| 阶段 | 方案 | 痛点 |
|------|------|------|
| **2021** Classifier Guidance（Dhariwal & Nichol） | 另训一个分类器 $p(y\\|x_t)$，用它的梯度引导采样 | 要额外训练一个噪声分类器，麻烦 |
| **2022** Classifier-**Free** Guidance | **同一个模型**随机丢弃条件训练，推理时组合 | 零额外成本，效果更好 |

**关键洞察**（Ho & Salimans, 2022）：训练时以 10% 概率把条件设为"空条件"，这样同一个 $\\epsilon_\\theta$ 既能做 $\\epsilon(x_t, c)$ 也能做 $\\epsilon(x_t, \\varnothing)$。推理时就可以做差组合。

**这个技巧同时解决了**：
1. 条件-无条件可以共享参数（省显存）
2. 不需要额外训分类器
3. 不需要有标签的数据（空条件随时可造）""")

md("""## CFG 公式一页纸

**训练阶段**（唯一改动：条件随机丢弃）：
```
for (x_0, c) in dataset:
    if random() < 0.1: c = ∅   # 10% 概率丢条件
    t ~ Uniform[1, T]
    ε ~ N(0, I)
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    loss = || ε_θ(x_t, t, c) - ε ||²    # c 可能是 ∅
```

**推理阶段**（每步 forward 两次）：
$$
\\hat\\epsilon = \\epsilon_\\theta(x_t, t, \\varnothing) + w \\cdot \\big[ \\epsilon_\\theta(x_t, t, c) - \\epsilon_\\theta(x_t, t, \\varnothing) \\big]
$$

**几何直觉**：在 $\\epsilon$-空间里把"条件方向"放大 $w$ 倍，让采样轨迹更强地偏向条件。画成向量外推就是：

```
   w=0            w=1                          w=7.5
    ●──────────────●━━━━━━━━━━━━━━━━━━━━━━━━━━━━▶
 ε_uncond        ε_cond                      ε̂ = ε_uncond + w·(ε_cond − ε_uncond)
 (无条件基线)   (普通条件生成)               (沿"条件方向"继续外推 w 倍)
```

$w=1$ 正好落在 $\\epsilon_{cond}$；$w>1$ 沿着 $\\epsilon_{cond} - \\epsilon_{uncond}$ 这个方向**冲出去**——条件信号被放大。

**成本**：每步推理要跑 2 次网络（双倍计算），所以工业界对 CFG 加速有大量研究（CFG++、guidance distillation 等）。""")

md("""## 代码手撕 CFG 采样（核心 3 行）

不做完整训练演示（和 02 结构一样，只是多个 label embedding），**只把 CFG 最关键的 3 行抽出来面试现场手撕**。""")

code("""import torch

@torch.no_grad()
def cfg_ddim_step(model, x_t, t, alpha_bars, cond, uncond, w=7.5):
    \"\"\"CFG + DDIM 单步采样。面试要会手撕这个函数。\"\"\"
    # ---- CFG 核心 3 行 ----
    eps_cond   = model(x_t, t, cond)      # 有条件
    eps_uncond = model(x_t, t, uncond)    # 无条件 (空 prompt / null embedding)
    eps = eps_uncond + w * (eps_cond - eps_uncond)    # <-- CFG 组合
    # ----------------------

    # DDIM 去噪一步（复用 03 的 ODE 公式）
    ab_t = alpha_bars[t]
    ab_prev = alpha_bars[t-1] if t > 0 else torch.tensor(1.0)
    x0_hat = (x_t - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    x_prev = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
    return x_prev

# 工程 trick：batch 拼接避免跑两遍
@torch.no_grad()
def cfg_batched(model, x_t, t, cond, uncond, w):
    \"\"\"一次 forward 算两种条件，减半延迟。工业部署标准写法。\"\"\"
    x_both = torch.cat([x_t, x_t], dim=0)
    c_both = torch.cat([uncond, cond], dim=0)
    eps_both = model(x_both, t, c_both)
    eps_uncond, eps_cond = eps_both.chunk(2, dim=0)
    return eps_uncond + w * (eps_cond - eps_uncond)

print('CFG 核心逻辑：uncond + w*(cond - uncond)')
print('工业部署：batch concat 一次 forward 拿两种预测')
""")

md("""## Guidance Scale `w` 的经验值

| w | 效果 | 适用 |
|---|------|------|
| `1.0` | 普通条件生成（没有放大） | debug |
| `3-5` | 温和放大，多样性好 | 艺术创作 |
| **`7.5`** | **SD 默认**，工业标准 | 通用文生图 |
| `10-15` | 强贴合 prompt，但图质容易崩 | 需要极强条件控制时 |
| `>20` | 失真、过饱和、伪影 | ❌ 基本不用 |

**经验规律**：$w$ 和 prompt-图对齐度是正相关，和图像自然度/多样性是负相关。选 `w` 就是在"听话"和"好看"之间找平衡。

**前沿改进**：
- **Dynamic CFG**：不同 timestep 用不同 w（早期强条件、后期弱条件），社区/后续论文常用做法
- **CFG++**（2024）：把 CFG 改写成流形投影问题，小 w 就能达到大 w 效果
- **APG (Adaptive Projected Guidance)**：防止 CFG 把图像推出真实流形""")

md("""## 为什么 CFG 能"放大条件信号"？数学解释

三步走完，每步只用一个已知事实：

**第 1 步｜把 $\\epsilon$ 翻译成 score**（05 课会证）：

$$\\epsilon_\\theta(x_t, t) \\approx -\\sigma_t \\nabla_x \\log p(x)$$

**第 2 步｜做差**：

$$\\epsilon_{cond} - \\epsilon_{uncond} = -\\sigma_t\\big[\\nabla_x \\log p(x|c) - \\nabla_x \\log p(x)\\big] = -\\sigma_t \\nabla_x \\log p(c|x)$$

（第二个等号用贝叶斯：$\\log p(x|c) - \\log p(x) = \\log p(c|x) - \\log p(c)$，而 $\\log p(c)$ 对 $x$ 求导为 0）

**第 3 步｜所以 CFG 组合式等价于**：

$$\\hat\\epsilon = \\epsilon_{uncond} - \\sigma_t \\cdot w \\cdot \\nabla_x \\log p(c|x)$$

> 📖 读式子：第二项就是一个**分类器的梯度**，但这个分类器没人训练——是 $\\epsilon$ 网络免费送的。"classifier-**free**" 的名字由此而来。

**理论暗面**：这也是 CFG 的原罪——近似等价于从 $p(x|c)^w \\cdot p(x)^{1-w}$ 采样，**当 w > 1 时这个分布是 "扭曲版" 的真实分布**，过大 w 就会跑出流形，出现过饱和、伪影。这也是 CFG++ 等后续工作要解决的问题。""")

md("""## 面试题连问

### Q1: CFG 相比 Classifier Guidance 的优势？

<details><summary>参考答案</summary>

1. **无需额外训练分类器**（Classifier Guidance 要单独训一个 $p_\\phi(y|x_t)$，且要在**带噪数据**上训）
2. **不需要有标签数据**（CFG 的"空条件"随时可造）
3. **同一模型复用**，存储/部署成本减半
4. **工程实测效果更好**（可能因为分类器引导的梯度质量不如 $\\epsilon_\\theta$ 学到的）
</details>

### Q2: 训练时为什么要丢 10% 条件？丢 50% 行不行？

<details><summary>参考答案</summary>

- **丢太少**（< 5%）：模型没学好 $\\epsilon(x_t, \\varnothing)$，CFG 差那支会坏
- **丢太多**（> 30%）：有条件那支没学好，条件生成精度下降
- **10%** 是 Ho & Salimans 的经验值，SD 系列也沿用

这是经典的"共享容量"问题——同一个网络要学两个分布，比例很关键。
</details>

### Q3: CFG 为什么让图像过饱和？怎么解决？

<details><summary>参考答案</summary>

- **原因**：当 $w > 1$，采样分布近似变成 $p(x|c)^w \\cdot p(x)^{1-w}$，不再是真实分布。$w$ 越大越偏离流形
- **现象**：颜色过鲜艳、对比度过高、手指变形、数字崩坏
- **解法**：
  - 降低 $w$（3-5 比 7.5 更自然）
  - **Dynamic CFG**：不同 timestep 用不同 w
  - **CFG++**：把 guidance 改成流形投影
  - **Rescaling**：做 CFG 后把 std 缩回原值（rescale trick）
</details>

### Q4: CFG 每步都要 forward 两次，能加速吗？

<details><summary>参考答案</summary>

1. **Batch 拼接**：一次 forward 跑 `[uncond; cond]`，工业标配
2. **Guidance Distillation（Meng et al.）/ LCM**：蒸馏出一个单支网络直接吃 `(x, t, c, w)`，输出 CFG 后的 eps。注意别和 SD-Turbo 混：SD-Turbo 走的是**对抗蒸馏（ADD）**，不是 guidance distillation
3. **CFG caching**：早期 t 的 uncond 分支差异小，可以间隔计算
4. **CFG++**：小 w 效果等价大 w，间接省计算
</details>

### Q5: Negative prompt 和 CFG 什么关系？

<details><summary>参考答案</summary>

Negative prompt（"不要出现 ugly/low quality"）本质是**把 uncond 那一支换成 "negative prompt 条件"**：

$$
\\epsilon = \\epsilon_{neg} + w \\cdot (\\epsilon_{cond} - \\epsilon_{neg})
$$

几何上：不是从"空条件"朝"cond"走，而是从"neg"朝"cond"走，路径更有针对性。Stable Diffusion WebUI 的 negative prompt 就是这样实现的。
</details>

### Q6: SD3 / FLUX 用 Flow Matching 后 CFG 还有用吗？

<details><summary>参考答案</summary>

**有用且必需**。CFG 是**条件生成**的 guidance 技术，和底层是 DDPM/DDIM/Flow Matching 无关。Flow Matching 预测的是速度场 $v$，CFG 就对 $v$ 做同样的组合：

$$
\\hat v = v_{uncond} + w \\cdot (v_{cond} - v_{uncond})
$$

SD3、FLUX、Sora 全都保留 CFG，只是底层采样从 DDIM 换成了 ODE 求解器。
</details>""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                 "language_info": {"name":"python","version":"3.11"}},
    "nbformat": 4, "nbformat_minor": 5
}
with open('/path/to/yg/code/learning_everything/diffusion_scratch/04_cfg.ipynb','w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('04_cfg.ipynb written:', len(cells), 'cells')
