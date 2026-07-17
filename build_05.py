"""05 Score Matching - 纯理解篇，打通 Diffusion ↔ SDE 视角。不撕代码。"""
import json

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})

md("""# 05 - Score Matching 视角（Diffusion 的第二个名字）

> **这一篇不撕代码，专门理解**。面试官如果问"你对 Diffusion 的理解够不够深"，答案就藏在这里。

**一句话直觉**：DDPM 预测的"噪声 $\\epsilon$"和 Score-based Model 预测的"分数 $s$"，**是同一个东西的两种说法**——差一个随 $t$ 变化的缩放因子 $-1/\\sigma_t$。理解了这层等价性，就能把"像素空间的去噪"翻译成"概率密度上的梯度下降"。

**类比**：同一座山两种描述
- **DDPM 视角**：山坡上每一点"指向山谷的方向"（去噪方向）
- **Score 视角**：同一座山的"等高线梯度"（$\\nabla \\log p$）
- 两种地图，描述同一座山。

---

**符号提醒**：$x_t$ / $\\bar\\alpha_t$ / $\\epsilon_\\theta$ / $\\sigma_t$ 忘了？回 01 课符号表和 02 课新增表。本课新增：

| 新符号 | 含义 |
|--------|------|
| $\\nabla_x \\log p(x)$ | score：对数密度对 $x$ 的梯度，指向高密度区 |
| $s_\\theta(x_t, t)$ | score 网络：学 $\\nabla_{x_t} \\log p_t(x_t)$ |""")

md("""## 核心等价关系（面试必答）

给定噪声过程 $x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1-\\bar\\alpha_t} \\epsilon$：

$$
\\boxed{\\;\\; s_\\theta(x_t, t) = \\nabla_{x_t} \\log p_t(x_t) = -\\frac{\\epsilon_\\theta(x_t, t)}{\\sqrt{1-\\bar\\alpha_t}} \\;\\;}
$$

**一句话翻译**：score 就是 $-\\epsilon / \\sigma_t$（带个负号和标准差缩放）。

<details><summary>3 行推导（为什么 score 恰好是 $-\\epsilon/\\sigma$）</summary>

固定 $x_0$ 时 $x_t \\sim \\mathcal{N}(\\sqrt{\\bar\\alpha_t}\\,x_0,\\ (1-\\bar\\alpha_t)I)$，对高斯 log 密度求梯度：

$$\\nabla_{x_t} \\log q(x_t|x_0) = -\\frac{x_t - \\sqrt{\\bar\\alpha_t}\\,x_0}{1-\\bar\\alpha_t} = -\\frac{\\epsilon}{\\sqrt{1-\\bar\\alpha_t}}$$

（第二个等号代入 $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon$）

Denoising score matching 定理（Vincent 2011）：学**条件** score 的最优解 = **边缘分布**的 score。所以拿 $-\\epsilon/\\sqrt{1-\\bar\\alpha_t}$ 当回归目标，训出来的就是 $\\nabla \\log p_t$。
</details>

**所以**：
- 训练 $\\epsilon$-pred 模型（DDPM 视角） ≡ 训练 score model（SDE 视角）
- 两个领域的论文可以互相引用、互相对比
- 一切采样器（DDPM/DDIM/DPM-Solver/Heun）本质都是在求"朝 score 方向走"这个 ODE/SDE""")

md("""## 三个视角对比（面试展开题）

| 视角 | 代表工作 | 建模对象 | 采样 |
|------|----------|----------|------|
| **DDPM 视角**（离散） | Ho 2020 | Markov 链 + $\\epsilon$-pred | 随机采样 |
| **Score-based 视角**（离散噪声级） | Song & Ermon 2019 NCSN | 多个噪声级下的 $\\nabla \\log p_\\sigma$ | annealed Langevin |
| **SDE/ODE 统一**（连续，现代） | Song et al. ICLR 2021 | $dx = f(x,t)dt + g(t)dW$ 和 probability flow ODE | 任意 ODE solver |

**翻译词典**（跨视角读论文用）：

| DDPM 语言 | SDE/score 语言 |
|-----------|----------------|
| $\\beta$ schedule | 扩散系数 $g(t)$ |
| $\\epsilon$-pred | score（差 $-1/\\sigma_t$ 缩放） |
| ancestral sampling | Euler-Maruyama 离散化 |

**三者关系**：
- DDPM 的反向过程 ↔ 一个特定 SDE 的反向 SDE
- DDIM ↔ 同一个 SDE 的 **probability flow ODE**（把随机项去掉，确定性版本）
- 这也是为什么 DDIM 能跳步而 DDPM 不能——**ODE 是光滑的，SDE 有噪声**

**这个理论大一统**就是为什么现在论文不区分"是 DDPM 还是 score-based"，都叫 "score-based diffusion" 或 "continuous-time diffusion"。""")

md("""## Score 的直觉：梯度指向高密度区

想象数据分布 $p(x)$ 是一片山脉，**真实图像在山顶**（高密度），**纯噪声均匀分布在平地**。

- $\\nabla \\log p(x)$ 指向"山更高的方向"
- **采样过程 = 从平地出发，顺着梯度爬山**，最终到达山顶

但直接学 $\\nabla \\log p(x)$ 太难（$p$ 未知）。Score Matching 的妙招：

**Noise-Conditional Score**：不学 $p(x)$ 的 score，学 $p_t(x_t)$ 的 score（加噪后的分布）
- $p_T$（高噪声）几乎就是高斯，score 容易学
- $p_0$（无噪声）就是真实分布
- 把 $t$ 作为条件，从 $T$ 采到 $0$ = 从平地爬到山顶

这就是 **NCSN (Noise Conditional Score Network)** 的思想，早 DDPM 将近一年（2019.07 vs 2020.06），本质一样。""")

md("""## 为什么这个视角重要？（工程价值）

看似玄学，其实有三个非常实用的工程后果：

**1. 采样器大一统**
Euler / Heun / RK4 / DPM-Solver / UniPC —— 所有 2022 后的高阶采样器，都是把反向 ODE 当数值微分方程求解。不懂 score/ODE 视角，看不懂它们的论文。

**2. Classifier Guidance 的推导**
$$
\\nabla \\log p(x|y) = \\nabla \\log p(x) + \\nabla \\log p(y|x)
$$
= score + classifier gradient。CFG 的理论基础就在这里。

**3. Flow Matching 的来源**
Flow Matching 本质是"不学 score，学速度场 $v$"。这个式子不用背，只记结论：**$v$ 是 $x$ 的线性项加 score 的线性项，系数全由 schedule 决定**——"学 $v$"和"学 score"是同一信息的两种线性打包，SD3 从 diffusion 切 FM 不需要新理论。

（系数推导选读）在高斯路径 $x_t = \\alpha_t x_0 + \\sigma_t \\epsilon$ 下，$v$ 和 score 有精确换算：
$$
v_t(x) = \\frac{\\dot\\alpha_t}{\\alpha_t} x - \\sigma_t^2 \\left(\\frac{\\dot\\sigma_t}{\\sigma_t} - \\frac{\\dot\\alpha_t}{\\alpha_t}\\right) \\cdot s_t(x)
$$

<details><summary>选读：这个系数怎么来的（3 行）</summary>

由 Tweedie 型恒等式 $\\mathbb{E}[\\epsilon|x_t] = -\\sigma_t\\, s_t(x_t)$，得 $\\mathbb{E}[x_0|x_t] = (x_t + \\sigma_t^2 s_t)/\\alpha_t$。

边缘速度场是条件速度的期望：$v_t(x) = \\dot\\alpha_t\\,\\mathbb{E}[x_0|x] + \\dot\\sigma_t\\,\\mathbb{E}[\\epsilon|x]$。

代入两个条件期望、按 $x$ 和 $s_t(x)$ 归并同类项，即得上式。（可用一维高斯闭式解数值验证：取 $\\alpha=1-t, \\sigma=t$，$x_0 \\sim \\mathcal{N}(0,1)$，在 $t=0.6$ 处直接算 $v$ 和按公式换算的结果完全一致。）
</details>

一句话：**Flow Matching 和 Score Matching 可以互相转换**，并不是两个独立体系。这也是 SD3 能无缝从 Diffusion 切换到 Flow Matching 的理论基础。""")

md("""## 面试题连问

### Q1: Score Matching 和 DDPM 是什么关系？

<details><summary>参考答案</summary>

**同一个问题的两种参数化**：
- DDPM 让网络预测噪声 $\\epsilon$
- Score Matching 让网络预测 score $s = \\nabla \\log p_t(x_t)$
- 两者差一个随 $t$ 变化的缩放因子：$s = -\\epsilon / \\sigma_t$

所以 DDPM 训出来的模型可以**零改动**用于 SDE 采样框架，只要做个除法换算。DPM-Solver 这类高阶采样器内部就是先把 $\\epsilon$-pred / v-pred 换算成 score/ODE 语言再求解的。
</details>

### Q2: 为什么 score-based 比直接估计密度 $p(x)$ 更好学？

<details><summary>参考答案</summary>

直接估计 $p(x)$ 要满足归一化约束 $\\int p = 1$（要么 normalizing flow 硬约束架构，要么 VAE 引入 KL 近似）。

而 $\\nabla \\log p$ 是梯度，**归一化常数求导后消失**（$\\nabla \\log(Z \\cdot \\tilde p) = \\nabla \\log \\tilde p$），所以**不受归一化约束**。网络可以是任意神经网络，训练就是普通 MSE（在 denoising score matching 下）。这是 score-based 方法最重要的技术优势。
</details>

### Q3: Probability Flow ODE 是什么？有什么用？

<details><summary>参考答案</summary>

对每个 SDE $dx = f\\,dt + g\\,dW$，存在一个**和它同分布**的 ODE（无随机项）：
$$
dx = \\left[ f - \\frac{g^2}{2} \\nabla \\log p_t \\right] dt
$$

**这个 ODE 就是 DDIM 的连续版本**。用处：
1. **确定性采样**（输入噪声→输出图是函数关系）——图像编辑/inversion 的基础
2. **精确似然计算**（ODE 有 change-of-variable 公式，能算 $\\log p(x)$）
3. **跳步采样**（ODE 可以用大步长求解器，SDE 不能）

DDIM、DPM-Solver、Flow Matching 本质都在解这个 ODE。
</details>

### Q4: v-prediction / x-prediction / ε-prediction 为什么都能用？

<details><summary>参考答案</summary>

三种 parameterization 都是等价的（线性可逆）：

$$
\\begin{aligned}
\\text{ε-pred:}& \\quad \\epsilon = \\text{网络输出} \\\\
\\text{x-pred:}& \\quad \\hat x_0 = (x_t - \\sqrt{1-\\bar\\alpha_t}\\,\\epsilon) / \\sqrt{\\bar\\alpha_t} \\\\
\\text{v-pred:}& \\quad v = \\sqrt{\\bar\\alpha_t}\\,\\epsilon - \\sqrt{1-\\bar\\alpha_t}\\,x_0
\\end{aligned}
$$

**工程选择**（换算数值稳定性，Salimans & Ho 2022）：
- $\\epsilon$-pred：$\\epsilon$ 本身恒为单位方差，loss 没问题；但换算 $\\hat x_0 = (x_t - \\sqrt{1-\\bar\\alpha_t}\\,\\hat\\epsilon)/\\sqrt{\\bar\\alpha_t}$ 要除以 $\\sqrt{\\bar\\alpha_t}$——**大 t（低 SNR）时它趋近 0**，除近零数把预测误差放大，数值问题在大 t
- $x_0$-pred：对称问题——换算 $\\hat\\epsilon = (x_t - \\sqrt{\\bar\\alpha_t}\\,\\hat x_0)/\\sqrt{1-\\bar\\alpha_t}$ 要除以 $\\sqrt{1-\\bar\\alpha_t}$，**小 t 时**它趋近 0，数值问题在小 t
- **v-pred**（Imagen Video / SD2.1-768）：两端换算都不除近零数，数值稳定，蒸馏场景（Progressive Distillation）尤其需要

**注意区分两个维度**：这里说的是**换算数值稳定性**；02 课 Q2 表格里"$t$ 大/小时好学"说的是**目标可辨识度**（预测目标在该区间信息量大不大）。两回事，面试别混。

**面试陷阱**：问"三种区别"，答"数值稳定性"才是加分项，说"它们等价"是基础分。
</details>

### Q5: 为什么有了 Flow Matching 还要懂 Score Matching？

<details><summary>参考答案</summary>

1. **理论基础相通**：FM 训练目标在高斯路径下可以推回 score matching 损失，懂一个通另一个
2. **历史文献读懂**：2020-2023 主流论文都用 score 语言（NCSN, EDM, Karras et al.），懂这套才看得懂 FID 记录保持者 EDM 的那套 $\\sigma(t)$ 设计
3. **采样器复用**：DPM-Solver、Heun 等高阶求解器是针对 score/ODE 框架设计的，Flow Matching 可直接借用
4. **面试区分度**：能把 DDPM / Score / Flow 三套框架串起来的候选人，明显比只懂一个的深度更高
</details>""")

nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
with open('/home/yg/yg/code/learning_everything/diffusion_scratch/05_score_matching.ipynb','w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('05 written:', len(cells), 'cells')
