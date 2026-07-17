# Diffusion From Scratch

从零推导 + 从零实现的扩散模型（Diffusion Model）中文学习笔记，共 10 个 Jupyter Notebook，覆盖从 DDPM 到 Flow Matching、Consistency Model、DiT 的完整技术演进线。

**特点**：

- **直觉先行**：每个概念先给生活类比和一句话直觉，再给数学推导，公式中每个符号都有解释，不跳步
- **公式与代码对照**：核心公式全部落到可运行的最小 PyTorch 实现，玩具数据（2D 高斯 / 合成分布）上直接可视化验证
- **面试导向**：标注了高频考点（adaLN-Zero、CFG、DDIM 与 DDPM 的关系、ε/score/v-pred 等价性等），每课末尾附检验性问题
- **演进脉络**：不是孤立讲 10 个算法，而是回答"为什么下一代技术要这么改"

## 目录

| Notebook | 主题 | 一句话 |
|----------|------|--------|
| [01](01_forward_diffusion.ipynb) | Forward Diffusion 前向加噪 | 把图一步步变成雪花噪声：β/α/ᾱ 记号、一步到位公式、noise schedule 与 SNR |
| [02](02_ddpm_training_sampling.ipynb) | DDPM 训练 + 反向采样 | 贝叶斯后验推导 μ̃/β̃，训练"猜噪声"的网络，反向走 1000 步生图 |
| [03](03_ddim.ipynb) | DDIM 确定性采样 | 不重训、只改采样：随机游走 → 确定性路径，50 步顶 1000 步 |
| [04](04_cfg.ipynb) | Classifier-Free Guidance | 条件生成的标配：有条件减无条件，把条件信号"放大声喊" |
| [05](05_score_matching.ipynb) | Score Matching 视角 | 预测噪声 ≡ 预测分数（∇log p）：Diffusion 的第二个名字 |
| [06](06_flow_matching.ipynb) | Flow Matching | 现代主流范式（SD3 / FLUX）：不绕加噪弯路，直接学向量场走直线 |
| [07](07_rectified_flow.ipynb) | Rectified Flow | Reflow 把路径越拉越直，一步生成的前哨站 |
| [08](08_consistency_lcm.ipynb) | Consistency Models / LCM | 训练"瞬移函数"：轨迹上任何位置一步跳到终点 |
| [09](09_dit_architecture.ipynb) | DiT 架构 ⭐ | U-Net 换 Transformer：Sora / SD3 / FLUX 的共同底座，adaLN-Zero 详解 |
| [10](10_mini_diffusion_system.ipynb) | 总装 & 全景图 | 三代技术演进地图 + DDPM vs Flow Matching 同场对比实验 + 选型指南 |

## 学习路径

```
01 前向加噪 ──▶ 02 DDPM ──▶ 03 DDIM ──▶ 04 CFG        ← 经典 Diffusion 一条龙
                  │
                  ▼
              05 Score 视角                              ← 换个角度看同一件事
                  │
                  ▼
06 Flow Matching ──▶ 07 Rectified Flow ──▶ 08 Consistency/LCM   ← 拉直路径、减少步数
                  │
                  ▼
              09 DiT 架构 ──▶ 10 总装                    ← 架构与全景
```

建议顺序阅读：每课开头都有"符号提醒"衔接前课，技术点层层递进。

## 运行环境

```bash
pip install torch numpy matplotlib jupyter
jupyter lab
```

所有实验都在玩具数据上进行，**CPU 即可运行**，无需下载数据集和预训练权重。

## 关于 build_*.py

`build_NN.py` 是各 notebook 的生成脚本（notebook 即代码的构建产物），日常学习直接打开 `.ipynb` 即可。`build_all.py` 已废弃，仅存档。

## License

MIT
