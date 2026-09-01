---
title: "Leveraging-Perturbation-Robustness-to-Enhance-Out-of-Distrib"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Leveraging_Perturbation_Robustness_to_Enhance_Out-of-Distribution_Detection_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:23:14"
field: "分布外检测（Out-of-Distribution Detection）"
keywords: ["OOD Detection", "Perturbation Robustness", "Adversarial Training", "Softmax-based Scoring", "Post-hoc Method", "Near-OOD Detection", "OpenOOD"]
innovations: ["提出对抗打分函数 g*，通过在输入邻域极小化原分数增强 IND/OOD 可分性", "证明对抗训练为 IND softmax 分数提供扰动下界，桥接对抗鲁棒性与 OOD 检测", "PRO 作为通用后处理插件，可无损叠加至 MSP/Entropy/GEN 等 softmax 得分之上"]
benchmarks: ["OpenOOD", "RobustBench", "CIFAR-10", "CIFAR-100", "ImageNet-1K"]
---

# 论文速读：Leveraging-Perturbation-Robustness-to-Enhance-Out-of-Distribution-Detection

## 一句话总结
本文提出后处理方案 PRO（Perturbation-Rectified OOD Detection），利用"OOD 输入的置信度对扰动更敏感"这一观察，通过对抗梯度下降在输入附近搜索局部最小值来增强 IND/OOD 的可分性；结合对抗训练模型，PRO 在小规模模型（CIFAR-10/100）的 near-OOD 检测上达到领先后处理方法的水平。

## 研究问题与动机
- **near-OOD 难区分**：现有 OOD 检测方法在区分训练分布附近的新类别数据（如 CIFAR-100 之于 CIFAR-10）时，softmax-based 方法易因 near-OOD 的虚假高置信度而失效。
- **梯度预处理方法（如 ODIN）存在缺陷**：OpenOOD 评测表明 ODIN 在多种任务上反而降低 MSP 性能，因其倾向于抬高 near-OOD 的预测置信度，使区分更难。
- **对抗鲁棒性与 OOD 检测的关联尚未被充分利用**：已有工作建立两者关联，但本文系统性地利用对抗训练带来的"IND 分数对扰动更鲁棒"的差异作为探测信号，填补这一空白。

## 核心贡献（创新点）
1. **提出对抗打分函数 $g^\star$**：通过对候选分数 $g$ 在 $\epsilon$-有界扰动内取最小值，使 OOD 分数被压低，增强 IND/OOD 可分性；与 ODIN 类方法（梯度最大化）本质不同，本方法是极小化而非极大化。
2. **建立对抗鲁棒性与 OOD 检测之间的理论桥梁**：证明对抗训练下的有界交叉熵损失为 IND 数据的 softmax 分数提供了扰动后的下界，从而形式化 IND 对扰动的鲁棒优势；这是首次给出此类理论绑定。
3. **提出无需修改模型结构的后处理通用增强策略**：PRO 可挂载到 MSP、Entropy、Temperature Scaling、GEN 等任意 softmax-based 得分之上，以简单预处理步骤换取显著性能提升，且不增加推理架构复杂度。

## 方法详解
- **核心观察**：在相同扰动界 $\epsilon$ 下，OOD 分数期望变化量大于 IND 分数期望变化量，即
  $$\mathbb{E}_{P_{\mathrm{OOD}}}[ \Delta z(g, \mathbf{x}) ] > \mathbb{E}_{P_{\mathrm{IND}}}[ \Delta z(g, \mathbf{x}) ],$$
  其中 $\Delta z(g, \mathbf{x}) = \max_{\|\delta\|_\infty \le \epsilon} |g(\mathbf{x}) - g(\mathbf{x}+\delta)|$。
- **对抗打分定义**：
  $$g^\star(\mathbf{x}) = \min_{\|\delta\|_\infty \le \epsilon} g(\mathbf{x}+\delta).$$
- **求解算法（FGM 迭代极小化）**：
  初始化 $\mathbf{x}_0$，对 $t=0,\dots,K$ 执行：
  $$\delta = -\epsilon \operatorname{sign}\!\big(\nabla_{\mathbf{x}_t} g(\mathbf{x}_t)\big), \quad \mathbf{x}_{t+1} = \mathbf{x}_t + \delta,$$
  最终取全部中间得分最小值：$g^\star(\mathbf{x}) \approx \min\{g(\mathbf{x}_0), \dots, g(\mathbf{x}_K)\}$。
- **可选变体**：使用 MSE/MSP、Shannon Entropy（PRO-ENT）、Temperature Scaling（PRO-MSP-T）、Generalized Entropy（PRO-GEN）作为基础分数 $g$。
- **对抗训练的作用**：对抗训练的有界损失 $\mathbb{E}[\max_{\|\delta\|_p<\epsilon} \mathcal{L}_{CE}] < \mathcal{E}$ 经 Jensen 不等式推导出 MSP 扰动下界的下界为 $\exp(-\mathcal{E})$，从而确保 IND 分数在扰动下相对稳定。

## 实验与结果
- **数据集与评测基准**：OpenOOD [43]；近 OOD（near-OOD）：CIFAR-100、Tiny-ImageNet（TIN）、SSB、NINCO；远 OOD（far-OOD）：MNIST、SVHN、Texture、Places365、iNaturalist、OpenImage-O。
- **主要基线**：MSP、Entropy、TempScaling、GEN、ODIN、MDS、EBO、VIM、KNN、Scale、ASH、ReAct、MLS、RMDS。
- **CIFAR-10 为 IND（Table 1 关键数字）**：
  - **默认模型**：PRO-GEN 平均 FPR@95=29.60、AUROC=91.79；near-OOD 上 PRO-GEN：CIFAR-100=37.38/89.50，TIN=30.37/91.90，超越 VIM（31.65/91.88）、KNN（27.52/92.19）。
  - **对抗鲁棒模型（LRR-CARD-Deck [8]）**：PRO-GEN 平均 FPR@95=19.82、AUROC=95.00；near-OOD 上 CIFAR-100=29.56/91.85，TIN=21.96/94.48。
- **CIFAR-100 为 IND（Table 2）**：鲁棒模型上 PRO-MSP 近 OOD FPR@95=77.39（远 OOD=58.42）；PRO-GEN 近 OOD=52.38/82.50，远 OOD=55.89/78.42，为最强后处理方法之一。
- **ImageNet（Fig.5）**：模型规模增大后 PRO 增益衰减；但在 PixMix/AugMix 辅助下，PRO-MSP/PRO-ENT 仍在 near-OOD 上获得 AUROC 提升。
- **最强结果**：鲁棒 CIFAR-10 模型上 PRO-GEN 平均 FPR@95=19.82 vs. 基线 MSP=44.92，**降幅超 50%**；near-OID 的 TIN 上 FPR@95 从 34.62 降至 21.96。

## 相关工作脉络
- **ODIN [28]**：同为梯度预处理方法，但 ODIN 最大化扰动后分数（提高 IND 置信度），PRO 极小化（压低 OOD 分数），二者策略相反；实验显示 ODIN 在 near-OOD 上效果劣于 PRO 且可能降低 MSP 性能。
- **VIM [41]/KNN [37]**：基于 IND 特征的判别方法，需访问训练数据；PRO 无需 IND 数据即可工作，是后处理、IND-free 的替代方案。
- **MSP [15] / Entropy [15] / GEN [30]**：纯 softmax-based 基线；PRO 是对这些得分的通用增强模块，非替代关系。
- **Scale [42] / ASH [9] / ReAct [36]**：通过激活修改增强 Energy-based 得分，在 ImageNet 规模上表现突出；PRO 在 ImageNet 上增益有限，体现其对模型规模的敏感性。
- **RobustBench [5]**：对抗鲁棒性评测平台；本文首次系统地将该平台的鲁棒模型引入 OOD 检测评测，桥接两个安全研究方向。
- **EBO [29]**：Energy-based 检测器；本文 Fig.6 可视化显示 logit-based 得分比 softmax-based 得分景观更陡峭，解释了 PRO 更适配 softmax 分数的原因。

## 局限性与未来方向
- **模型规模敏感性**：随类别数和模型规模增大（ImageNet），IND 分数对扰动的敏感度上升，PRO 增益明显衰减（Fig.7）。
- **远 OOD 检测提升有限**：PRO 在 near-OOD 场景效果最佳，对 far-OOD 的改善较弱（Fig.5、Sec.5.2.3）。
- **鲁棒训练方法的异质性**：并非所有对抗训练策略均利好 PRO（NoisyMix、SIN-IN 导致 softmax 得分退化）。
- **未来方向**：（1）探索更适合大规模模型的扰动框架；（2）研究哪些对抗训练协议能最大化 PRO 增益；（3）将 PRO 与能量/激活修改类方法结合。

## 研究启发与可借鉴点
- **"分数敏感性差异"思想可迁移**：本工作揭示 IND/OOD 在扰动下分数变化的系统性差异，这一信号可用于其他检测任务（如异常检测、开放集识别），设计类似的极小化预处理器。
- **对抗鲁棒性训练的显式利用**：标准训练已产生一定鲁棒性差异，对抗训练进一步强化；未来可探索将鲁棒预训练与 OOD 检测联合优化。
- **实验设计的严谨性**：同时评测默认模型与鲁棒模型、near/far-OOD 分层报告、可视化 score landscape 和 shift 分布等多角度验证，为后续评测提供了完整模板。
- **后处理模块化思路**：PRO 作为 score 的插件式增强，可无缝叠加在任意 softmax-based 方法之上，为团队协作中"复用现有方法"提供灵活接口。

## 关键术语表
- **Out-of-Distribution (OOD) Detection**：判断输入是否来自与训练分布不同的数据分布的任务，保障模型在开放世界的部署安全。
- **In-Distribution (IND)**：与模型训练数据同分布的输入。
- **Maximum Softmax Probability (MSP)**：以 softmax 输出最大概率作为 IND/OOD 判定分数，是最基础的 OOD 检测基线。
- **Generalized Entropy (GEN)**：softmax-based 扩展得分，通过参数 $\gamma$ 和 $M$ 调整各预测概率的非线性组合。
- **Adversarial Score Function ($g^\star$)**：在原分数 $g$ 基础上，通过有界扰动极小化得到的增强分数。
- **Near-OOD vs. Far-OOD**：near-OOD 是与训练分布结构相似的新类别（如 CIFAR-100 vs. CIFAR-10），检测难度大；far-OOD 则是明显不同的数据（如 MNIST、Texture）。
- **Lipschitz 常数**：描述函数对输入扰动的敏感程度；IND 分数的有效 Lipschitz 常数更小，因此更鲁棒。
- **OpenOOD Benchmark**：全面评测 OOD 检测方法的综合基准，涵盖 CIFAR 和 ImageNet 上的多种指标与模型。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100（公开）；Tiny-ImageNet、MNIST、SVHN、Texture、Places365、ImageNet-1K（均公开）。
- **代码/权重**：论文未提及开源仓库；鲁棒模型权重来自 RobustBench（[5]）及 [8]。
- **关键超参**：扰动步长 $\epsilon$ 和步数 $K$ 在 OpenOOD 的超参列表中选择最优值；温度 $T$ 可选。
- **骨干网络**：CIFAR 使用 ResNet-18/ResNet-50；鲁棒模型使用 WideResNet（详见 RobustBench）。
