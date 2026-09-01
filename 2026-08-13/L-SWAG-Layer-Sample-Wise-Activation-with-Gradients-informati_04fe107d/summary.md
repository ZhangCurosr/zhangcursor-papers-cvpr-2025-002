---
title: "L-SWAG-Layer-Sample-Wise-Activation-with-Gradients-informati"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Casarin_L-SWAG_Layer-Sample_Wise_Activation_with_Gradients_Information_for_Zero-Shot_NAS_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:22:47"
field: "神经架构搜索 (NAS)"
keywords: ["Zero-Shot NAS", "Vision Transformer", "Proxy Metric", "Neural Architecture Search", "L-SWAG", "LIBRA-NAS", "Autoformer"]
innovations: ["提出L-SWAG指标，结合层-样本级梯度统计与表达性度量，首次在ViT搜索空间上系统性地超越参数量基线并提升多基准ranking相关性", "提出LIBRA-NAS集成算法，通过信息增益最小化和偏差匹配策略无训练代价地组合互补代理，显著提升跨基准性能", "构建包含ViT的扩展基准并适配GeLU网络，统一评估了20种ZC代理在新旧空间上的表现"]
benchmarks: ["NASBench-201", "TransNAS-Bench-101", "Autoformer"]
---

# 论文速读：L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers

## 一句话总结
本文针对Vision Transformer (ViT) 搜索空间上现有零成本（Zero-Cost, ZC）代理指标性能下降的问题，提出了**L-SWAG**（层-样本级激活梯度信息）指标，同时构建了包含ViT任务的新基准；并设计了**LIBRA-NAS**集成算法，在仅0.1 GPU天的搜索时间内于ImageNet上找到了测试误差为17.0%的优质ViT架构。

## 研究问题与动机
1. **现有ZC代理的适用性局限**：当前SOTA的ZC代理主要在设计用于卷积神经网络（ConvNet）的搜索空间（如NAS-Bench-201）上评估，在新兴的ViT搜索空间上表现不佳，多数甚至无法超越简单的参数量（# Parameters）指标。
2. **缺乏统一公平评估**：不同代理通常在不同设置下被独立评估，缺乏在统一设置（特别是包含ViT任务）下的对比，导致难以厘清各方法的真实贡献。
3. **单一代理的信息局限**：不同代理往往包含互补的信息，且对搜索空间特征（如更偏好梯度基还是无梯度基）存在不同的偏差，单一代理难以在所有场景下达到最优相关性。
4. **ViT模型的表达性需求**：ViT广泛使用GeLU激活函数，而许多现有ZC代理仅针对ReLU网络设计，需适配以支持更广泛的架构类型。

## 核心贡献（创新点）
1. **提出L-SWAG指标**：通过层-样本级的梯度信息统计与表达性度量相结合，统一刻画卷积网络和Transformer网络的训练性和表达力，在ViT搜索空间上显著优于现有方法。
2. **构建ViT新基准并扩展评估**：在Autoformer ViT搜索空间上训练并评估了2000个模型在6个不同任务上的性能，完成了对所有主流ZC代理的公平、统一基准测试，并将GeLU网络纳入评估范围。
3. **提出LIBRA-NAS集成算法**：设计了一种基于信息增益（Information Gain）和偏差重对齐（Bias Re-alignment）的机器学习集成方法，无需训练预测器即可策略性地组合多个代理，在各种基准上显著提升了 ranking 一致性。

## 方法详解
**1. L-SWAG (Layer-Sample Wise Activation with Gradients)**
公式：$\mathrm {L-SWAG} = \Lambda^{\hat{L}} \times \Psi_{\mathcal{N}, \boldsymbol{\theta}}^{\hat{L}}$

*   **可训练性项 ($\Lambda^{\hat{L}}$)**：改进自ZiCO，但放弃了梯度均值（$\mu$），仅利用梯度标准差（$\sigma$）。通过对选定层（$\hat{l}$ 到 $\hat{L}$）的参数梯度绝对值方差的倒数求和取对数得出。这基于理论证明（线性回归损失上界）和经验验证，去除了$\mu$能提升在ViT等复杂空间上的表现。
*   **表达性项 ($\Psi_{\mathcal{N}, \boldsymbol{\theta}}^{\hat{L}}$)**：定义为层-样本级激活模式（SWAP）集合的基数。即统计经过ReLU或GeLU激活后，给定输入批次中所有样本的二值化激活模式的种类数量，以此衡量网络的表达能力。
*   **层选择策略**：通过分析大量随机初始化网络的梯度统计量分布（分位数），发现特定层的梯度统计与最终性能相关性更高（表现为图2b中的峰值），仅选取这些高信息量层进行计算，既加速了计算又提升了相关性。

**2. LIBRA-NAS (Low Information gain and Bias Re-Alignment)**
一种代理集成策略，输入为一组预计算的ZC代理及其在基准上的偏差值：
*   **步骤一**：选择在给定基准上Spearman相关系数（$\rho$）最高的代理作为 $z_1$。
*   **步骤二**：在相关性接近的候选代理中，计算每个代理相对于 $z_1$ 的条件熵和信息增益（IG），选择使IG最小（即与 $z_1$ 提供的信息重叠度最高、冗余度最大但又能补充相同偏差）的代理作为 $z_2$。
*   **步骤三**：在剩余候选代理中，选择其偏差（此处指与验证准确率相关的参数数量等偏差）与验证准确率偏差最接近的代理作为 $z_3$。
*   最终输出最优的三个代理组合用于排名。

## 实验与结果
*   **数据集与基准**：NASBench-201 (Cifar-10/100, ImageNet16-120), NASBench-101, NASBench-301, TransNAS-Bench-101 (Micro/Macro, 7个子任务), 以及新建的 **Autoformer** ViT搜索空间（6个任务：ImageNet, Cifar10, Cifar100, Pets, SVHN, Spherical-Cifar100）。
*   **L-SWAG 排序一致性**：在多个基准上平均Spearman $\rho$ 达到 **0.72**，优于第二名的NWOT (0.62)。在Autoformer ViT搜索空间中，L-SWAG是唯一 consistently 超过或持平于参数量/FLOP计数这一简单基线的指标。
*   **LIBRA-NAS 搜索结果**：
    *   在19个任务中有13个任务的代理组合性能优于其他集成方法。
    *   在 **Autoformer + ImageNet1k** 搜索实验中，LIBRA-NAS仅用时 **0.1 GPU天**，搜索到了测试误差为 **17.0%** 的架构，显著优于进化（Evolution）和基于梯度（Gradient-based）的方法（见表2）。
    *   消融实验证实了移除梯度均值（no $\mu$）、层选择（L̂）和表达性项（Ψ）均能带来性能提升，且层选择对Macro搜索空间效果显著，表达性项对Micro搜索空间更有益。

## 相关工作脉络
1.  **NAS-Bench-SuiteZero [22]**：提供了一个统一的ZC代理评估代码库和多种代理的比较。本文在此基础上扩展了ViT搜索空间和更多近期代理，并提出了不需要训练预测器的集成方法LIBRA。
2.  **ZiCO [26]**：通过梯度系数的变异系数来评估可训练性。L-SWAG在理论和方法上继承并改进了ZiCO，论证了去除梯度均值并使用层选择和表达性项的必要性和有效性。
3.  **SWAP-NAS [36] / NWOT [33]**：基于激活模式的无梯度表达性指标。L-SWAG借鉴了其表达性度量思路，但将其与梯度信息结合，并扩展到GeLU网络，解决了纯无梯度方法在细粒度Micro搜索空间表现不佳的问题。
4.  **Te-NAS [7] / T-CET [46]**：结合了线性区域数和NTK条件数等复杂理论指标的代理。本文指出计算NTK的高昂成本及其在现代DNN上的假设失效问题，强调了L-SWAG在理论严谨性和计算效率上的平衡。
5.  **AZ-NAS [23]**：同样主张使用代理集成并包含ViT搜索空间。但本文认为AZ-NAS将代理直接集成到NAS搜索中的评估方式未能充分剥离搜索算法本身的影响，且其ViT空间性能普遍较好难以区分代理能力；本文通过在2000个已训练ViT上评估代理与验证准确率的 correlation 来进行更公平的基准测试。

## 局限性与未来方向
1.  **LIBRA的评估限制**：目前LIBRA算法的有效性主要基于经验分析，缺乏更深入的理论保证。
2.  **应用领域扩展**：当前L-SWAG主要在静态图像识别任务上验证，未来可扩展到视频理解（Video Understanding）等不同输入模态的任务。
3.  **ViT空间的复杂性**：尽管L-SWAG在ViT上表现优异，但在TransNAS-Bench-101 Macro Autoencoder等极少数复杂任务上，仍略逊于Fisher等特定指标。

## 研究启发与可借鉴点
1.  **多源指标融合思路**：LIBRA-NAS提供的“信息增益最小化+偏差匹配”的代理组合策略，为如何无训练代价地融合互补的模型属性指标提供了可借鉴的范式，可迁移到其他需要多指标评估的领域。
2.  **激活函数适配**：论文展示了如何将主要针对ReLU设计的ZC代理（如SWAP）适配到GeLU网络（ViT常用），这为推广现有方法到新架构提供了具体实践参考。
3.  **层统计重要性分析**：通过绘制梯度统计量的分位数分布来识别高信息量层（Fig 2b），这种可视化分析方法可用于指导其他基于梯度的代理设计，优化计算成本。
4.  **统一基准的重要性**：强调了在统一实验设置（特别是跨ConvNet和Transformer）下评估代理的必要性，为后续研究确立了更严格的对比标准。

## 关键术语表
*   **Zero-Shot NAS (ZC-NAS)**：零成本神经架构搜索，通过计算无需训练的代理指标来预估架构性能，从而避免昂贵的训练过程。
*   **Proxy Metric (代理指标)**：用于快速评估神经网络性能的单次前向传播（或少量计算）指标，如梯度范数、Fisher信息、线性区域数等。
*   **Spearman $\rho$**：斯皮尔曼秩相关系数，用于衡量代理指标排序与验证准确率排序之间单调关系的相关性，值越接近1表示代理 ranking 越准确。
*   **Autoformer**：一个广泛使用的Vision Transformer搜索空间，包含多种不同深度和宽度的子架构变体。
*   **Information Gain (信息增益)**：在LIBRA中，用于衡量在选择一个代理后，能减少关于验证准确率的不确定性（熵）的程度。
*   **Bias (偏差)**：在此处指代理指标或验证准确率可能与某些网络结构性特征（如参数量）存在系统性关联，LIBRA通过匹配偏差来优化组合效果。

## 可复现要素
*   **数据集**：NASBench-201, NASBench-101, NASBench-301, TransNAS-Bench-101 (已公开)；Autoformer ViT搜索空间及2000个训练好的模型（论文提供了训练细节，可复现）。
*   **代码**：基于开源的 **NASBench-SuiteZero** 框架开发（论文未提供独立开源代码链接，但声明代码基于该开源项目）。
*   **关键超参**：batch size 通常为64（TransNAS-Bench-101为32）；层选择基于10个分位数 bin (PERC_BINS=10)；LIBRA中选择代理时相关性阈值范围为 $\rho_{best} - 0.1 < \rho_h \leq \rho_{best}$。
