---
title: "LMO-Linear-Mamba-Operator-for-MRI-Reconstruction"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_LMO_Linear_Mamba_Operator_for_MRI_Reconstruction_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:23:12"
field: "Medical Image Analysis / Compressed Sensing MRI"
keywords: ["MRI Reconstruction", "Neural Operator", "Mamba", "Deep Learning", "Accelerated MRI", "Functional Space Mapping"]
innovations: ["Proposed Linear Mamba Operator (LMO), the first Mamba-based Neural Operator for MRI reconstruction, combining Scanning Integration (SI) and Convolution Integration (CI) to capture global and local features.", "Proved Continuous-Discrete Equivalence (CDE) for SI and CI in bandlimited function spaces, theoretically ensuring the model's generalization ability.", "LMO demonstrated superior generalization across varying acceleration rates (AR), being the only model that achieves retraining performance without retraining when AR changes."]
benchmarks: ["IXI dataset", "fastMRI dataset (knee)", "Multi-coil brain dataset"]
---

# 论文速读：LMO-Linear-Mamba-Operator-for-MRI-Reconstruction

## 一句话总结
本文提出了线性 Mamba 算子（LMO），首次将基于 Mamba 的神经算子应用于 MRI 重建，通过融合扫描积分与卷积积分在带限函数空间中进行映射，显著提升了重建质量、可解释性及跨加速率的泛化能力。

## 研究问题与动机
- **可解释性与一致性不足**：现有深度学习 MRI 重建方法（如 CNN、Transformer）虽性能优异，但多为黑盒，缺乏可解释性；而深度展开网络（DUNs）虽具可解释性，但在信号空间操作时难以捕捉复杂信号的内在特征，且产生的解剖结构一致性较差。
- **泛化能力薄弱**：现有方法在处理分布外情况（如加速率 AR 变化）时，泛化性能往往灾难性下降，缺乏应对不同采样约束的能力。
- **计算复杂度与全局建模的矛盾**：神经算子（NOs）虽能学习函数空间映射并具备高泛化潜力，但基于傅里叶或 Transformer 的 NO 计算复杂度较高（$O(n \log n)$ 或 $O(n^2)$）；而基于卷积的 CNO 虽为 $O(n)$，但主要关注局部特征。
- **Mamba 在算子学习中的缺失**：Mamba 等状态空间模型兼具 $O(n)$ 复杂度和长程依赖建模优势，但因其缺乏适合算子学习的积分形式，尚未被有效引入 MRI 重建领域。

## 核心贡献（创新点）
- **提出扫描积分（SI）与卷积积分（CI）**：设计了两种新的积分形式，分别捕捉信号的全局长程依赖和局部特征，并证明了其在带限函数空间中的连续-离散等价性（CDE）。
- **构建 Linear Mamba Operator（LMO）架构**：将 SI 和 CI 集成到神经算子框架中，使网络映射完全在带限函数空间进行，从而获得超越实例级映射的优越泛化能力。
- **理论严谨与高可解释性**：所有计算根植于扎实的数学推导，使得模型不仅性能领先，还具备良好的理论可解释性。
- **卓越的性能与效率**：在单线圈和多线圈 MRI 重建任务上，LMO 在 PSNR 和 SSIM 指标上均显著优于现有 SOTA 方法，同时保持 $O(n)$ 计算复杂度。
- **无需重训练的泛化能力**：LMO 是唯一能够在加速率（AR）改变时，无需额外重训练步骤即可达到重训练性能的模型。

## 方法详解
- **问题设定与带限函数空间**：将 MRI 重建视为学习从欠采样测量 $\mathcal{V}$ 到目标图像 $\mathcal{X}$ 的映射 $\mathcal{G}^\dagger: \mathcal{A} \to \mathcal{U}$，其中 $\mathcal{A}, \mathcal{U}$ 为索博列夫空间。为便于实现连续-离散等价，选择带限函数空间 $\mathcal{B}_w(\Omega)$ 作为工作空间，其函数的傅里叶变换支撑集限于 $[-w, w]^2$。
- **神经算子架构**：LMO 采用组合映射 $\mathcal{G}_\theta := \mathcal{Q} \circ \eta_{t-1}(W_{t-1} + K_{t-1}) \circ \cdots \circ \eta_0(W_0 + K_0) \circ \mathcal{P}$，其中 $\mathcal{P}, \mathcal{Q}$ 为提升/投影映射，$W_t$ 为线性算子，$\eta_t$ 为激活函数，$K_t$ 为核积分算子。
- **扫描积分（SI）**：借鉴 Mamba 思想，将核积分区间设为 $(-\infty, x)$。通过假设核函数 $K_t(x, y) = Ce^{Ax} \cdot Be^{-Ay}$，推导出状态空间模型（SSM）的离散化形式：$h(x_k) = \bar{\mathbf{A}}h(x_{k-1}) + \bar{\mathbf{B}}v(x_k)$ 和 $v'(x_k) = Ch(x_k)$。通过在四个方向（行、列及其反方向）扫描并合并，实现全局特征捕获，复杂度 $O(n)$。
- **卷积积分（CI）**：在物理空间定义卷积积分 $(K(v))(x) = \int_\Omega K(x-y)v(y)dy \approx \sum_{i,j=1}^k \kappa_{ij}v(x-y_{ij})$。该局部卷积操作在带限函数空间内封闭，不会破坏函数带限性质，避免了传统卷积在高维分辨率下的退化问题。
- **连续-离散等价（CDE）**：强调架构设计遵循 CDE 原则，确保连续算子的离散表示与其原始算子等价，这是模型泛化性的理论基础。

## 实验与结果
- **数据集**：IXI 数据集（578 张 T2 图像，单线圈）、knee fastMRI 数据集（单线圈）、以及包含 5 名志愿者多线圈脑部图像的数据集（多线圈）。
- **基线方法**：HQS-Net, H-DSLR, PGIUN（深度展开）；Unet, SwinIR, U-Mamba（纯深度学习）。
- **单线圈 MRI 重建**：在 IXI 数据集径向掩码 ×4 加速下，LMO 取得 PSNR 48.16dB，显著领先。在参数相近的对比实验中，LMO 较 PGIUN, H-DSLR, HQS-Net 分别提升 0.83dB, 1.97dB, 9.44dB。
- **多线圈 MRI 重建**：在 ×6 加速率的随机掩码下，LMO 同样取得最佳性能。
- **泛化能力分析**：在尺度泛化（SG）实验中，LMO 展现了卓越的跨加速率泛化能力。例如，在 ×8 训练、×4 测试的情况下，LMO 性能依然优异，而其他方法性能大幅下降。LMO 是唯一在 AR 改变时无需重训练即可达到重训练性能的模型。
- **消融实验**：纯 CI 或纯 SI 配置的性能分别下降 13.48dB 和 7.71dB，证实了结合全局与局部信息的重要性。

## 相关工作脉络
- **传统压缩感知 MRI（CS-MRI）**：使用手工设计的先验（如稀疏性、TV）辅助重建，泛化性差。
- **深度学习 MRI 重建**：CNNs (Unet), Transformers (SwinIR), Mamba (U-Mamba) 等通用架构直接应用，缺乏可解释性且非专为 MRI 设计。
- **深度展开网络（DUNs）**：如 HQS-Net, PGIUN, H-DSLR，将优化过程展开为网络层，具可解释性，但主要在信号空间操作，泛化能力有限。
- **神经算子（NOs）**：FNO (傅里叶神经算子，$O(n \log n)$), CNO (卷积神经算子，$O(n)$，局部特征), Oformer (Transformer 基，$O(n^2)$)。LMO 填补了将高效 NO 应用于 MRI 加速的空白，并引入 Mamba 实现 $O(n)$ 全局建模。

## 局限性与未来方向
- **局限性**：论文未明确自述模型局限性。从方法层面推断，可能包括多方向扫描实现的复杂性以及对特定数据分布的潜在依赖。
- **未来方向**：探索 LMO 在其他医学成像模态（如 CT, PET）上的适用性，以扩大其在医学领域的影响。

## 研究启发与可借鉴点
- **算子学习视角**：将 MRI 重建建模为函数空间间的映射而非实例映射，为提升模型泛化性提供了新的理论框架和设计思路。
- **SI 与 CI 的融合**：结合扫描（全局）和卷积（局部）两种积分形式以更全面地提取特征，这一策略可迁移至其他需要同时捕捉长程依赖和局部细节的视觉任务。
- **CDE 原则的应用**：在设计用于连续函数学习的神经网络时，保证离散实现与连续算子的等价性是确保泛化能力的关键设计原则。
- **Mamba 与算子学习的结合**：将 Mamba 的状态空间模型与积分算子相结合，实现了高效的 $O(n)$ 全局特征提取，为处理高分辨率图像或序列数据提供了新范式。
- **零重训练的泛化能力**：证明了通过函数空间映射学习，模型可泛化至不同加速率，这对实际临床中多变采样条件的适应性具有重要价值。

## 关键术语表
- **MRI Reconstruction**：从欠采样的 k-space 数据恢复高质量磁共振图像的过程。
- **Neural Operator (NO)**：学习算子（函数空间之间映射）的深度学习模型，如 FNO, CNO。
- **Deep Unrolling Network (DUN)**：将迭代优化算法的每一步展开为网络层，以提高模型可解释性的方法。
- **Scanning Integration (SI)**：LMO 提出的基于 Mamba 状态的积分形式，用于捕捉信号的全局长程依赖。
- **Convolution Integration (CI)**：LMO 提出的在物理空间进行局部卷积的积分形式，以捕捉信号的局部特征。
- **Bandlimited Function Space**：函数的傅里叶变换在某一频率范围外为零的函数空间，MRI 信号可视为此类空间的元素。
- **Continuous-Discrete Equivalence (CDE)**：连续算子的离散数值实现与其原始连续算子在数学上的等价性，是保证算子泛化能力的关键。
- **Acceleration Rate (AR)**：指 MRI 采样率相对于奈奎斯特采样率的降低倍数，如 ×4, ×8。

## 可复现要素
- **数据集**：IXI 数据集、knee fastMRI 数据集（公开）；多线圈脑部数据集（论文未指明具体来源，但包含五个志愿者的多线圈图像）。
- **代码**：已开源，链接为 https://github.com/ZhengJianwei2/LMO。
- **权重**：论文未提及。
- **关键超参**：论文主体未详细列出，提及实验设置遵循原始作者的建议；网络深度、隐藏维度、学习率等可能在附录或补充材料中。
