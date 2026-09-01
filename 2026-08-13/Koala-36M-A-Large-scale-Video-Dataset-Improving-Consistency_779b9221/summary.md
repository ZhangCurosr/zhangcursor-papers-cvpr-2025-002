---
title: "Koala-36M-A-Large-scale-Video-Dataset-Improving-Consistency"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Koala-36M_A_Large-scale_Video_Dataset_Improving_Consistency_between_Fine-grained_Conditions_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:22:39"
field: "视频生成数据集与数据工程"
keywords: ["video dataset", "text-to-video generation", "data curation", "video splitting", "structured captioning", "dataset filtering", "metric conditioning"]
innovations: ["提出 VTSS 单一综合评分替代多阈值视频过滤策略", "设计结构化六维度字幕系统与 CSS+3σ 时序过渡检测", "通过 AdaLN 注入多指标完成细粒度质量条件控制"]
benchmarks: ["VBench"]
---

# 论文速读：Koala-36M: A Large-scale Video Dataset Improving Consistency between Fine-grained Conditions and Video Content

## 一句话总结
本文提出 Koala-36M，一个包含 3600 万高质量视频片段的文本-视频数据集，通过更精确的视频分割、平均 200+ 词的细粒度结构化字幕、基于 VTSS 的智能数据过滤以及 Metric Conditioning 训练策略，显著提升细粒度条件与视频内容之间的一致性，其 VBench 总得分较 Panda-70M 提升约 0.097。

## 研究问题与动机
1. **字幕粒度不足**：现有大规模数据集（如 Panda-70M）的字幕过于简略，且视频中存在频繁场景切换，导致文本-视频语义对齐困难，影响生成模型训练。
2. **数据过滤方法粗糙**：传统方法依赖多个独立阈值手动筛选，忽略了多指标间的联合分布，导致低质量视频未被有效剔除，同时误删高质量数据。
3. **训练数据质量异质性**：即使经过过滤，视频间质量差异仍较大，模型难以区分高质量与低质量数据，限制了生成可控性。

## 核心贡献（创新点）
1. **提出 Koala-36M 数据集**：首个同时具备千万级规模（3600万视频）与高分粒度字幕（平均 202 词）的大规模视频数据集，相比 Panda-70M 在各项 dataset metric 上均有显著提升。
2. **引入视频过渡检测改进方法**：提出基于 BGR 直方图相关性与 Canny SSIM 的 Color-Struct SVM（CSS）模块，并结合 3σ 时序平滑，有效区分缓变过渡与快动作场景，减少语义不一致。
3. **设计结构化字幕生成系统**：从主体、动作、环境、视觉语言、摄像机语言、世界知识六个维度分别生成并合并字幕，平均长度达 200+ 词，显著提升文本-视频对齐质量。
4. **提出 VTSS（Video Training Suitability Score）**：构建 Training Suitability Assessment Network（TSA），以动态分支（3D Swin Transformer）、静态分支（ConvNext）和标签分支（多指标）通过 Weight Cross-Gating Block（WCGB）融合，输出单一综合评分，替代传统多阈值过滤。
5. **引入 Metric Conditioning 训练策略**：将运动分数、美学分数、清晰度等指标编码为频率嵌入，通过 MLP 后以 AdaLN 方式注入 timestep 嵌入，使生成模型能感知并区分不同质量级别数据，在推理阶段支持细粒度控制。

## 方法详解
- **视频分割（4.1）**：以 BGR 直方图相关性 $d_{color}$ 和 Canny Luminance SSIM 结构化距离 $d_{struct}$ 作为特征，训练线性 SVM 分类器检测帧间过渡；结合时序建模，假设变化服从高斯分布，当前帧变化超出 3σ 置信区间则判定为显著过渡，从而区分缓变与快动场景。
- **结构化字幕（4.2）**：基于 GPT-4V 生成初始字幕，再在 LLaVA 基础上微调 caption 模型；使用高分辨率视觉编码器（对 token 做 2×2 平均池化降计算量），并采用静态图像与动态视频混合训练策略。
- **VTSS 网络（4.3）**：
  - 标注标准涵盖 **动态质量**（运动区域 >30% 且运动稳定）、**静态质量**（构图、美学、色彩）与 **视频自然性**（无特效/字幕/Logo，内容安全）。
  - 由 8 位专家独立标注并进行偏差消除（Appendix B）。
  - 网络结构：动态分支用 3D Swin Transformer，静态分支用 ConvNext，标签分支提供多指标特征；通过 WCGB 学习融合权重，将标签特征注入动态与静态分支。
  - 最终 VTSS 分布近似双高斯，取阈值 2.5 过滤得到 3600 万片段。
- **Metric Conditioning（4.4）**：将 motion score、aesthetic score、clarity score 等指标编码为频率嵌入，经 MLP 投影后直接加到 timestep embedding，并通过 AdaLN 送入 transformer block；相比 OpenSora 等文本注入方式，支持更高精度的数值控制且各指标解耦更好。

## 实验与结果
- **训练设置**：基于 Sora-like 3D-full attention transformer 架构，T5 文本嵌入 + 3D causal VAE，2s 时长、256×256 分辨率，80G A100 × 8，batch size 32，lr=1e-4，共 140M 样本。
- **评测基准**：VBench [13]，并对 prompt 做扩展以适应领域差异。
- **核心结果（Table 2）**：
  - **Koala-36M (condition)** 总得分 **0.7460**，较 Panda-70M（0.6493）提升 **+0.0967**；语义分数从 0.3093 提升至 0.5915，质量分数从 0.7343 提升至 0.7846。
  - Aesthetic Quality 从 0.3988 → 0.5318，Object Class 从 0.3017 → 0.7794，Human Action 从 0.2400 → 0.8080，Color 从 0.5942 → 0.8960。
  - 消融表明：重分割+重字幕（Koala-w/o TSA）已优于 Panda-70M；VTSS 过滤（Koala-36M vs Koala-37M-manual）保留更多高质量数据；Metric Conditioning 带来最大单项增益（Total 0.7156 → 0.7460）。

## 相关工作脉络
1. **Panda-70M**：当前最大公开视频-文本数据集，但其字幕简略（平均 13.2 词），且未经细粒度过渡检测与智能过滤；Koala-36M 与之同源但质量显著更高。
2. **MiraData / VidGen**：采用结构化字幕，但规模远小于 Koala-36M，且未引入 VTSS 过滤与 Metric Conditioning。
3. **HD-VILA-100M / HowTo100M**：规模大但字幕依赖 ASR，质量较低；Koala-36M 强调字幕深度与细粒度对齐。
4. **Stable Video Diffusion**：提供完整数据处理流程但数据集不开源；本文开源数据集与代码，推动可复现研究。
5. **传统视频质量评估模型**：聚焦美学与技术指标，而 VTSS 关注"训练适用性"，引入动态/静态联合建模与专家标注对齐。

## 局限性与未来方向
1. **规模仍有限**：作者自述 Koala-36M 尚不足以支持 1B+ 参数视频生成模型的训练，需进一步扩展数据规模。
2. **VTSS 阈值依赖**：当前取固定阈值 2.5，未探索自适应或任务相关的最优截断策略。
3. **Metric Conditioning 的推理控制**：需人工设定各指标分数，尚未探索自动最优配置策略。
4. **高分辨率/长视频支持**：当前实验仅 256×256 / 2s，未验证在更高分辨率下的泛化性。

## 研究启发与可借鉴点
1. **视频过渡检测的时序建模**：CSS + 3σ 高斯置信区间方法可迁移至其他视频分割任务，避免仅依赖帧间特征差的阈值启发式方法。
2. **多指标联合建模替代多阈值**：VTSS 思路（将多维度质量评估压缩为单一可学习分数）适用于任何需要综合排序/过滤的视频或图文数据集。
3. **Metric Conditioning 的可迁移性**：通过 AdaLN 注入数值指标的方法可推广至图像生成、音频生成等需要细粒度质量控制的模态。
4. **结构化字幕的系统设计**：六维度拆分生成策略可复用于其他多模态对齐任务，提升 LMM 的数据理解深度。
5. **混合静态/视频微调策略**：同时训练图像与视频有助于缓解视频数据不足问题，可借鉴于视频理解模型的预训练/微调。

## 关键术语表
**Koala-36M**：本文提出的 3600 万条高质量视频-文本对数据集，平均字幕 202 词，分辨率 720p。
**VTSS（Video Training Suitability Score）**：由 TSA 网络输出的单一综合评分，用于替代传统多阈值过滤策略。
**TSA（Training Suitability Assessment Network）**：含动态分支（3D Swin）、静态分支（ConvNext）与标签分支的多流融合网络。
**WCGB（Weight Cross-Gating Block）**：学习融合权重以将标签特征注入动态/静态分支的模块。
**Metric Conditioning**：将数据质量指标编码为嵌入并通过 AdaLN 注入扩散模型训练/推理的细粒度控制方法。
**CSS（Color-Struct SVM）**：基于 BGR 直方图相关性与 Canny SSIM 的帧间过渡检测分类器。
**VBench**：视频生成模型的综合性评测基准，涵盖质量、语义、一致性等多维度指标。
**AdaLN（Adaptive Layer Normalization）**：将条件嵌入自适应调节层归一化参数，用于注入 quality metrics。

## 可复现要素
- **数据集**：Koala-36M 已开源，地址 https://koala36m.github.io/
- **代码**：已开源（论文声明）
- **关键超参**：batch size=32，lr=1e-4，视频时长 2s，分辨率 256×256，VTSS 阈值=2.5；训练总量 140M 样本
- **模型架构**：Sora-like 3D-full attention transformer，T5 text encoder，3D causal VAE
- **caption 模型基座**：LLaVA + GPT-4V 数据生成
- **评估工具**：VBench
