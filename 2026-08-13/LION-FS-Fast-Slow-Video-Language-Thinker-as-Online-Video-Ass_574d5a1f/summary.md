---
title: "LION-FS-Fast-Slow-Video-Language-Thinker-as-Online-Video-Ass"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_LION-FS_Fast__Slow_Video-Language_Thinker_as_Online_Video_Assistant_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:22:43"
field: "在线视频理解与对话"
keywords: ["在线视频对话", "第一人称视频理解", "快慢思考机制", "多模态大语言模型", "流式视频处理", "特征路由"]
innovations: ["提出快慢双路径视频语言思考者框架，解耦响应判定与生成", "设计Token Aggregation/Dropping Router实现自适应特征融合与冗余裁剪", "开发训练无关的多粒度关键帧增强策略提升响应精度"]
benchmarks: ["Ego4D Narration Stream Benchmark", "Ego-Exo4D Benchmark"]
---

# 论文速读：LION-FS-Fast-Slow-Video-Language-Thinker-as-Online-Video-Ass

## 一句话总结
本文提出LION-FS，一个模拟人类"快慢思考"认知过程的在线视频助手框架，通过Fast Path的路由化特征聚合与裁剪实现高效实时的响应判定，再通过Slow Path的多粒度关键帧增强实现精确的响应生成，在Ego4D和Ego-Exo4D数据集上同时达到最优的准确性和效率。

## 研究问题与动机
- **响应判定准确性不足**：现有方法（如VideoLLM-online/LIVE）仅处理低帧率视频并使用粗粒度图像token，难以有效捕捉帧间时序关系和第一人称场景特有的视角/交互信息。
- **响应生成精度有限**：对所有视频帧固定保留有限token数量，无法自适应提取第一人称视频中的细粒度时空细节和人机交互特征。
- **训练与推理效率低下**：LIVE试图通过扩展所有帧的token来提升性能，但在简单的响应判定阶段无需大量token扩展，而在关键帧生成阶段却token不足，造成资源浪费与性能瓶颈。
- **第一人称与第三人称视觉特征不匹配**：通用图像编码器（如SigLIP）在第三人称图像上预训练，难以理解第一人称视角场景，需引入专用第一人称编码器补充时序信息。

## 核心贡献（创新点）
- **提出"快慢双路径"视频语言思考者框架**：将响应判定（Fast Path）与响应生成（Slow Path）解耦，模拟人类直觉式快速判断与深思熟虑式详细生成的认知过程，与已有工作（如LIVE的单路径处理）本质不同。
- **设计Token Aggregation Router实现自适应特征融合**：以SigLIP的CLS token作为视觉引导（Visual Guidance），动态加权聚合第三人称通用空间特征与第一人称密集时序特征，在不增加token数量的前提下实现双编码器互补，区别于简单的拼接或相加策略。
- **提出Token Dropping Router进行冗余特征动态裁剪**：基于可学习的路由权重在每个Transformer层自适应丢弃冗余token（按β分位数阈值筛选），在几乎不损失性能的同时降低计算开销，与固定裁剪或随机裁剪方法有本质区别。
- **开发训练无关的多粒度关键帧增强策略**：对判定为需要响应的关键帧，分别进行全局均匀网格增强（Grid Tokens，4×3×3池化）和局部自适应手物交互区域增强（Box Tokens），并通过Multimodal Thinking Template注入LLM，突破训练数据中原子动作描述的局限。

## 方法详解
**整体框架**：LION-FS将在线视频对话分为Fast Path（响应判定）和Slow Path（响应生成）两条路径，采用流式损失与标准语言建模损失联合训练。

**损失函数**：
$$\text{Loss} = \frac{1}{N} \sum_{j=1}^{N} \left( -w s_j \log P_j^{\text{[EOS]}} - l_{j+1} \log P_j^{\text{[Txt]}_{j+1}} \right)$$
其中$P_j^{\text{[EOS]}}$为LLM预测EOS标记的概率（流式损失），$P_j^{\text{[Txt]}_{j+1}}$为自回归预测下一个文本token的概率（LM损失），$s_j$和$l_{j+1}$为二值条件系数。

**Fast Path双编码与Token Aggregation Router**：
- 使用$E_{gen}$（SigLIP）处理2 FPS视频帧，提取通用空间特征；使用$E_{ego}$（EgoVLPv2）处理8 FPS视频帧（每4帧一组），提取第一人称时序特征。
- 每个0.5秒片段编码为10个token（含1个CLS token和3×3池化token）。
- 以SigLIP的CLS token作为Visual Guidance [VG]，通过路由函数$G_f$生成自适应聚合权重：
$$[\text{Frm}]_i = G_f([\text{VG}])_0 \times [\text{Frm}_s]_i + G_f([\text{VG}])_1 \times [\text{Frm}_t]_i$$
实现without increasing token numbers的特征融合。

**Fast Path Token Dropping Router**：
- 对每个token生成路由权重标量$r_{(i,n)}^l = w_\theta^T [\text{Frm}]_{(i,n)}^l$，仅保留权重超过β分位数阈值$P_\beta^l$的top-k token参与当前层计算。
- 采用Interleaved Layers配置（每隔一层插入dropping router），配合$\beta=0.5$的丢弃率，平衡性能与效率。

**Slow Path多粒度关键帧增强**：
- **Grid Tokens（全局均匀增强）**：将关键帧划分为4个均匀网格，每个网格执行3×3池化，形成4×3×3的fine-grained spatial tokens。
- **Box Tokens（局部自适应增强）**：使用Faster R-CNN检测手部位置，通过优化手-物anchor box距离和NMS获取交互对象边界框，从原始576个patch token中提取对应区域token并进行全局池化。
- **Multimodal Thinking Template**：将增强后的token整合为提示模板"Stream: [Frame Tokens] [Grid Tokens] User: Please focus on [Box Tokens]. Assistant: "，引导更精确的响应生成。

## 实验与结果
**数据集**：Ego4D Narration Stream Benchmark和Ego-Exo4D Benchmark（第一人称视频对话数据集）。

**评估指标**：LL-PPL（语言建模困惑度）、TimeDiff（时间对齐误差）、Fluency（流畅度）、LM-Correctness（正确率）。

**主要结果**（Table 1）：
- **Ego-Exo4D Narration Validation**：LION-FS取得LL-PPL=2.04、TimeDiff=0.74、Fluency=36.5%、LM-Correctness=48.2%，均优于VideoLLM-online（2.24/0.78/33.7%/44.8%）和VideoLLM-MoD（2.12/0.82/33.8%/45.3%）。
- **Ego4D Narration Validation**：LION-FS取得LL-PPL=2.09、TimeDiff=2.15、Fluency=46.1%、LM-Correctness=52.4%，在Fluency和LM-Correctness上优于基线。
- **效率提升**：LION-FS处理视频流的帧率为现有方法的**4倍**。

**消融实验关键结论**：
- Token Aggregation Router的Adaptive Routing策略在Ego-Exo4D上取得最佳Fluency=38.1%和LM-Correctness=48.0%（Table 2）。
- Token Dropping Router在Interleaved Layers + $\beta=0.5$配置下取得LL-PPL=2.16、Fluency=36.5%、LM-Correctness=47.0%，同时FLOPs降至51.40T（Table 3）。
- 多粒度增强中Grid Tokens (4×3×3) + Box Tokens组合取得LL-PPL=2.04、LM-Correctness=48.2%的最优结果（Table 5）。

## 相关工作脉络
- **VideoLLM-online/LIVE [5]**：开创性地将视频流对话引入在线视频助手框架，但仅使用低帧率粗粒度图像特征，LION-FS通过Fast/Slow双路径和双编码器设计显著超越其准确性和效率。
- **VideoLLM-MoD [59]**：采用Mixture-of-Depths机制进行视频token深度路由，但同样受限于单一路径处理范式；LION-FS在此基础上引入快慢分离思想，将计算资源分配至关键帧增强而非全程冗余处理。
- **SlowFast-LLaVA [62]**：在视频不同帧率下进行多粒度pooling以丰富特征，但属于离线视频理解任务；LION-FS将快慢机制迁移至在线流式对话场景，并引入第一人称专用编码器和路由机制。
- **FaST [52]**：为复杂视觉查询选择性激活"System 2"，但面向静态图像理解；LION-FS将其思想应用于视频流，通过Fast Path实时判定何时需要Slow Path的详细生成。
- **SLOWFAST-VGEN [17]**：利用慢快学习循环进行动作驱动的长视频生成，与LION-FS的快慢分离在目的（生成vs理解/对话）和实现方式上存在本质差异。

## 局限性与未来方向
- **依赖预训练编码器能力**：Fast Path的效果部分依赖于SigLIP和EgoVLPv2的预训练质量，若第一人称视频分布偏移较大，$E_{ego}$可能无法充分捕捉新颖交互模式。
- **Slow Path为训练无关方法**：多粒度关键帧增强采用固定的池化和检测策略，未进行端到端可微优化，可能限制了与LLM的协同调优空间。
- **手部/物体检测的延迟**：Faster R-CNN的实时性可能成为在线场景的瓶颈，尤其在低算力设备上部署时。
- **未来方向**：可扩展至多模态交互（音频、触觉等）；探索Slow Path的可微增强策略；研究更高效的端侧部署方案以适应智能眼镜等可穿戴设备。

## 研究启发与可借鉴点
- **快慢分离的认知架构**：将人类"直觉快速判断+深思详细生成"的双系统思维迁移至AI视频理解，为流式多模态任务提供了高效的计算资源分配范式，可借鉴至实时语音助手、自动驾驶感知等场景。
- **路由机制平衡性能与效率**：Token Aggregation和Dropping Router通过轻量级门控实现特征自适应选择，避免暴力拼接带来的计算爆炸，该思路可扩展至长上下文推理、多传感器融合等领域。
- **训练无关的关键帧增强**：对判定为关键的帧施加额外视觉增强（无需重新训练模型），以极低成本换取响应质量的显著提升，体现了"按需计算"的设计哲学，值得在资源受限的在线系统中推广。
- **第一人称与第三人称特征的协同**：将通用视觉知识库与领域专属编码器通过路由融合，既保留了泛化能力又弥补了领域偏差，该策略可迁移至其他视角/domain adaptation任务。

## 关键术语表
- **LION-FS**：Fast & Slow Video-Language Thinker as onLIne videO assistaNt，本文提出的在线视频助手框架，通过快慢双路径实现实时、主动、时间精确的对话响应。
- **Fast Path**：负责实时响应判定的快速路径，通过Token Aggregation和Dropping Router高效处理高分辨率视频流，判断是否需要立即响应。
- **Slow Path**：负责响应生成的慢速路径，对关键帧进行多粒度增强（Grid/Box Tokens），引导LLM生成更精确、细粒度的回复。
- **Token Aggregation Router**：以视觉引导（Visual Guidance）生成自适应权重，融合第三人称通用特征与第一人称时序特征的路由模块，保持token数量不变。
- **Token Dropping Router**：根据可学习的路由权重在Transformer层中动态丢弃冗余token的模块，降低计算开销。
- **Grid Tokens**：对关键帧进行全局均匀网格划分后的细粒度空间token（4×3×3池化），捕获全局细节。
- **Box Tokens**：基于手部-物体交互检测提取的局部自适应token，聚焦人-环境交互区域。
- **Multimodal Thinking Template**：将增强后的多粒度token整合成的多模态提示模板，引导LLM生成更精确的响应。

## 可复现要素
- **数据集**：Ego4D Narration Stream Benchmark和Ego-Exo4D Benchmark（公开数据集）
- **代码开源**：https://github.com/JiuTian-VL/LION-FS
- **关键超参**：
  - 帧率：2 FPS（SigLIP）、8 FPS（EgoVLPv2，每4帧一组）
  - 丢弃率：$\beta = 0.5$
  - Dropping配置：Interleaved Layers（每隔一层）
  - Grid增强：4×3×3池化
  - 损失权重：$w=1$
- **硬件**：80G A800 GPU
