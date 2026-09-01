---
title: "LSNet-See-Large-Focus-Small"
source: https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_LSNet_See_Large_Focus_Small_CVPR_2025_paper.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 20:23:18"
field: "高效计算机视觉"
keywords: ["轻量级视觉网络", "大核卷积", "动态卷积", "token mixing", "LSNet", "异构尺度感知"]
innovations: ["提出感知-聚合解耦的LS卷积，大核静态感知+小核动态聚合", "基于人类视觉'See Large, Focus Small'机制设计轻量级LSNet模型族", "LS卷积可泛化替换ResNet/DeiT中的标准卷积与自注意力并显著提升精度"]
benchmarks: ["ImageNet-1K", "COCO-2017", "ADE20K", "ImageNet-C", "ImageNet-A", "ImageNet-R", "ImageNet-Sketch"]
---

# 论文速读：LSNet-See-Large-Focus-Small

## 一句话总结
本文受人类视觉系统"周边视野大范围感知 + 中央凹小范围聚焦"机制启发，提出 LS（Large-Small）卷积操作，通过大核静态卷积进行宽视野感知、小核动态卷积进行自适应精细聚合，构建了轻量级视觉网络 LSNet，在 ImageNet 分类及检测/分割下游任务上均实现 SOTA 性能与推理效率的最佳权衡。

## 研究问题与动机
- 现有轻量级 CNN/ViT 主要依赖自注意力或标准卷积做 token mixing，两者在感知与聚合过程中均存在效率-性能瓶颈。
- 自注意力感知与聚合共享同一全域上下文范围（同质尺度），扩大感受野会带来显著的二次方计算复杂度，且在低信息背景区域产生冗余聚合。
- 传统卷积的聚合权重由固定核参数决定，缺乏对不同 token 上下文邻域的敏感度，表达能力受限，难以在极低计算预算下充分利用模型容量。
- 需要在有限 FLOPs 约束下，探索一种既能扩展感知范围、又能自适应聚合的 token mixing 机制。

## 核心贡献（创新点）
- 提出"See Large, Focus Small"异构尺度 token mixing 策略，将感知（大范围）与聚合（小范围）解耦为两个独立步骤，区别于自注意力与卷积的同尺度设计。
- 设计 LS 卷积：先用大核静态深度可分离卷积建模宽领域空间关系生成上下文自适应权重，再用小核动态卷积（带分组机制）在高度相关邻域内融合特征，相比简单拼接大/小核卷积具有更强的判别力。
- 构建 LSNet 轻量级模型族（T/S/B，0.3G/0.5G/1.3G FLOPs），在 ImageNet 分类、COCO 检测/分割、ADE20K 语义分割及多项鲁棒性基准上均达到当前最优性能-效率权衡。
- 验证 LS 卷积的强泛化性：可直接替换 ResNet50 中的 3×3 卷积、DeiT-T 中的 self-attention，分别带来 +1.9% 和 +0.8% top-1 提升。

## 方法详解
- **整体公式框架**：对 token $x_i$，感知与聚合使用不同上下文区域 $\mathcal{N}_P(x_i)$ 与 $\mathcal{N}_A(x_i)$（前者远大于后者），输出 $y_i = \mathcal{A}(\mathcal{P}(x_i, \mathcal{N}_P(x_i)), \mathcal{N}_A(x_i))$。
- **Large-Kernel Perception（LKP）**：先经 PW 将通道减半至 $C/2$ 降成本；再用 $K_L \times K_L$ 大核 depth-wise convolution 捕获宽视野上下文；最后经 PW 生成上下文自适应权重 $w_i \in \mathbb{R}^D$，公式为 $w_i = \mathrm{PW}(\mathrm{DW}_{K_L \times K_L}(\mathrm{PW}(\mathcal{N}_{K_L}(x_i))))$。
- **Small-Kernel Aggregation（SKA）**：将特征图按 $G$ 组划分，同组通道共享聚合权重；把 $w_i$ 重塑为 $w_i^* \in \mathbb{R}^{G \times K_S \times K_S}$，用小核动态卷积对局部邻域 $\mathcal{N}_{K_S}(x_{ic})$ 做自适应加权融合，公式为 $y_{ic} = w_{ig}^* \circledast \mathcal{N}_{K_S}(x_{ic})$。
- **复杂度**：总计算量 $\mathcal{O}\left(\frac{HWC}{4}(3C + 2K_L^2 + (2G+4)K_S^2)\right)$，对输入分辨率呈线性复杂度。
- **LSBlock 结构**：LS 卷积 + 残差连接 + 额外 DW + SE 模块（引入局部归纳偏置）+ FFN（通道混合）。
- **LSNet 架构**：overlapping patch embedding；前 3 个阶段堆叠 LSBlock（分辨率依次为 $H/8 \times W/8$、$H/16 \times W/16$、$H/32 \times W/32$）；最后一阶段因分辨率较小改用 MSA 块捕获长程依赖；默认 $K_L=7$、$K_S=3$、$G=C/8$。

## 实验与结果
- **ImageNet-1K 分类**：LSNet-B（1.3G FLOPs）top-1 达 80.3%，超越 AFFNet 0.5% 且推理速度近 3 倍；LSNet-S（0.5G）达 77.8%，超越 UniRepLKNet-A（+0.8%）与 FasterNet-T1（+1.6%）；LSNet-T（0.3G）达 74.9%，超越 StarNet-S1（+1.4%）与 EfficientViT-M3（+1.5%）。蒸馏后 LSNet-B* 达 81.6%。
- **COCO 检测/分割**：RetinaNet 下 LSNet-B 相比 PoolFormer-S12 提升 3.0 AP、相比 PVT-Tiny 提升 2.5 AP；Mask R-CNN 下 LSNet-S 相比 SHViT-S3 提升 0.5 $\mathrm{AP}^b$ 与 2.5 $\mathrm{AP}^m$。
- **ADE20K 语义分割**：LSNet-B 达 43.0 mIoU，超越 SwiftFormer-L1（+1.6）与 FastViT-SA24（+2.0）。
- **鲁棒性（ImageNet-C/A/R/Sketch）**：LSNet-B 在 ImageNet-C 上 mCE 为 59.3，较 UniRepLKNet-A 降低 1.3；在 ImageNet-A/R/Sketch 上分别领先 1.2%/1.5%/1.5%。
- **消融**：去掉 LS 卷积降 2.3%；去掉 LKP 降 1.1%（$K_L=7$ 附近饱和）；去掉 SKA 降 1.5%（$K_S=3$ 最优）；$C/G=8$ 为精度-效率最佳平衡点；额外 DW/SE 分别贡献 +0.5% / +0.3%。

## 相关工作脉络
- **ConvNeXt / RepViT**：以高效标准卷积与重参数化为核心，侧重纯 CNN 路径的效率优化，未引入异构尺度感知-聚合解耦。
- **EfficientViT / EdgeNeXt**：通过级联分组注意力或分裂深度转置注意力融合 CNN-ViT 优势，但感知与聚合仍共享同一上下文范围。
- **FastViT / EfficientFormer**：利用结构重参数化与大核卷积提升 Hybrid ViT 效率，聚合阶段仍为静态权重，缺乏动态适配能力。
- **UniRepLKNet**：统一大核感知框架，但在小范围聚合环节仍依赖静态卷积，无法根据上下文自适应调整局部融合策略。
- **MobileViT / EdgeViT**：早期 CNN-ViT 拼接范式，在精度或延迟任一维度上未能同时逼近本文水平。

## 局限性与未来方向
- 未覆盖极端低算力场景（<0.1G FLOPs），Tiny 以下规模的缩放策略有待探索。
- 大核感知对极小目标的空间定位精度可能受限，缺乏针对密集预测任务的专门分析。
- 末阶段仍使用 MSA 块，未来可进一步用 LS 卷积替代以统一架构并降低最后一层的计算负担。
- 仅验证了图像分类与通用下游任务，视频、音频、点云等多模态泛化尚未系统评估。

## 研究启发与可借鉴点
- **"感知-聚合解耦"范式**可迁移至其他高效网络设计：将上下文建模与特征融合拆分为异尺度两步，既能扩展感受野又不引入全局复杂度。
- **组动态聚合机制**（共享权重分组 + 动态核生成）为低开销自适应卷积提供可复用模块，适用于检测/分割头部的局部特征精炼。
- **大核静态感知 + 小核动态聚合**的组合思路可与线性注意力、状态空间模型等新兴 token mixing 方案结合，进一步压缩延迟。
- 本文消融中 $K_L=7$ 饱和、$K_S=3$ 最优的规律，为后续轻量大核卷积设计提供了明确的超参先验。

## 关键术语表
- **LS 卷积（Large-Small Convolution）**：结合大核静态卷积进行宽视野感知、小核动态卷积进行自适应精细聚合的新型 token mixing 操作。
- **LKP（Large-Kernel Perception）**：基于大核 depth-wise 卷积 + 点积卷积生成上下文自适应权重的感知模块。
- **SKA（Small-Kernel Aggregation）**：基于分组动态卷积在小邻域内对特征进行自适应加权的聚合模块。
- **Token Mixing**：视觉网络中对空间 token 间信息进行交换与融合的核心操作，由感知与聚合两步构成。
- **Depth-wise Convolution**：逐通道独立执行的卷积操作，显著降低计算量与参数量。
- **Group Dynamic Convolution**：将通道划分为多组、每组共享动态生成的卷积核，以在可控开销下实现上下文自适应融合。
- **Overlapping Patch Embedding**：相邻 patch 存在重叠的图像分块嵌入方式，可保留更多边界上下文信息。

## 可复现要素
- **数据集**：ImageNet-1K（公开）、COCO-2017（公开）、ADE20K（公开）。
- **代码与权重**：代码已开源 https://github.com/jameslahm/lsnet，模型权重随代码提供。
- **关键超参**：$K_L=7$、$K_S=3$、$G=C/8$、默认 PW 降维至 $C/2$；分类训练 100 epochs，泛化实验训练 300 epochs；蒸馏教师为 RegNetY-16GF（82.9% top-1）。
