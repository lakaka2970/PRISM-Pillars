# PRISM-Pillars 创新评估与实验计划

<div align="center">

**客观创新度评估 · 性能预测 · 完整实验体系 · 算力环境手册**

版本 1.0 · 2026-08-06

</div>

本文档是项目推进的主参考：第一部分给出对 `paper_plans/` 创新方案的客观评估与性能预测；第二部分定义在远端算力设备上执行的完整实验体系；第三部分是算力环境与数据部署手册。验收阈值与退出标准以 `project_constraints.md` 为准，本文档与其冲突时以 `project_constraints.md` 为准。

---

## 目录

- [第一部分 创新评估与性能预测](#第一部分-创新评估与性能预测)
  - [0. 结论速览](#0-结论速览)
  - [1. 方案创新点梳理](#1-方案创新点梳理)
  - [2. 领域高水平论文的创新门槛（实证基准）](#2-领域高水平论文的创新门槛实证基准)
  - [3. 逐创新点客观评估](#3-逐创新点客观评估)
  - [4. 审稿风险点与应对](#4-审稿风险点与应对)
  - [5. 性能提升预测](#5-性能提升预测)
- [第二部分 实验体系](#第二部分-实验体系)
  - [6. 实验总览与依赖关系](#6-实验总览与依赖关系)
  - [7. P0：RadarPillars 严格复现](#7-p0radarpillars-严格复现)
  - [8. P1：时序基线判决性实验（全方案生死线）](#8-p1时序基线判决性实验全方案生死线)
  - [9. P2–P4：PRISM 三模块逐步验证](#9-p2p4prism-三模块逐步验证)
  - [10. P5–P7：借鉴模块验证（带退出标准）](#10-p5p7借鉴模块验证带退出标准)
  - [11. P8：联合微调与最终模型](#11-p8联合微调与最终模型)
  - [12. 主实验表设计（Table 1–6）](#12-主实验表设计table-16)
  - [13. 鲁棒性实验](#13-鲁棒性实验)
  - [14. 跨数据集实验](#14-跨数据集实验)
  - [15. 统计显著性协议](#15-统计显著性协议)
  - [16. 效率测量协议](#16-效率测量协议)
- [第三部分 算力环境与数据部署](#第三部分-算力环境与数据部署)
  - [17. 远端设备档案](#17-远端设备档案)
  - [18. SSH 接入](#18-ssh-接入)
  - [19. 代码同步状态](#19-代码同步状态)
  - [20. VoD 数据集：状态、结构与部署步骤](#20-vod-数据集状态结构与部署步骤)
  - [21. 环境部署步骤（命令级）](#21-环境部署步骤命令级)
  - [22. 执行排期与 GPU 预算](#22-执行排期与-gpu-预算)
  - [23. 风险清单与缓解](#23-风险清单与缓解)
  - [24. 附录：命令速查](#24-附录命令速查)

---

# 第一部分 创新评估与性能预测

## 0. 结论速览

| 维度 | 结论 |
|------|------|
| **创新程度** | 领域内**中上水平（约前 30–40%）**，达到 IROS / ITSC / IEEE RA-L / T-IV 及 SCI Q1–Q2 期刊门槛；不足以支撑 CVPR/NeurIPS 级主张 |
| **最强创新** | Doppler 各向异性概率证据（物理可观性 → 概率路由），领域内检测方向暂无先例 |
| **最大威胁** | HyperDet（2026, arXiv 2602.11554）已抢占"Doppler 引导回波纠正"叙事；SGE-Flow 占据确定性补偿基线位 |
| **VoD 预测（S 版）** | 49.5–51.0 mAP（基线 48.76），P(Δ≥1.0 且 p<0.05) ≈ 40–50% |
| **VoD 预测（C 版）** | 51.0–53.5 mAP；超越 MAFF-Net（54.6）概率 <10% |
| **真正机会** | 跨域相对 Drop 缩小（VoD→TJ4DRadSet、K-Radar 恶劣天气），比单域 mAP 更容易过显著性检验 |
| **生死线实验** | P1 单调链：naive < ego-motion < 确定性 < 各向同性 < 各向异性 |
| **投稿定位** | 首选 T-IV / RA-L / IROS；跨域结果强则冲 TITS |

---

## 1. 方案创新点梳理

按 `great_upgrade_3.md` + `project_constraints.md` 的最终口径，论文主张**三个原创贡献**，其余为工程组件：

| # | 模块 | 缩写 | 内容 | 定位 | 代码状态 |
|---|------|------|------|------|---------|
| 1 | Doppler 各向异性不确定管 + 概率 Pillar 路由 | DAUT + RAPR | 历史点 = 各向异性高斯证据（σt ≥ σr），可靠性加权软路由 | 核心创新一 | 已实现（`pcdet/models/radar_evidence/`） |
| 2 | 自监督时序证据可靠性 | STER | temporal-support 伪标签 + FocalBCE + ranking 损失 | 核心创新二 | 已实现（`temporal_reliability.py`） |
| 3 | 因果局部时序 Pillar 融合 | CRLF | Mahalanobis + 可靠性 + 证据量 + 时间衰减偏置，门控残差融合 | 核心创新三 | 已实现（`pcdet/models/temporal/`） |
| 4 | RepDWC 骨干 + Lite-MDFEN 颈部 + CenterHead | — | 借鉴 RadarNeXt | 工程实现（不作创新主张） | 部分实现（配置已备 `prism_pillars_rf_s.yaml`） |
| 5 | 雷达物理增强 + 一致性损失 | — | RCS/Doppler/丢点/ghost 增强 | 辅助训练策略 | 部分实现（ghost aug 工具已在 `loss_utils.py`） |

检测器 `PRISMPillarsRF` 已注册于 `pcdet/models/detectors/__init__.py`，**但尚未经过任何训练验证**。

---

## 2. 领域高水平论文的创新门槛（实证基准）

### 2.1 已发表工作的创新模式与收益

| 论文 | Venue | 创新模式 | 单模块消融增益 | VoD mAP |
|------|-------|---------|--------------|---------|
| RadarPillars | ITSC/IROS'24 | 3 个针对性小创新（速度分解、PillarAttention、uniform scaling） | +3.8 / +3.4 / +2.5 | 50.70 |
| MAFF-Net | RA-L'25 | 稀疏 pillar attention + Doppler 速度聚类查询 + 去噪 | — | 54.6（含 LiDAR 蒸馏，非公平对比） |
| SCKD | AAAI'25 | 单一机制（跨模态蒸馏）但增益巨大 | +10.4 | 52.08 |
| RadarGaussianDet3D | 2025 | 表示层替换（硬 pillar → 可微高斯）+ 新损失 | — | 52.0 |
| SMURF | T-IV'23 | 多表示融合（KDE 密度分支） | — | 50.97 |
| RadarNeXt | EURASIP JASP'25 | 效率导向架构（RepDWC + MDFEN），Pareto 主张 | DCNv3 位置 +2.33 | 50.48 |
| SGE-Flow | Sensors'26 | **确定性** Doppler 位移补偿 + 帧间 flow | — | — |
| HyperDet | 2026（arXiv 2602.11554） | 输入级增强：时空累积 + 跨传感器验证 + Doppler 引导补偿 + LiDAR 引导生成式增强 | 稳定提升 | — |

### 2.2 领域门槛画像（审稿人实际执行的标准）

1. **VoD 头部已饱和**：四年间 radar-only mAP 从 44.9（PointPillars）爬到 54.6（MAFF-Net，含蒸馏），单模块增益普遍收缩到 **+0.5 ~ +2**。
2. **显著性门槛抬升**：RadarPillar 复现仓库三 seed 标准差 ≈ 1 mAP。**+1 mAP 以下的提升难以通过显著性检验**，2026 年的审稿人开始默认要求 mean±std 与 bootstrap 检验。
3. **三类被认可的创新**：表示替换（高斯/KDE）、物理信息利用（Doppler 聚类、速度分解）、训练范式（蒸馏）。纯注意力变体已难以单独支撑论文。
4. **多数据集与鲁棒性**正在成为隐性要求（T-IV/RA-L 尤其）。

---

## 3. 逐创新点客观评估

### 3.1 创新一：Doppler 各向异性概率证据（DAUT + RAPR）——评级：中上

**物理前提是经典的，应用是新的。** 径向可观/切向不可观是雷达跟踪领域常识（Doppler ambiguity cone 等）。通过 arXiv 检索确认：**各向异性雷达测量不确定性已大量用于 4D 雷达配准/里程计**（RaDiVe 速度差异点不确定性、图论野值剔除中的各向异性点不确定性、ELMAR 跨模态不确定性等），**但尚无 4D 雷达检测论文将 per-point 各向异性高斯证据用于 BEV pillar 路由**——这是真实的迁移空白。

- 直接竞争者全是**确定性**补偿（SGE-Flow、HyperDet 的 Doppler-guided compensation）。
- 退化关系论证干净：σ→0、q→1 时退化为确定性补偿，说明本方法是确定性补偿的严格推广。
- 消融设计（P1 的 B2 vs B3 vs B5 链）恰好回答"这不就是高斯扩散吗"的质疑。
- **两步归一化修正**（先归一化几何概率 π，再乘可靠性 q）数学上正确，避免 q 在分子分母中抵消——这是方案严谨性的加分项。

**风险**：整个方案的成立系于"各向异性 > 确定性"这一组对比。若 B5 只比确定性补偿高 <0.5 mAP，创新一即空心化。

### 3.2 创新二：自监督可靠性（STER）——评级：中等

与 HyperDet"跨传感器验证筛回波"、MAFF-Net 去噪分支同属"回波质量筛选"家族；差异化在于**无额外传感器的自监督支持度**。审稿人必问的两个漏洞：

1. **循环性**：支持度 s_i 依赖 Σ，而 Σ 可学习——方案已用 detach/固定协方差缓解，论文必须显式说明并做消融（固定 Σ vs 学习 Σ 构造标签）。
2. **系统性错误**：对"整体一致平移的错误补偿"（历史点恰好落在别处点云上）无法识别——需报告失败案例。

单独不足以成为卖点，与创新一绑定为"概率证据构建"整体更稳。

### 3.3 创新三：因果局部融合（CRLF）——评级：中低

Query-based 局部时序注意力 + 几何偏置在相机 BEV（BEVFormer 系）与 LiDAR 时序融合中已有大量同构设计；本模块属于合理工程组合。**只能作为证据框架的下游融合机制主张**。加分项：门控退化为单帧保证下限（历史证据不足时不伤基线）。

### 3.4 辅助项

- **雷达物理增强**降级为辅助策略正确——雷达增强 + 一致性训练已有先例，独立主张会被驳回；但它是跨域实验的关键支撑。
- **RepDWC / MDFEN / CenterHead** 明确标注非创新，避免"拼凑"指控。残余风险：模块总数偏多的"厨房水槽"观感，靠模块退出机制（`project_constraints.md` §6）与论文中清晰的因果链叙事化解。

### 3.5 综合评级

> 创新程度处于该领域已发表工作的**中上区间**，达到 IROS / ITSC / RA-L / T-IV 及 SCI Q1–Q2 的录用门槛；不够 CVPR/NeurIPS/TPAMI 的"新洞察"标准。最大差异化资产：**Doppler 可观性 → 各向异性概率证据 → 检测路由**的完整因果链与"Correct-then-Refine"方法论叙事。

---

## 4. 审稿风险点与应对

| # | 攻击点 | 应对（必须落入论文） |
|---|--------|---------------------|
| 1 | **HyperDet 已做 Doppler 引导运动补偿 + 回波可靠性精炼** | Related work 明确区分：确定性输入级预处理 vs 可微概率证据建模；HyperDet 训练需 LiDAR 监督，本方法纯 radar-only；最好补一组 HyperDet 式确定性预处理复现对比 |
| 2 | **SGE-Flow 确定性补偿是直接基线** | P1 中 deterministic 是必经对照；若优势 <0.5 mAP 需重估方案 |
| 3 | **切向不可观是常识，不能 claim "首次发现"** | 措辞用 "to the best of our knowledge, among the first to *encode* it as anisotropic evidence in pillar routing"，不主张物理发现 |
| 4 | **自监督标签循环性** | detach 协方差消融 + 固定 Σ 构造标签对照 |
| 5 | **模块堆叠、增益微小** | 严格消融链（每个模块独立开关）、报告负结果（退出模块照实写）、单域收益弱时主打跨域 Drop |
| 6 | **提升不过 seed 方差** | ≥3 seed + bootstrap（§15），以 mean±std 为主表 |
| 7 | **无 testing split** | 明确声明以 val 为开发评估，test 集需官方渠道，后续补充（§20.2） |

---

## 5. 性能提升预测

### 5.1 锚点

- 本地 RadarPillars 5 帧复现：**48.76 mAP**（论文 50.70；Car 36.29 / Ped 41.09 / Cyc 68.90）
- Seed 间 σ ≈ 1 mAP（RadarPillar 仓库观察值）
- PRISM 模块已实现但未训练；现有 rot 系列实验 ~34 mAP（判断为非 VoD-EAA 协议或 TJ4DRadSet，待确认，不作为预测锚点）

### 5.2 模块级增量预测（相对 48.76，基于文献同类消融类比）

| 模块链 | 预测增量 | 依据 |
|--------|---------|------|
| 确定性 Doppler 补偿（vs naive 累积） | +0.3 ~ +1.0 | 动态/远距目标受益；VoD 静态占比高限制上限 |
| 各向异性路由（vs 确定性） | +0.3 ~ +0.8 | 成立前提：P1 单调链通过 |
| 可靠性加权 | +0.2 ~ +0.5 | 主要降低 FP，mAP 收益偏小 |
| CRLF（vs 简单拼接/平均） | +0.3 ~ +0.8 | 局部注意力的典型量级 |
| CenterHead + MDFEN（C 版） | +1.0 ~ +2.0 | 主要来自 Pedestrian（头替换经验值 +7.8 Ped on PointPillars→CenterPoint，但雷达上收缩） |

### 5.3 汇总预测

| 配置 | 预测 VoD mAP | 概率 |
|------|-------------|------|
| **S 版**（C=32, AnchorHead） | 49.5–51.0（中位 ~50.3） | P(≥50.0) ≈ 55%；**P(Δ≥1.0 且 p<0.05) ≈ 40–50%** |
| **C 版**（C=48, CenterHead） | 51.0–53.5 | P(≥52.0) ≈ 45% |
| 超越 MAFF-Net（54.6） | — | P < 10%（无 LiDAR 蒸馏不现实） |
| 效率达标（≥60.4 FPS @A4000 当量，≤0.6M 参数） | — | P ≈ 50–60%（时序模块延迟预算偏紧） |

**跨域预测（方案真正可能出彩处）**：

- VoD→TJ4DRadSet：绝对 mAP 仍低（参照 RadarNeXt 32.30），但相对 Drop 有望缩小 **10–25%**。
- K-Radar 恶劣天气：物理增强 + 可靠性抑噪，相对 Drop 有望缩小 **15–30%**。
- 论文叙事建议：**以时序鲁棒性与跨域可靠性为主结果，单域 mAP 为辅**。

---

# 第二部分 实验体系

## 6. 实验总览与依赖关系

```
P0 基线复现 ──> P1 时序基线判决链 ──> P2 CRLF 单验 ──> P3 STER ──> P4 可学习Σ
   (阻塞一切)      (阻塞一切)                                        │
                                                                      ▼
                    P5 RepDWC ──> P6 Lite-MDFEN ──> P7 检测头 ──> P8 联合微调
                    (可与 P2-P4 并行)
```

- **P0、P1 是硬门槛**：不通过则整个方案停止或转向（见 §8 失败预案）。
- P5–P7 每个模块带独立退出标准（`project_constraints.md` §6），触发退出是**预期行为**，照实写入论文。
- 所有训练在远端 `PRISM-4090`（RTX 4090 D）执行；输出统一落 `/root/autodl-tmp` 以避开 30GB 根分区。

## 7. P0：RadarPillars 严格复现

**目标**：建立公平基线与可复现训练管线。

| 项 | 内容 |
|----|------|
| 配置 | `tools/cfgs/vod_models/vod_radarpillar.yaml` |
| 帧数变体 | 1 / 3 / 5 帧（`--set DATA_CONFIG.NUM_SWEEPS {1,3,5}`） |
| 训练 | `--batch_size 16 --epochs 60 --fix_random_seed`（主实验 seed=666） |
| 显著性 | 5 帧变体额外以 seed 42 / 2023 各跑一次（共 3 seed） |
| **成功标准** | 5 帧 mAP 与复现基准 48.76（或论文 50.70）偏差 ≤ 1.0；3 seed 标准差实测值记录在案 |
| 失败处理 | 偏差 >1.5 → 排查数据管线（增强、类别映射、IoU 阈值、NMS 0.10、R40 协议）后重跑，不得继续 |

**命令**（远端）：

```bash
cd /root/PRISM-Pillars
python tools/train.py --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16 --epochs 60 --fix_random_seed --extra_tag p0_5f_seed666
# 3/1 帧与其余 seed 同理，extra_tag 区分
```

## 8. P1：时序基线判决性实验（全方案生死线）

**目标**：以最低成本验证核心假设链条。固定 q=1、固定 σr=0.10 / σt=0.50、无 temporal attention、标准 RadarPillars 骨干。

| # | 方法 | 实现要点 | 预期 |
|---|------|---------|------|
| 1 | naive 累积 | 历史帧直接拼接 | 基线 |
| 2 | ego-motion 对齐 | 历史点经 pose 变换到当前坐标系 | > #1 |
| 3 | 确定性 Doppler 补偿 | μ = p + Δt·v_comp·u，hard assignment | > #2 |
| 4 | 各向同性高斯路由 | Σ = σ²I 软路由 | > #3 |
| 5 | 固定各向异性路由 | Σ = σr²uuᵀ + σt²nnᵀ，σt>σr | > #4 |

**成功标准**：mAP **严格单调**：#1 < #2 < #3 < #4 < #5（允许并列，不允许倒挂超过 seed 噪声 0.3）。

**判决与失败预案**：

- 单调成立 → 进入 P2，论文核心假设成立。
- #3 ≤ #2（确定性补偿无效）→ 说明 VoD 上运动错位不是主要矛盾，转查数据（历史帧是否已预补偿、Δt 符号——先跑 `tests/test_time_sign.py`）。
- #5 ≤ #4（各向异性无效）→ 核心创新空心化：降级为"可靠性 + 局部融合"主线重估，或终止方案。
- 每个变体至少 2 seed；#3 与 #5 各 3 seed（将直接进论文 Table 2）。

## 9. P2–P4：PRISM 三模块逐步验证

### P2：CRLF 独立验证

```bash
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.RELIABILITY.ENABLED False MODEL.DOPPLER_TUBE.LEARNABLE False \
            MODEL.LITE_MDFEN.ENABLED False \
    --extra_tag p2_crlf_only
```

对照组：同配置但 `TEMPORAL_FUSION.ENABLED False`（= P1#5 水平）。成功标准：ΔmAP ≥ 0.5 且延迟增量 ≤ 15%（否则触发 CRLF 退出，回退朴素拼接）。

### P3：STER 可靠性

分阶段训练（防坍缩）：

| Epoch | λ_rel | 说明 |
|-------|-------|------|
| 1–5 | 0，q=1 | 预热，其余模块先收敛 |
| 6–15 | 0 → 0.20 线性升 | σ 保持固定 |
| 16+ | 0.20 | 启用 BCE + ranking + ghost 增强 + sweep dropout |

验证指标（论文 Table 5 素材）：Spearman(q, s)、GT 内/外平均 q、motion-tail 平均 q、q 分桶后的 FP 变化、坍缩监控（mean(q) 曲线）。退出条件：训练不稳定或 ΔmAP < 0.3 → 回退解析可靠性 q = exp(−η₁d − η₂e)。

### P4：可学习 Σ

先冻结 STER 5 epoch 只训 σ-MLP，再联合解冻（分组学习率：Reliability 0.5×、Σ 1.0×、Temporal 0.5×、Backbone 0.2×）。监控 σ 触界频率；有界参数化应使 σr∈[0.03,0.60]、σt∈[σr,2.00]。退出条件：不稳定或频繁触界 → 回退解析增长 σr=0.10+0.15|Δt|、σt=0.50+0.50|Δt|。

## 10. P5–P7：借鉴模块验证（带退出标准）

| 阶段 | 内容 | 对照 | 退出条件 | 回退 |
|------|------|------|---------|------|
| P5 | RepDWC 骨干 | Dense Conv / 普通 DWConv | ΔmAP < −0.3 | BaseBEVBackbone |
| P6 | Lite-MDFEN（单 DCNv3 + raw bypass） | 无 MDFEN；DCN 位置消融（高/中/低分辨率、2×DCN） | ΔmAP < 0.5 或 ΔLatency > +10% | 仅保留 RepDWC |
| P7 | CenterHead（+IoU/dIoU） | AnchorHeadSingle | ΔmAP < 0.5 或 ΔLatency > +10% | AnchorHead |

P5 必须附带部署态等价性测试（`tests/test_rep_parameterization.py`，max_diff < 1e-4）。

## 11. P8：联合微调与最终模型

- 全模块联合微调 10–20 epoch，base LR ≤ 1e-4，backbone 0.5×。
- 产出两个投稿配置：
  - **PRISM-Pillars-RF-S**：C=32、3 历史帧、AnchorHead —— 公平性与效率证明。
  - **PRISM-Pillars-RF-C**：C=48、5 历史帧、CenterHead + IoU/dIoU —— 性能上限。
- 主表同时报告 S 与 C。

## 12. 主实验表设计（Table 1–6）

| 表 | 内容 | 来源阶段 |
|----|------|---------|
| Table 1 | VoD 主结果：PointPillars / RadarPillars 1/3/5f / RadarNeXt / MAFF-Net*（引用） / PRISM-S / PRISM-C，含 Params、GFLOPs、FPS、P95、显存 | P0+P8 |
| Table 2 | 时序证据建模判决链（P1 五变体 + learned Σ + full PRISM），静态/动态、近/中/远分段 | P1–P4 |
| Table 3 | Backbone/Neck/Head 组合矩阵 | P5–P7 |
| Table 4 | MDFEN 消融（FPN/PAN/无 DCN/DCN 位置/双 DCN/无 bypass/final） | P6 |
| Table 5 | 可靠性消融（q=1 / random q / 无 L_rel / BCE / BCE+rank / +ghost aug / 前景概率替代）+ Spearman、ECE、FP | P3 |
| Table 6 | TJ4DRadSet 第二数据集（分别训练与迁移） | §14 |

*MAFF-Net 使用 LiDAR 蒸馏，表注标明非公平对比。

## 13. 鲁棒性实验

| 扰动 | 强度梯度 |
|------|---------|
| 随机点丢失 | 10% / 30% / 50% |
| Doppler 噪声 | 0.1 / 0.3 / 0.5 m/s |
| 自车速度偏置 | 0.2 / 0.5 / 1.0 m/s |
| 历史帧丢失 | 1 / 2 / 3 sweeps |
| RCS 缩放 | 0.8 / 1.0 / 1.2 |
| Ghost 注入 | 5% / 10% / 20% |

对比对象：RadarPillars-5f、PRISM-S（可选 RadarNeXt 引用值）。产出：扰动强度–mAP / Recall / FP 曲线，历史帧数–mAP / Latency 曲线。

## 14. 跨数据集实验

### TJ4DRadSet（必做，Table 6）

- **前置开发**：`pcdet/datasets/tj4dradset/`（当前不存在）——坐标系统一、track ID 索引、真实 Δt、类别映射（Vehicle/Pedestrian/Cyclist）、OpenPCDet info 与 gt-database。
- 实验：① 从头训练（验证模块非 VoD 过拟合）；② VoD→TJ4D 迁移；③ 利用 track ID 做同目标跨帧预测稳定性。
- **禁止**因类别协议差异直接声称严格跨域泛化；只在交集类别评估。

### K-Radar（强烈推荐，扩展验证）

- Normal → Rain / Fog / Snow 与 all-weather → adverse-weather。
- 主数据形式为 4D radar tensor，需固定 tensor→点提取规则（不得按天气分别调阈值）。
- 报告相对 Drop =（AP_source − AP_target）/ AP_source。

## 15. 统计显著性协议

1. 基线与 PRISM 各 ≥3 seed（42 / 666 / 2023，主实验固定 666）。
2. val 集逐帧 AP 差异 → 1000 次 bootstrap → ΔmAP 95% CI；判据：CI 下限 ≥ 0.5 且 p < 0.05。
3. 3 seed mAP 标准差 ≤ 0.30（约束值；若实测基线 σ≈1，则论文以实测值为准并加大 seed 数到 5）。
4. 消融每项 ≥2 run，报告均值±标准差。
5. **禁止**挑选最优 checkpoint 汇报；test 集（若获得）仅最终评估一次。

## 16. 效率测量协议

- 工具：`tools/benchmark_latency.py`（仓库已有规划）；CUDA sync、100 warmup、≥1000 iter、FP32 主报告 + FP16 补充、batch 1 与 4。
- 分模块计时拆分（数据加载 / 对齐 / VFE / STER+DAUT+RAPR / CRLF / RepDWC / MDFEN / Head / 后处理）。
- 目标：S 版端到端 ≤18 ms（A4000 当量）；参数量 ≤0.6M（训练态）/ ≤0.4M（部署态）；GFLOPs ≤3.0。
- DCNv3 若需自定义 ONNX/TensorRT 算子，论文不得仅凭 PyTorch FPS 宣称边缘部署。
- 4090 数值与 A4000 不可直接对比，论文需注明硬件或以相对延迟表述。

---

# 第三部分 算力环境与数据部署

## 17. 远端设备档案

| 项 | 值 |
|----|----|
| 服务商/主机 | seetacloud (AutoDL 容器) `autodl-container-e4254eb262-a6d06801` |
| GPU | NVIDIA GeForce RTX 4090 D ×1（24 GB，425W） |
| 驱动 / CUDA | 595.71.05 / CUDA 12.1（`/usr/local/cuda-12.1`，运行时兼容 13.2） |
| Python | miniconda3 base，Python 3.10.8；torch **2.1.2+cu121**（已验证 `cuda.is_available()=True`） |
| 已有包 | numpy 1.26.3、PyYAML、tensorboard、tqdm |
| 缺失依赖 | easydict、numba/llvmlite、scikit-image、spconv-cu12x、sharedarray（视代码）、pcdet 本身（setup.py develop） |
| 内存 / CPU | 503 GB / 128 核（宿主机共享，容器配额以实测为准） |
| 磁盘 | `/root` overlay **30 GB（勿放大文件）**；`/root/autodl-tmp` **80 GB**（/dev/md0，数据集与输出专用） |
| 项目路径 | `/root/PRISM-Pillars`（已同步） |
| 数据路径 | `/root/autodl-tmp/datasets/vod/` |

**注意**：seetacloud 实例重启后端口会变；届时需更新本机 `~/.ssh/config` 中 `PRISM-4090` 的 Port 并重新 `ssh` 一次接受 host key。

## 18. SSH 接入

本机 `~/.ssh/config` 已配置别名（密钥认证，免密）：

```
Host PRISM-4090
  HostName connect.westb.seetacloud.com
  Port 20801
  User root
  IdentityFile ~/.ssh/id_ed25519
```

常用入口：`ssh PRISM-4090`、`scp <file> PRISM-4090:<path>`。

**传输经验（2026-08-06 实测）**：家庭上行上限 ≈ 5.4 MB/s（原生 scp）；paramiko SFTP 仅 1.1 MB/s（勿用于大文件）。断点续传备份脚本：`%TEMP%\vod_upload.py`（SFTP 续传）。

## 19. 代码同步状态

- 已同步：`pcdet/ tools/ tests/ docker/ docs/ .git/` + 根文件（setup.py、test.py、README.md、LICENSE、.gitignore、.vscode/settings.json）。排除：`build/`（166 MB 编译产物，远端重编译）、`output/`（1.7 GB）、`.pytest_cache/`。
- 完整性：`git ls-files` 本地/远端均 266，HEAD 一致（65202f3）；抽查文件对象哈希与索引一致。
- 行尾：远端已执行 CRLF→LF 规范化并设 `core.autocrlf=input`；`git status` 显示的少量 M/D 为本地未提交变更的忠实映射（minor_upgrade*.md 为本地已删除未提交），非同步错误。

## 20. VoD 数据集：状态、结构与部署步骤

### 20.1 上传清单（已全部完成并校验）

| 文件 | 大小 | 状态 |
|------|------|------|
| view_of_delft_PUBLIC_2.zip | 14,602,639,127 B | ✅ 已上传（字节数一致；`unzip -t` CRC 无错误；scp 原生 5.4 MB/s，43 min） |
| label_2_with_track_ids.zip | 10,865,278 B | ✅ 已上传并解压至 `label_2_with_track_ids/label_2`（6435 帧） |
| view-of-delft-dataset-main.zip | 66,802,576 B | ✅ 已上传并解压至 `devkit/view-of-delft-dataset-main` |

### 20.2 部署后实测事实（2026-08-06 核验）

1. **训练点云完整**：`radar_5frames/training/velodyne` = **8682** 帧（zip 条目 8683 含目录项）；`radar/training/velodyne` 同为 8682。
2. **官方 ImageSets 随包提供（序列级划分，无需自制）**：train **5139** / val **1296** / test **2247** / train_val 6435 / full 8682（5139+1296+2247=8682 ✓）。
3. **全部 8682 帧点云均在 training/velodyne 内**（含官方 test 帧），**test 帧无标签**——开发评估使用官方 val（1296 帧有标签）；论文若需 test 成绩须走官方评估渠道。
4. **符号链接无需手动修复**：Linux `unzip` 正确还原了 zip 内的 symlink 属性（Windows 解压工具会生成文本存根，勿用本地解压树）。`radar_5frames/training/{calib,label_2,pose,image_2}` 与 `ImageSets` 均已指向 `lidar/`、`radar/` 共享数据。
5. **track-id 标签包**（`/root/autodl-tmp/datasets/vod/label_2_with_track_ids/label_2`，6435 帧）：与原始标签唯一差异为**第 2 字段由 truncation 占位符（恒 0）替换为 track ID**。**P0 基线与主实验继续使用原始 label_2**（与公开协议一致）；track-id 版本仅用于后续时序一致性/track 稳定性实验，届时通过配置或软链切换，禁止混用。
6. 磁盘占用：解压后 `/root/autodl-tmp` 约 40 GB 已用 / 41 GB 可用。zip 暂保留，待 P0 跑通后可删除主 zip 回收 14.6 GB。

### 20.3 部署记录（已执行，供复现参考）

```bash
cd /root/autodl-tmp/datasets/vod
unzip -tq view_of_delft_PUBLIC_2.zip            # No errors detected
unzip -oq view_of_delft_PUBLIC_2.zip            # symlinks 自动还原
unzip -oq label_2_with_track_ids.zip -d label_2_with_track_ids
unzip -oq view-of-delft-dataset-main.zip -d devkit
mkdir -p /root/PRISM-Pillars/data/VoD
ln -sfn /root/autodl-tmp/datasets/vod/view_of_delft_PUBLIC \
        /root/PRISM-Pillars/data/VoD/view_of_delft_PUBLIC
# 核对：radar_5frames/training/velodyne = 8682 文件；label_2 经软链可读 6435 文件
```

引导脚本存档：`docker/bootstrap_vod.sh`（含 symlink 存根兜底修复逻辑，供其他机器复现）。

### 20.4 info 生成（环境就绪后）

```bash
cd /root/PRISM-Pillars
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
# 产出 vod_infos_train.pkl / vod_infos_val.pkl / vod_dbinfos_train.pkl
# 注意：sequence-level split 检查（ImageSets 内 train/val 序列不重叠）
```

## 21. 环境部署步骤（命令级）

> ✅ **已完成（2026-08-07）**：依赖安装、pcdet 编译（sm_89）、infos 生成、1 epoch 冒烟训练与完整评估全部通过。实际版本锁定与踩坑修复记录见 `docs/remote_environment.md`（关键点：numpy 锁 1.26.4、numba 锁 0.65.1、eval1 导入修复等 5 项）。以下为原始部署命令，供重建环境时参考。

```bash
ssh PRISM-4090
cd /root/PRISM-Pillars
PY=/root/miniconda3/bin/python; PIP=/root/miniconda3/bin/pip

# 1) 补齐依赖（base 已有 torch 2.1.2+cu121）
$PIP install easydict numba llvmlite scikit-image opencv-python-headless \
             tensorboardX 2>/dev/null || true
$PIP install spconv-cu120    # 若无 cu121 wheel，cu120 在 CUDA 12.1 运行兼容；失败则换 spconv-cu121

# 2) 编译安装 pcdet（nvcc 12.1；首次约 5–15 分钟）
$PY setup.py develop

# 3) 冒烟验证
$PY -c "from pcdet.config import cfg; print('pcdet OK')"
$PY tests/test_time_sign.py 2>/dev/null || echo "（测试脚本以仓库实际文件名为准）"
ls tests/ && for t in tests/test_*.py; do echo "== $t"; $PY "$t" || true; done

# 4) 数据软链 + info 生成（见 §20.3/20.4）

# 5) P0 冒烟训练（1 epoch 快跑，验证数据-模型-评估全链路）
$PY tools/train.py --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16 --epochs 1 --extra_tag smoke
```

**已知兼容风险**：本仓库 OpenPCDet 派生代码的历史编译环境为 py3.8；在 py3.10 + torch2.1 + nvcc12.1 下编译 `pcdet/ops`（roiaware_pool3d / iou3d_nms / pointnet2 等）可能遇到废弃 API（如 `AT_CHECK`、THC 头文件）报错——逐个按现代写法修补即可，属预期工作量。若 spconv 导入失败且短期无解：确认 PillarVFE 路径是否依赖 spconv（pillar 路线通常可绕过），必要时以 `--set` 关闭相关分支。

## 22. 执行排期与 GPU 预算

4090 约为 A4000 的 1.5–2 倍性能，`project_constraints.md` 的 220 A4000-小时 ≈ **110–150 个 4090-小时**。

| 阶段 | 内容 | 预计 run 数 | 单 run（4090） | 小计 |
|------|------|-----------|---------------|------|
| P0 | 基线 1/3/5f + 3 seed | 7 | 1.5–2 h | ~12 h |
| P1 | 5 变体 ×2–3 seed | 12 | 2 h | ~24 h |
| P2 | CRLF + 对照 | 4 | 3–4 h | ~14 h |
| P3 | STER 消融 | 6 | 4 h | ~24 h |
| P4 | 可学习 Σ | 4 | 4 h | ~16 h |
| P5 | RepDWC 4 配置 | 4 | 3 h | ~12 h |
| P6 | MDFEN 7 消融 | 7 | 3.5 h | ~25 h |
| P7 | Head 4 配置 | 4 | 3.5 h | ~14 h |
| P8 | 联合微调 | 3 | 2 h | ~6 h |
| 杂项 | 失败重跑/调参缓冲 | — | — | ~20 h |
| **合计** | | | | **~170 h（约 7 天满负荷）** |

周节奏建议：W1 环境+P0；W2 P1；W3 P2–P3；W4 P4–P5；W5 P6–P7；W6 P8+主表；W7+ 鲁棒性、TJ4DRadSet、K-Radar、写作。

## 23. 风险清单与缓解

| # | 风险 | 概率 | 缓解 |
|---|------|------|------|
| 1 | 提升不过 seed 方差（显著性失败） | 中 | ≥3–5 seed + bootstrap；叙事转跨域 Drop；加大效应量（聚焦动态/远距子集报告） |
| 2 | P1 单调链倒挂 | 中 | §8 判决树；先验单元测试（Δt 符号、协方差正定、概率守恒） |
| 3 | pcdet 编译失败（py3.10/cu121） | 中 | §21 修补路径；备选 conda py3.8 新环境 |
| 4 | autodl-tmp 80 GB 不够 | 低 | 解压后删 zip（−14.6 GB）；output 定期清理 ckpt（保留 best+last） |
| 5 | HyperDet 叙事撞车 | 高 | §4 应对；加速投稿时间窗 |
| 6 | 无 test split | 确定 | val 开发评估 + 声明；后续官方渠道补测 |
| 7 | seetacloud 端口变更/实例回收 | 中 | 重要 ckpt 定期拉回本机；config 端口一行即改 |
| 8 | 单卡排队/抢占 | 低 | 关键 run 用 `--ckpt` 断点恢复 |

## 24. 附录：命令速查

```bash
# 登录与同步
ssh PRISM-4090
scp <local> PRISM-4090:/root/PRISM-Pillars/<path>

# 训练（模板）
cd /root/PRISM-Pillars
/root/miniconda3/bin/python tools/train.py \
    --cfg_file tools/cfgs/vod_models/<cfg>.yaml \
    --batch_size <B> --epochs <E> --fix_random_seed --extra_tag <tag>

# 评估
/root/miniconda3/bin/python tools/test.py \
    --cfg_file tools/cfgs/vod_models/<cfg>.yaml --batch_size 8 \
    --ckpt output/vod_models/<model>/<tag>/ckpt/checkpoint_epoch_<N>.pth

# 磁盘/进程监控
df -h /root/autodl-tmp; nvidia-smi; tail -f output/**/log_train_*.txt
```

---

*维护说明：本文档随实验推进更新；每次阶段判决（P0–P8 的成功/退出判定）后在对应小节追加实测结果与决策记录。*
