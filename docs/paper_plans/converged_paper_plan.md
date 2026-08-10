# PRISM-Pillars-RF 收敛方案（一）：论文贡献与模型方法核心思路

<div align="center">

**统一定位 · 三核心贡献 · 最终方法 · 训练证据 · 论文撰写指南**

版本 1.0 · 2026-08-09 · 收敛自 `paper_plans/` 六份方案文档与 `Train_reports/` 三份训练报告

</div>

> **文档地位**
>
> 本文档与 `converged_experiment_guide.md`（收敛方案二：实验开展指南）共同构成后续论文撰写与模型改进工作的**唯一收敛参考**。
> 六份源文档（`great_upgrade.md`、`great_upgrader_2.md`、`construct_guide.md`、`final_upgrade.md`、`great_upgrade_3.md`、`assessment_and_experiment_plan.md`）保留为思路演进记录；**与本文档冲突时，以本文档为准**。
> 验收阈值以 `project_constraints.md` 为准，训练操作命令以 `Train_guide.md` 为准，本文档不重复其内容。

---

## 目录

- [0. 结论速览](#0-结论速览)
- [1. 定位、标题与中心思想](#1-定位标题与中心思想)
- [2. 最终贡献清单（收敛结论）](#2-最终贡献清单收敛结论)
- [3. 最终模型架构](#3-最终模型架构)
- [4. 模块方法细节（最终采用公式）](#4-模块方法细节最终采用公式)
- [5. 损失函数体系](#5-损失函数体系)
- [6. 训练策略收敛](#6-训练策略收敛)
- [7. 训练实证依据（Train_reports 收敛参考）](#7-训练实证依据train_reports-收敛参考)
- [8. 论文撰写指南](#8-论文撰写指南)
- [附录 A：跨域 Drop 与跨数据集实验全案](#附录-a跨域-drop-与跨数据集实验全案)
  - [A.1 含义与度量定义](#a1-含义与度量定义)
  - [A.2 为什么是"真正机会"（意义）](#a2-为什么是真正机会意义)
  - [A.3 域偏移来源与 PRISM 作用机制](#a3-域偏移来源与-prism-作用机制)
  - [A.4 跨域 Drop 实验设计方案](#a4-跨域-drop-实验设计方案)
  - [A.5 数据集选型与推荐](#a5-数据集选型与推荐)
  - [A.6 TJ4DRadSet 适配方案](#a6-tj4dradset-适配方案)
  - [A.7 K-Radar tensor→点固定提取规则设计](#a7-k-radar-tensor点固定提取规则设计)
  - [A.8 私有/自采数据集实验协议模板](#a8-私有自采数据集实验协议模板)

---

## 0. 结论速览

| 维度 | 收敛结论 |
|------|---------|
| 项目名 | **PRISM-Pillars-RF**（Physics-Guided Reliable Temporal Evidence Fusion with Re-parameterized Foreground Refinement） |
| 中心思想 | 历史雷达回波是**不确定的概率证据**而非确定几何点；**先纠正时序证据，再增强空间表征**（Correct-then-Refine） |
| 核心贡献 | **三个**：各向异性概率证据（DAUT+RAPR）、自监督时序可靠性（STER）、因果局部 Pillar 融合（CRLF） |
| 工程组件 | RepDWC 骨干 + Lite-MDFEN 颈（借鉴 RadarNeXt） |
| 辅助策略 | 雷达物理过程增强 + 跨增强一致性损失（支撑鲁棒性叙事，不作独立创新） |
| 投稿定位 | 首选 T-IV / RA-L / IROS；跨域结果强则冲 TITS；达到 SCI Q1–Q2 门槛；不足以支撑 CVPR/NeurIPS 级主张 |
| 最强差异化 | Doppler 可观性 → 各向异性概率证据 → 检测路由的完整因果链，领域内检测方向暂无先例 |
| 实证现状 | r0 完整训练（四 PRISM 模块联合）VoD 3D mAP **53.29（std）/ 52.21（R40）**，大幅超出预测上界 51.0；但**公平对照与消融隔离尚未完成**（见 §7.3） |
| 生死线 | P1 单调链实验（naive < ego < 确定性 < 各向同性 < 各向异性）**尚未执行**，是创新一成立的核心证据，必须补齐 |
| 跨域机会 | **跨域相对 Drop 缩小**比单域 mAP 更容易过显著性检验，是物理正确性的间接证明；完整方案见**附录 A**（TJ4DRadSet 必做 + K-Radar 可选 + 私有数据协议） |

---

## 1. 定位、标题与中心思想

### 1.1 论文标题（收敛）

> **PRISM-Pillars-RF: Physics-Guided Reliable Temporal Evidence Fusion with Re-parameterized Foreground Refinement for 4D Radar 3D Object Detection**

收敛理由与约束：

1. 代码、配置（`prism_pillars_rf_s.yaml`）与 r0 训练均已按 RF 架构（含 RepDWC/Lite-MDFEN）实现，标题与实现一致。
2. **不得**在标题中使用 "Domain-Robust" / "Domain-Generalized"（早期 `great_upgrade.md`、`construct_guide.md` 的提法已废弃）：跨数据集协议差异大，直接主张跨域泛化会产生不公平比较；跨域只作为支撑实验（§8.6）。
3. 回退预案：若 RepDWC / Lite-MDFEN 在后续验收中触发退出标准（见收敛方案二 §10），标题退化为 `PRISM-Pillars: Physics-Guided Reliability-Aware Probabilistic Temporal Fusion for 4D Radar 3D Object Detection`（`final_upgrade.md` 版本），正文相应删除效率组件章节。

### 1.2 一句话中心思想

> **历史雷达回波是不确定的概率证据，而非确定的几何点；可靠的多帧雷达检测，应先修正不确定的历史证据，再增强融合后的空间表征。**

英文中心陈述（全文反复呼应）：

> Historical 4D radar returns should not be deterministically accumulated as equally reliable points. Instead, they should be represented as reliability-weighted anisotropic probabilistic evidence and selectively retrieved by current pillars through causal local temporal fusion.

支撑中心思想的物理事实链：

1. 自车位姿只能补偿传感器自身运动，不能恢复目标运动；
2. Doppler 只直接约束**径向**速度，**切向速度不可观**；
3. 因此确定性补偿 $\hat p = p + \Delta t \cdot v_r u$ 会把错误写进几何结构（动态拖影、错误 Pillar 分配、ghost 积累）；
4. 历史回波还受到多径、低 RCS、杂波影响，**并非同等可信**；
5. 结论：历史点应表示为"可靠性加权的各向异性概率证据"，并由当前帧按需检索融合。

### 1.3 三个科学问题 + 一个工程问题

| # | 问题 | 回答模块 | 对应实验证据 |
|---|------|---------|-------------|
| Q1 | 历史雷达点能否被当成确定点？ | **DAUT**（各向异性不确定管） | P1 单调链、Table 2 |
| Q2 | 所有历史回波是否同等可信？ | **STER**（自监督可靠性） | 可靠性消融与诊断（Table 5） |
| Q3 | 历史证据应该怎样进入当前帧？ | **RAPR + CRLF**（概率路由 + 因果局部检索） | Table 2 末段、attention 消融 |
| Q4 | 加入时序模块后如何保持实时性？ | RepDWC + Lite-MDFEN + 局部 Top-K（工程） | 效率测量协议（收敛方案二 §9） |

### 1.4 方法论分层（Correct-then-Refine）

| 阶段 | 层级 | 解决的问题 | 采用方法 |
|------|------|-----------|---------|
| Correct | 点级历史证据 | 回波错位、ghost、切向速度不可观 | DAUT 概率证据建模 + STER 可靠性 |
| Correct | Pillar 时序融合 | 历史信息是否值得使用、如何使用 | RAPR 概率路由 + CRLF 局部检索 |
| Refine | BEV 空间特征 | 前景稀疏、不规则、断裂 | Lite-MDFEN 单 DCNv3 前景细化 |
| Refine | 高效部署 | 时序计算导致延迟上升 | RepDWC 重参数化骨干 |

---

## 2. 最终贡献清单（收敛结论）

### 2.1 三个原创贡献（论文只主张这三条）

**创新一：Doppler 各向异性概率证据（DAUT + RAPR）——最强创新**

> We propose a Doppler-aware anisotropic evidence model that represents historical radar returns as spatial probability distributions rather than deterministic compensated points, explicitly distinguishing Doppler-observable radial motion from poorly observable tangential motion.

- 物理前提（径向可观/切向不可观）是雷达跟踪领域常识，**不得主张"首次发现"**；措辞用 "among the first to *encode* it as anisotropic evidence in pillar routing"。
- 经 arXiv 检索确认：各向异性雷达测量不确定性已用于 4D 雷达配准/里程计，但**尚无 4D 雷达检测论文将 per-point 各向异性高斯证据用于 BEV pillar 路由**——这是真实的迁移空白。
- 直接竞争者（SGE-Flow、HyperDet 的 Doppler-guided compensation）全是**确定性**补偿。
- 创新一成立系于"各向异性 > 确定性"这一对比（P1 生死线）；若优势 <0.5 mAP 即空心化，须降级方案。

**创新二：自监督时序证据可靠性（STER）**

> We introduce a self-supervised temporal evidence reliability estimator that learns to suppress unsupported historical returns without additional point-level annotations.

- 与 HyperDet"跨传感器验证"、MAFF-Net 去噪同属"回波质量筛选"家族；差异化在于**无额外传感器的自监督支持度**。
- 单独不足以成为卖点，与创新一绑定为"概率证据构建"整体叙事更稳。

**创新三：因果局部时序 Pillar 融合（CRLF）**

> We develop a causal reliability-aware local pillar fusion mechanism in which current pillars selectively retrieve historical evidence according to feature similarity, anisotropic motion uncertainty, evidence reliability, evidence mass, and temporal distance.

- Query-based 局部时序注意力在相机 BEV / LiDAR 时序中已有同构设计，**只能作为证据框架的下游融合机制主张**，评级中低。
- 加分项：门控机制在历史证据不足时退化为单帧，保证不伤基线下限。

### 2.2 工程组件（明确不作创新主张）

| 组件 | 来源 | 论文写法 |
|------|------|---------|
| RepDWC 可重参数化骨干 | 借鉴 RadarNeXt | "To retain practical efficiency, the proposed temporal evidence model is implemented with a re-parameterizable BEV backbone..." |
| Lite-MDFEN（单 DCNv3 + 原始旁路颈） | 借鉴 RadarNeXt | 同上，归入工程实现 |
| AnchorHead / CenterHead | OpenPCDet | S 版用 AnchorHead 保证公平对比；CenterHead 仅作为 C 版配置 |

> 规避"厨房水槽（模块堆砌）"指控：论文中用 Correct-then-Refine 因果链叙事 + 每个模块独立开关的严格消融 + 照实报告退出模块来化解。

### 2.3 辅助训练策略

**雷达物理过程增强（Radar Process Augmentation）+ 跨增强一致性损失**：

- RCS 仿射扰动、距离相关丢点、方位噪声、Doppler 偏置/缩放、自车补偿噪声、sweep dropout、ghost 注入；
- 定位：**辅助训练策略**，不作独立贡献主张（雷达增强+一致性训练已有先例，独立主张会被驳回）；
- 价值：跨域/鲁棒性实验的关键支撑，是"跨域 Drop 缩小"叙事的来源。

### 2.4 已放弃 / 降级思路一览（收敛裁决，附理由）

| 思路 | 来源 | 裁决 | 理由 |
|------|------|------|------|
| 完整 2D 速度 WLS 反演 | final_upgrade 前版 | **放弃** | 检测前无可靠点分组；同目标视线方向相近导致矩阵病态；切向本不可观，审稿人必质疑 |
| LiDAR 教师蒸馏 | 早期讨论 | **放弃** | 破坏 radar-only 主线，增加实验复杂度（MAFF-Net 用蒸馏导致其 54.6 mAP 不可公平对比，正是本方案的对照论据） |
| 独立时序一致性损失 $L_{temp}$ | great_upgrade 模块 D | **第一版不采用** | 时序注意力已可经检测损失端到端训练；损失项过多增加调参难度 |
| 动态网络宽度 | 早期讨论 | **推迟** | 与物理时序主线联系不足 |
| 跨域泛化作为标题/主主张 | great_upgrade / construct_guide | **降级为支撑实验** | 数据集类别/范围/评价协议差异大；只能称 cross-sensor transfer study |
| 全局时序注意力 | great_upgrade 模块 C 初版 | **弃用** | $O(p^2)$ 成本；改为局部 Top-K 检索 $O(pK)$ |
| K-Radar 作为主开发集 | great_upgrade | **降为可选扩展** | 主数据为 4D radar tensor，预处理链复杂；不为投稿同时维护三条数据链 |
| MAN TruckScenes | great_upgrade | **不纳入** | 多雷达 360° 数据处理复杂度过高，留作后续工作 |
| CenterHead 作为创新主张 | construct_guide | **降级为配置变体** | 检测头替换不构成创新；仅 C 版使用 |

### 2.5 历史命名对照（防止阅读旧文档时混淆）

| 旧名（源文档） | 收敛后名称 | 备注 |
|---------------|-----------|------|
| Point Reliability Estimator / PRE / Module A | **STER** | Self-Supervised Temporal Evidence Reliability |
| Doppler Uncertainty Tube / Module B | **DAUT** | Doppler-Aware Uncertainty Tube |
| Probabilistic Pillar Routing / Module D(路由) | **RAPR** | Reliability-Aware Probabilistic Routing；与 DAUT 合并为创新一主张 |
| DAER（final_upgrade） | DAUT + RAPR | 同一创新的早期合并命名，不再使用 |
| Reliability-aware Temporal Pillar Attention / Module C | **CRLF** | Causal Reliability-Aware Local Fusion（局部化后的最终形态） |
| RBB | RepBEVBackbone（RepDWC） | 工程组件 |
| SR-MDFEN | Lite-MDFEN | 工程组件 |
| Radar Process Augmentation / Module D(增强) | RADAR_AUG | 辅助策略 |

---

## 3. 最终模型架构

### 3.1 总体结构（双流）

```text
                          ┌──────────────────────────────┐
Current radar frame P_t ─►│ 当前帧确定性分支               │
                          │ 速度分量分解 + PillarVFE       │
                          │ + PillarAttention             │
                          └──────────────┬───────────────┘
                                         │ Current Pillars（Query）
                                         ▼
Historical frames P_{t-k} ─► 自车运动补偿对齐
              │
              ├─ 共享点特征编码
              ├─ STER  自监督可靠性 q_i
              ├─ DAUT  各向异性不确定管 (μ_i, Σ_i)
              ├─ RAPR  可靠性加权概率路由 → 历史证据 Pillar
              │        （含证据质量 m_j、平均可靠性 q̄_j）
              │                          Key / Value
              ▼                                │
        CRLF 因果局部时序融合 ◄────────────────┘
              │   F_i^out = F_i + g_i · Ĥ_i
              ▼
     PointPillar Scatter → BEV
              ▼
     RepDWC 可重参数化骨干（三阶段 [3,5,5], C=[32,32,32]）
              ▼
     Lite-MDFEN 颈（单 DCNv3 作用于高分辨率原始特征 + raw bypass）
              ▼
     AnchorHead（S 版）/ CenterHead（C 版）→ 3D 检测
```

关键结构决策（收敛）：

- **双流**：当前帧走 RadarPillars 原始确定性编码；只有历史帧走概率证据链；历史证据不直接与当前 BEV 相加，由当前 Pillar 作为 Query 选择性访问——避免"soft routing 融合一次、attention 又融合一次"的逻辑重复。
- **检测器** `PRISMPillarsRF` 已注册并训练验证（r0）；代码位于 `pcdet/models/radar_evidence/`、`pcdet/models/temporal/`、`pcdet/models/backbones_2d/`。

### 3.2 两条架构纪律

1. **当前帧不做概率扩散**——当前帧坐标是直接观测，保留 RadarPillars 原始确定性编码，防止目标轮廓被平滑。
2. **DCNv3 不进入物理证据建模之前**——可变形卷积会改变空间响应，必须放在物理引导的时序融合之后。正确顺序：点级物理修正 → 概率时序融合 → BEV 可变形增强。

### 3.3 模块清单与参数量

| # | 模块 | 缩写 | 论文位置 | 参数量 | 定位 |
|---|------|------|---------|--------|------|
| 1 | 雷达点特征编码（含速度分解） | — | Method §3.2 | ~1K | 继承 RadarPillars |
| 2 | 自监督时序可靠性 | STER | §3.3 | ~3K | 创新二 |
| 3 | 时序支持度构建（伪标签） | — | §3.3 | 0（固定运算） | 创新二配套 |
| 4 | Doppler 各向异性不确定管 | DAUT | §3.4 | ~4K | 创新一 |
| 5 | 概率 Pillar 路由 | RAPR | §3.5 | 0（无参数） | 创新一 |
| 6 | 因果局部 Pillar 融合 | CRLF | §3.6 | ~80K | 创新三 |
| 7 | RepDWC 骨干 | — | §3.7 | ~30K（部署态） | 工程 |
| 8 | Lite-MDFEN 颈 | — | §3.8 | ~25K | 工程 |
| 9 | AnchorHead / CenterHead | — | §3.9 | ~30K / ~50K | 工程 |

总参数量：PRISM-Pillars-RF-S 约 0.5M（训练态）/ 0.35M（部署态）；RadarPillars 基线约 0.27M。约束上限见 `project_constraints.md` §5（S 版训练态 ≤0.6M、部署态 ≤0.4M、GFLOPs ≤3.0）。

---

## 4. 模块方法细节（最终采用公式）

> 本节公式以 `final_upgrade.md` 的修正版 + `great_upgrade_3.md` 的最终版为准；早期 `construct_guide.md` 中的 softplus 参数化、含 q 一并归一化的路由公式**已废弃**。

### 4.0 统一点格式与共享编码

点格式（12 维）：`[x, y, z, log_rcs, v_rel, v_comp, delta_t, range, sin_az, cos_az, sweep_idx, source_sensor_id]`；
STEr/DAUT 使用的附加局部统计量：`local_density, local_rcs_mean, local_doppler_std, ego_comp_residual`（由 `radar_geometry.py` 分块计算，已修复 cdist OOM）。

视线单位向量与速度分解：

$$
\mathbf u_i=\frac{[x_i,y_i]^\top}{\sqrt{x_i^2+y_i^2}+\epsilon},\qquad
v_{x,i}=v^{comp}_{r,i}u_{x,i},\quad v_{y,i}=v^{comp}_{r,i}u_{y,i}
$$

数据约束：sequence 级划分；真实 $\Delta t$（秒，非帧编号）；先自车位姿变换再径向预测；只用过去帧（因果性，`tests/test_causal_sequence.py` 验证）；$\Delta t$ 符号由 `tests/test_time_sign.py` 单元测试锁定。

### 4.1 DAUT：Doppler 各向异性不确定管

**均值**（确定性径向补偿部分）：

$$
\boldsymbol\mu_i=\mathbf p_i+\Delta t_i\, v^{comp}_{r,i}\,\mathbf u_i
$$

**时间相关协方差**（预测速度不确定性而非直接预测位置标准差）：

$$
s_{r,i}=\sigma_{p,r}+|\Delta t_i|\sigma_{v,r,i},\qquad
s_{t,i}=\sigma_{p,t}+|\Delta t_i|\sigma_{v,t,i}
$$

$$
\mathbf\Sigma_i=s_{r,i}^{2}\mathbf u_i\mathbf u_i^\top+s_{t,i}^{2}\mathbf n_i\mathbf n_i^\top+\sigma_0^2\mathbf I,\qquad \mathbf n_i=[-u_y,u_x]^\top
$$

**物理约束**：$s_{t,i}\ge s_{r,i}$（切向不可观）。

**有界参数化**（防止 σ 无限增大）：

$$
\sigma_{v,r}=\sigma_{r,\min}+(\sigma_{r,\max}-\sigma_{r,\min})\operatorname{sigmoid}(a_r)
$$

$$
\sigma_{v,t}=\sigma_{v,r}+(\sigma_{t,\max}-\sigma_{v,r})\operatorname{sigmoid}(a_t),\qquad [a_r,a_t]=\operatorname{MLP}[r_i,\log RCS_i,|v^{comp}_{r,i}|,|\Delta t_i|,d_i]
$$

**参数取值**（与当前配置一致）：

| 参数 | 值 | 说明 |
|------|----|----|
| $\sigma_{p}$ (SIGMA_POSITION_BASE) | 0.03 m | 定位基差 |
| $\sigma_{r,\min}/\sigma_{r,\max}$ | 0.03 / 0.60 m | 径向速度不确定范围 |
| $\sigma_{t,\max}$ | 2.00 m | 切向上限 |
| 固定初值（开发/回退用） | $\sigma_r=0.10,\ \sigma_t=0.50$ m | P1 实验与可学习 σ 不稳定时的回退 |
| 解析回退 | $s_r=0.10+0.15|\Delta t|,\ s_t=0.50+0.50|\Delta t|$ | P4 退出标准触发时使用 |

**退化关系（论文论证要点）**：当 $\sigma\to0$ 且 $q_i=1$ 时，概率路由退化为确定性 Doppler 补偿——本方法是确定性补偿的**严格推广**。

### 4.2 STER：自监督时序证据可靠性

输出 $q_i\in[0,1]$：**不是前景概率**，而是"该历史回波是否值得被时序融合相信"（静止车辆/护栏可以是高可靠回波；动态拖影/多径 ghost/低 RCS 孤立点是低可靠回波）。

网络：`Linear(C_in,32) → LayerNorm → SiLU → Linear(32,32) → SiLU → Linear(32,1) → Sigmoid`。

**时序支持度伪标签**（在当前帧点云中寻找支持）：

$$
s_i=\max_{j\in\mathcal P_t}\exp\left[-\frac12(\mathbf p_j-\boldsymbol\mu_i)^\top\bar{\mathbf\Sigma}_i^{-1}(\mathbf p_j-\boldsymbol\mu_i)\right]
$$

关键修正（收敛裁决，必须遵守）：

1. $\bar{\mathbf\Sigma}_i$ 必须使用**固定协方差**或**学习协方差 detach() 后的值**——否则可靠性网络与不确定性网络"共谋"扩大 σ，使所有点获得虚假高支持（审稿人必问的循环性问题）；
2. 三段式伪标签：$y_i=1$（$s_i>0.6$），$y_i=0$（$s_i<0.2$），其余 ignore；
3. 损失：$\mathcal L_{rel}=\mathcal L_{FocalBCE}(q_i,y_i)+0.2\,\mathcal L_{rank}$，排序损失 $\mathcal L_{rank}=\max(0,m-q_i^++q_i^-)$，$m=0.2$。

**防坍缩**：

- 全 $q\to1$：负样本、ghost 注入、ranking loss；
- 全 $q\to0$：训练前期加轻量均值约束 $|mean(q)-\rho_q|$，$\rho_q=0.5\sim0.7$；
- 训练期监控 $mean(q)$ 曲线（坍缩监控是 Table 5 必备素材）。

### 4.3 RAPR：可靠性加权概率 Pillar 路由

对历史点 $i$，在其均值附近 $K_r\times K_r$（默认 5×5）个 Pillar 内计算几何概率：

$$
g_{ij}=\exp\left[-\frac12(\mathbf c_j-\boldsymbol\mu_i)^\top\mathbf\Sigma_i^{-1}(\mathbf c_j-\boldsymbol\mu_i)\right]
$$

**两步归一化（关键数学修正，不得写错）**：

$$
\pi_{ij}=\frac{g_{ij}}{\sum_{j'\in\mathcal N(i)}g_{ij'}+\epsilon}\quad\text{(第一步：只归一化几何概率)}\qquad\Longrightarrow\qquad w_{ij}=q_i\,\pi_{ij}\quad\text{(第二步：再乘可靠性)}
$$

> 若在归一化前把 $q_i$ 放入分子分母，$q_i$ 会被抵消，可靠性实际不生效——这是早期 `construct_guide.md` 公式的已知缺陷。

历史 Pillar 特征与证据质量：

$$
\mathbf H_j=\frac{\sum_iw_{ij}\phi(\mathbf z_i)}{\sum_iw_{ij}+\epsilon},\qquad m_j=\sum_iw_{ij},\qquad \bar q_j=\frac{\sum_iw_{ij}q_i}{m_j+\epsilon}
$$

**证据质量门**（防止低可靠点产生虚高平均特征）：$\widetilde{\mathbf H}_j=(1-e^{-m_j})\mathbf H_j$。

送入 CRLF 的量：历史特征、证据质量 $m_j$、平均可靠性 $\bar q_j$、Pillar 聚合协方差、平均 $\Delta t$。

实现要点：第一版用 PyTorch `scatter_add_`（已实现）；2D 协方差逆用解析式，不逐点调用 `torch.linalg.inv`；注意空 Pillar NaN、FP16 数值稳定。推荐参数：`NEIGHBOR_SIZE=5, MIN_RELIABILITY=0.05, MAX_HISTORY_POINTS=2048, USE_EVIDENCE_MASS_GATE=true`（与配置一致）。

### 4.4 CRLF：因果局部时序 Pillar 融合

当前帧有效 Pillar $\mathbf F_i^t$ 作为 Query，只在半径 $R=3$ 的局部区域检索有效历史 Pillar，保留 Top-$K_t=16$。**不做全局注意力**，复杂度 $O(pK_t)$。

**注意力打分（四项先验）**：

$$
e_{ij}=\frac{(\mathbf W_Q\mathbf F_i)(\mathbf W_K\mathbf H_j)^\top}{\sqrt d}+b^{geo}_{ij}+\alpha\log(\bar q_j+\epsilon)+\gamma\log(1+m_j)-\beta|\Delta t_j|
$$

$$
b^{geo}_{ij}=-\frac12(\mathbf c_i-\mathbf c_j)^\top(\mathbf\Sigma_j+\sigma_c^2\mathbf I)^{-1}(\mathbf c_i-\mathbf c_j)
$$

含义：特征相似 / 马氏几何一致 / 低可靠不主导 / 证据量奖励 / 时间越远越谨慎。初始值 $\alpha=1.0,\ \gamma=0.5,\ \beta=1.0$（与配置一致）。

**熵门控融合**：

$$
\widehat{\mathbf H}_i=\sum_j\operatorname{softmax}_j(e_{ij})\mathbf W_V\mathbf H_j,\qquad E_i=-\sum_ja_{ij}\log(a_{ij}+\epsilon)
$$

$$
g_i=\operatorname{sigmoid}\left[\operatorname{MLP}(\mathbf F_i,\widehat{\mathbf H}_i,\max_ja_{ij},E_i,\bar q_i,m_i)\right],\qquad \mathbf F_i^{out}=\mathbf F_i+g_i\widehat{\mathbf H}_i
$$

**下限保证**：历史证据不足、注意力熵高或可靠性低时 $g_i\to0$，模型退化为单帧 RadarPillars——历史模块不伤基线。

推荐配置（当前值）：`HIDDEN_DIM=64, NUM_HEADS=4, LOCAL_RADIUS=3, TOPK=16`。消融范围：半径 {1,3,5}、TopK {8,16,32}、heads {2,4,8}、β {0,0.5,1,2}。

### 4.5 RepDWC 可重参数化骨干（工程组件）

- 三阶段 `[3,5,5]` blocks、stride `[1,2,2]`、均匀通道 `C=[32,32,32]`（S 版；C 版候选 `[48,48,48]`）；不照搬 RadarNeXt 的 C=64（RadarPillars 已证明极稀疏雷达适合更小均匀通道）。
- 训练态多分支（DWConv 3×3/1×1 + BN、PWConv 3×3/1×1 + BN、identity BN），部署态折叠为单路径 Conv；折叠公式与流程见 `great_upgrade_3.md` §9.2 与 `tools/convert_to_deploy.py`。
- **等价性验收**：转换前后 $\max|f_{train}(x)-f_{deploy}(x)|<10^{-4}$（FP16 放宽至 $10^{-3}$），覆盖 stride 1/2、有无 identity、batch 1/8（`tests/test_rep_parameterization.py`）。

### 4.6 Lite-MDFEN 颈（工程组件）

保留 RadarNeXt 消融支持的两个原则：**只用一个 DCNv3；保留未修改的原始特征旁路**（多个 DCNv3 收益不稳定，部分配置反而损失精度）。

融合拓扑（自上而下 + 自底向上双路径）：

$$
T_2=\operatorname{RepDWC}[\operatorname{Concat}(F_2,\operatorname{Up}(F_3))],\qquad E_1=\operatorname{DCNv3}(F_1)
$$

$$
T_1=\operatorname{RepDWC}[\operatorname{Concat}(F_1,E_1,\operatorname{Up}(T_2))],\qquad B_2=\operatorname{RepDWC}[\operatorname{Concat}(T_2,\operatorname{Down}(T_1))]
$$

$$
F_{neck}=\operatorname{Concat}[T_1,\operatorname{Up}(B_2),\operatorname{Up}(F_3)]\ \xrightarrow{1\times1\ conv}\ 输出
$$

配置（当前值）：`CHANNELS=32, DCN_KERNEL_SIZE=3, DCN_GROUPS=4, DCN_PATH=HIGH_RES_RAW_FEATURE, PRESERVE_RAW_BYPASS=true, OUTPUT_CHANNELS=96`。DCNv3 依赖 MMCV；不可用时自动降级标准 Conv2d（约 1–2 mAP 损失，需在论文实现细节中说明）。

### 4.7 检测头策略

| 配置 | 头 | 用途 | 准入条件 |
|------|----|----|---------|
| **PRISM-Pillars-RF-S** | AnchorHeadSingle | 与 RadarPillars 严格公平对比 + 效率证明 | — |
| **PRISM-Pillars-RF-C** | CenterHead + IoU/dIoU | 性能上限展示 | $\Delta mAP\ge0.5$ 且 $\Delta Latency\le10\%$，否则主模型保持 AnchorHead |

论文主表同时报告 S 与 C。

---

## 5. 损失函数体系

第一版总损失（**四项**，不含 $L_{temp}$）：

$$
\mathcal L=\mathcal L_{det}+\lambda_{rel}\mathcal L_{rel}+\lambda_\sigma\mathcal L_\sigma+\lambda_{inv}\mathcal L_{inv}
$$

| 损失项 | 权重 | 内容 | 生效条件 |
|--------|------|------|---------|
| $\mathcal L_{det}$ | 1.0 | Focal + SmoothL1 + 方向分类（cls:loc:dir = 1:2:0.2，沿用 RadarPillars） | 始终 |
| $\mathcal L_{rel}$ | **0.20** | FocalBCE + 0.2·ranking（§4.2） | 有历史帧且启用 STER；分阶段 warm-up |
| $\mathcal L_\sigma$ | **0.01** | $\operatorname{mean}[\max(0,s_r-s_{r,\max})+\max(0,s_t-s_{t,\max})+\max(0,s_r-s_t)]$（有界参数化下最后一项基本不需要） | 启用可学习 DAUT 时 |
| $\mathcal L_{inv}$ | **0.05** | 跨增强一致性，仅 GT/高置信前景区域 $\Omega$，带 stopgrad：$\frac1{|\Omega|}\sum_{j\in\Omega}\|\operatorname{norm}(F_j^a)-\operatorname{stopgrad}[\operatorname{norm}(F_j^b)]\|_2^2$ | 雷达增强双视图可用时 |

---

## 6. 训练策略收敛

### 6.1 联合训练 vs 分阶段训练（收敛裁决）

| 策略 | 证据 | 裁决 |
|------|------|------|
| 全程联合训练 | **r0 已验证**：80 epoch 联合训练四 PRISM 模块，mAP 53.29/52.21，全程稳定无坍缩 | ✅ 主路线成立 |
| 分阶段协议（P0: q=1/λ=0 → P1: λ warm-up → P2: learned σ） | 代码已实现（commit ec92570），**尚未训练验证**；预期增益 +0.5–1.0，主要提升 STER 质量与训练稳定性 | ⏳ 作为 Cycle 1 增强项验证；若联合训练在多 seed 下同样稳定，可退回联合训练 |

### 6.2 已实现的分阶段协议（配置 `PHASED_TRAINING`）

| 阶段 | Epoch | 内容 |
|------|-------|------|
| Phase 0 | 1–5 | q=1，λ_rel=0，σ 固定（其余模块先收敛） |
| Phase 1 | 6–15 | λ_rel 从 0 线性升至 0.20，q 开始学习，σ 保持固定 |
| Phase 2 | 16+ | learned σ 启用，全模块完整训练（BCE + ranking + ghost 增强 + sweep dropout） |

### 6.3 优化器与调度（与基线对齐，保证公平）

Adam + OneCycle（lr_peak=0.003，PCT_START=0.4，DIV_FACTOR=10，weight_decay=0.01，GRAD_NORM_CLIP=10）；80 epoch；梯度裁剪 10。AMP 混合精度（bfloat16，Ada 平台数值稳定）+ cdist 分块计算后 **bs=8** 可跑（4090D 24GB）。禁用 GT 采样增强（时序公平）。

### 6.4 P4 可学习 σ 的分组学习率（当分阶段协议启用时）

先冻结 STER 5 epoch 只训 σ-MLP，再联合解冻：Reliability 0.5×、Σ 1.0×、Temporal 0.5×、RadarPillars 主干 0.2×。监控 σ 触界频率（有界参数化应使 $\sigma_r\in[0.03,0.60]$、$\sigma_t\in[\sigma_r,2.00]$）。

---

## 7. 训练实证依据（Train_reports 收敛参考）

### 7.1 关键数字（截至 2026-08-09）

| 实验 | 平台/协议 | 结果 | 来源 |
|------|----------|------|------|
| RadarPillar 基线（60 ep, bs=16） | RTX 5090 | bbox R40 mAP **58.07**（best ep55）；3D R40 mAP **47.21**（best ep54）/ 46.91（ep59）；Car 是最弱类别且波动最大 | `round1_report.md` |
| RadarPillars 5f 历史复现锚点 | — | 3D mAP **48.76**（Car 36.29 / Ped 41.09 / Cyc 68.90）；论文值 50.70；验收基准取 49.00 | assessment §5.1 / project_constraints |
| **PRISM-Pillars-RF-S Round 0**（80 ep, bs=4, 四模块联合） | RTX 4090D | 3D mAP **53.29（std）/ 52.21（R40）** @ ep79；Car 41.45 / Ped 45.93 / Cyc 72.50（std AP）；loss 3.90→0.657；9h40m；无 OOM | `20260809_RTX4090D_r0_report.md` |
| Smoke（2 ep, bs=4） | RTX 4090D | 确认 bs=8 因 `torch.cdist` O(N²) OOM；bs=4 安全；吞吐 2.7–3.0 it/s | `20260809_RTX4090D_smoke_report.md` |

对照预测：assessment 对 S 版预测 49.5–51.0（P(Δ≥1.0 且 p<0.05)≈40–50%）——r0 单 run 超出上界 +2.29，是积极信号，但**先完成 §7.3 的公平性核验再写入论文**。

### 7.2 方法可靠性分级（用于选择后续改造方法）

| 方法/改造 | 实证状态 | 可靠性判定 |
|-----------|---------|-----------|
| DAUT+RAPR / STER / CRLF 四模块联合 | r0 完整训练，大幅超预测 | ✅ **高**（但缺 P1 对照与逐项消融隔离，论证链未闭合） |
| RepDWC 骨干、Lite-MDFEN 颈 | r0 中参与训练，稳定 | ✅ 较高（部署等价性测试、DCN 位置消融待做） |
| AnchorHead（S 版）、OneCycle lr=0.003、80 ep 联合训练 | r0 验证 | ✅ 高 |
| cdist 分块、AMP bf16、bs=8 | 代码已提交（ec92570/52659d5/38768ea/7b868eb） | ✅ 基础设施可靠 |
| 雷达物理过程增强 | 代码已实现，**未训练验证** | ⏳ Cycle 1 验证（预期 +1.5–2.5 与跨域 Drop 缩小） |
| 分阶段训练协议 | 代码已实现，**未训练验证** | ⏳ Cycle 1 验证（预期 +0.5–1.0） |
| P1 时序基线单调链 | **未执行** | ⛔ 生死线，必须补 |
| CenterHead（C 版） | 未开始 | ⏳ Cycle 3 |
| TJ4DRadSet 适配 | 数据集代码不存在（`pcdet/datasets/tj4dradset/` 待开发） | ⏳ |
| K-Radar | 未开始 | 可选扩展 |

### 7.3 必须正视的三个 caveat（写论文前解决）

1. **协议差距**：r0（80 ep、bs=4、4090D）与 round1 基线（60 ep、bs=16、5090）、48.76 锚点（更早复现）协议不一致；+4.5~6.4 mAP 的表观差距中可能有训练预算/硬件/评估期（r0 仅评 ep70–80）贡献。**必须补同协议（同硬件、同 80 ep、同增强策略、3 seed）的正面对照**。
2. **无消融隔离**：r0 是全模块联合结果，Table 2/Table 5 所需的逐模块增量尚无数据；无法排除"增益主要来自额外 20 epoch / bs 差异"的质疑。
3. **单 seed**：53.29 是单 seed 单 run；领域 seed 间 σ≈1 mAP，论文主表必须 mean±std + bootstrap（收敛方案二 §8）。

### 7.4 r0 暴露的可改进点（Cycle 1 已对应）

- Car 类波动最大（bs=4 梯度噪声）→ bs=8（已实现）；
- GPU 利用率 58–67%，存在 CPU 数据端瓶颈 → 数据加载优化（后续）；
- ep70–80 仍有 ±0.3 mAP 波动、未完全收敛 → 可延长至 100–120 ep 观察。

---

## 8. 论文撰写指南

### 8.1 主张边界

**应该主张**：

> PRISM-Pillars-RF models historical radar returns as reliability-weighted anisotropic probabilistic evidence rather than deterministic compensated points, enabling robust multi-frame 4D radar object detection under temporal misalignment.

**不应主张**：首次解决多帧运动拖影；首次利用 Doppler；比 RadarPillars 更快更轻；所有数据集/类别显著提升；"Domain-Generalized"（除非跨传感器实验严格且稳定）。

安全措辞模板：

> To the best of our knowledge, PRISM-Pillars-RF is among the first radar-only pillar-based detectors that jointly models point-level temporal reliability, Doppler-induced anisotropic motion uncertainty, and probabilistic temporal pillar routing for multi-frame 4D radar detection.

### 8.2 摘要模板（数字待实验完成后填）

> 4D imaging radar provides robust range, elevation and Doppler measurements for autonomous driving, but its point clouds remain extremely sparse and noisy. Multi-frame accumulation increases point density, yet existing approaches generally treat motion-compensated historical returns as deterministic and equally reliable points. This assumption is problematic because Doppler only constrains radial motion, while tangential motion, multipath reflections and compensation errors introduce substantial spatial uncertainty. We present PRISM-Pillars-RF, a lightweight radar-only detector that models historical returns as reliability-weighted anisotropic probabilistic evidence. A Doppler-aware anisotropic uncertainty tube constructs probabilistic historical evidence, a self-supervised temporal reliability estimator suppresses unsupported returns, and a causal local pillar fusion module lets current pillars selectively retrieve historical evidence according to feature similarity, motion uncertainty, reliability, evidence mass and temporal distance. To retain practical efficiency, the fused representation is refined by a re-parameterizable BEV backbone and a single-deformable raw-bypass neck. Experiments on [datasets] demonstrate that PRISM-Pillars-RF improves multi-frame detection and robustness while retaining the computational efficiency of pillar-based radar detectors.

**摘要随叙事主线调整**（对应 §8.9 决策树）：若最终走鲁棒性叙事，把末句改为强调跨域证据，例如 "...improves multi-frame detection and, importantly, reduces cross-domain performance degradation under unseen radars and adverse weather..."，并补一句跨域 Drop 缩小的定量结论；数字一律待实验完成后填入。

### 8.3 Introduction 六段逻辑

1. 4D 雷达优势（全天候、Doppler、低成本）与稀疏问题；
2. RadarPillars / RadarNeXt 等高效 pillar 检测器进展（速度分解、PillarAttention、uniform scaling 是继承的基线，不是本文创新）；
3. 多帧累积的必要性与缺陷（拖影、错位、ghost）；
4. 现有确定性补偿的问题（Doppler 只约束径向；SGE-Flow/HyperDet 均为确定性路线）；
5. 本文洞察：*Historical returns are uncertain temporal evidence rather than deterministic geometric points.* + Correct-then-Refine 方法论；
6. 三个贡献（严格对应 §2.1，不引入无关模块）。

**鲁棒性叙事增强（可选第 7 段）**：若按 §8.9 走跨域/鲁棒性主线，在第 4 段后补一段"第四痛点"——即使单域补偿正确，**跨传感器/跨天气的域偏移仍使历史证据的可靠性分布漂移**，确定性融合会把域偏移放大为结构错误；由此引出"物理可观性建模天然利于跨域泛化"的论点，并在贡献表述中把跨域 Drop 缩小列为实验贡献。

### 8.4 Related Work 五小节 + 竞品定位

```text
2.1 Radar-only 4D object detection
2.2 Pillar-based efficient radar detection
2.3 Multi-frame radar aggregation
2.4 Motion uncertainty and reliability
2.5 Re-parameterizable and deformable feature extraction
```

| 竞品 | 与本文关系 | 写法要点 |
|------|-----------|---------|
| **HyperDet**（2026, arXiv 2602.11554） | 最大叙事威胁：已占"Doppler 引导回波纠正" | 明确区分：确定性输入级预处理 vs 可微概率证据建模；HyperDet 训练需 LiDAR 监督，本文纯 radar-only；最好补一组 HyperDet 式确定性预处理复现对比 |
| **SGE-Flow** | 确定性补偿直接基线 | P1 中 deterministic 是必经对照；若本文优势 <0.5 mAP 需重估方案 |
| **MAFF-Net** | VoD 54.6 头部 | 表注标明含 LiDAR 蒸馏、非公平对比 |
| **RadarNeXt** | 效率基线（RepDWC/MDFEN 来源） | 引用其 67.10 FPS@A4000、VoD 50.48；本文效率组件致敬并做退出验证 |
| **RadarGaussianDet3D** | 表示层替换路线 | 区分：其高斯原语为稠密化 BEV；本文协方差受 Doppler/可观性显式约束，目标是纠正多帧错位 |
| **SMURF / SCKD** | 引用对比值 | 标记 reported / reproduced |

### 8.5 Method 九小节结构

```text
3.1 Overview and Correct-then-Refine principle
3.2 Radar feature encoding（统一点格式、速度分解、序列加载）
3.3 Self-supervised temporal reliability（STER）
3.4 Doppler anisotropic uncertainty tube（DAUT）
3.5 Reliability-aware probabilistic routing（RAPR，含两步归一化与证据门）
3.6 Causal local temporal pillar fusion（CRLF，含熵门控与退化下限）
3.7 Re-parameterizable BEV backbone（RepDWC 训练/部署态）
3.8 Single-deformable raw-bypass neck（Lite-MDFEN）
3.9 Detection head and loss（S/C 两版 + §5 损失）
```

### 8.6 Experiments 十小节结构

```text
4.1 Datasets and protocols（VoD 主，TJ4DRadSet 次；EAA/R40；序列级划分；val 开发评估声明——无公开 test 标签）
4.2 Implementation details（硬件、AMP bf16、bs、OneCycle、DCNv3 降级说明）
4.3 Main comparison（Table 1）
4.4 Temporal evidence analysis（Table 2 = P1 判决链 + learned Σ + full）
4.5 Reliability and uncertainty analysis（Table 5 + q/σ 可视化）
4.6 Backbone and neck analysis（Table 3/4）
4.7 Robustness experiments（扰动梯度曲线；与 4.8 互为印证：受控域偏移 vs 真实域偏移）
4.8 Cross-sensor transfer and cross-domain Drop（TJ4DRadSet 必做 + K-Radar 天气可选；实验矩阵 CS-1~CS-7 / W-1~W-5、指标与红线见**附录 A.4**；协议无法统一时不得称 domain generalization）
4.9 Efficiency and deployment（分模块延迟、部署态等价性、参数量/GFLOPs；DCNv3 需自定义 ONNX/TRT 算子时不得仅凭 PyTorch FPS 宣称边缘部署）
4.10 Qualitative results and failure cases（必须包含失败案例：整体一致平移的错误补偿等 STER 无法识别的情形）
```

### 8.7 图表清单

主文 9 图：① 总体架构（Correct→Fuse→Refine）；② Doppler 各向异性 Tube（径向窄/切向宽 vs 确定性点）；③ 概率 Pillar 路由（一点向多 Pillar 分配概率质量）；④ 因果局部检索融合；⑤ 历史帧数–mAP/延迟曲线；⑥ q 与 σ 可视化（高 q 在真实目标、低 q 在拖影/ghost）；⑦ naive / deterministic / PRISM 定性对比；⑧ 扰动鲁棒性曲线；⑨ **跨域 Drop 对比柱状图**（基线 vs PRISM，按跨传感器/天气场景分组，见附录 A.4.6）。补充材料：超参曲线、距离区间、延迟细分、Drop–距离曲线、更多失败案例。

表格 Table 1–6 的详细设计见收敛方案二 §4；跨域 Drop 结果表（Table 6 及天气鲁棒性表）设计见附录 A.4.6。

### 8.8 审稿风险与应对（必须落入论文）

| # | 攻击点 | 应对 |
|---|--------|------|
| 1 | HyperDet 已做 Doppler 引导补偿 + 可靠性精炼 | §8.4 区分 + radar-only 定位 + 确定性预处理复现对比 |
| 2 | SGE-Flow 确定性补偿是直接基线 | P1 必经对照；优势 <0.5 则重估 |
| 3 | 切向不可观是常识 | 措辞 "among the first to encode..."，不主张物理发现 |
| 4 | 自监督标签循环性（s_i 依赖可学习 Σ） | detach/固定 Σ 构造标签 + 消融（固定 vs 学习 Σ） |
| 5 | 模块堆叠、增益微小 | 每模块独立开关的严格消融；退出模块照实写；单域弱则主打跨域 Drop 与鲁棒性 |
| 6 | 提升不过 seed 方差 | ≥3 seed + bootstrap，mean±std 主表 |
| 7 | 无 test split | 声明 val 为开发评估，test 需官方渠道后续补 |

### 8.9 投稿定位

**创新程度自评**：领域内**中上（约前 30–40%）**；达到 IROS / ITSC / IEEE RA-L / T-IV 及 SCI Q1–Q2 门槛；不够 CVPR/NeurIPS 的"新洞察"标准。领域门槛认知：VoD 头部已饱和（44.9→54.6），单模块增益普遍收缩到 +0.5~+2；2026 审稿人默认要求 mean±std 与 bootstrap；三类被认可的创新——表示替换、物理信息利用、训练范式——本文属"物理信息利用"。

**条件投稿决策树**（在 Cycle 1/2 结果出来后择路，避免提前锁死叙事）：

| 实验结果组合 | 叙事主线 | 目标会议/期刊 |
|---|---|---|
| 单域 ΔmAP ≥ 1.0 且过显著性，跨域 Drop 亦缩小 | 时序概率证据融合（单域+跨域双证据） | **T-IV / RA-L**（首选）；若跨域证据完整（TJ4D+K-Radar）冲 **TITS** |
| 单域增益 <1.0 或不过显著性，但跨域 ΔDrop 显著 | **转向鲁棒性叙事**：以时序鲁棒性 + 跨域可靠性为主结果，单域 mAP 为辅 | T-IV（鲁棒性导向）/ RA-L / IROS；标题避免 "Detection" 刷点暗示，突出 Reliable/Robust |
| 单域与跨域均不过显著性 | 触发方案重估：检查 P1 判决链与模块退出标准（收敛方案二 §3/§10） | 暂缓投稿；必要时降级为 ITSC/IROS 短文（效率+消融导向） |
| 跨域 Drop 缩小 + 私有受控数据归因证据（附录 A.8 路径 B） | 鲁棒性 + 独有受控域偏移归因 | 差异化最强，优先冲 **TITS / T-IV** |

**两条不变的原则**：

1. 无论择路，P1 判决链与三 seed 显著性都是前置条件——没有它们，任何叙事都不成立；
2. 跨域叙事一旦作为主线，TJ4DRadSet（必做）与鲁棒性扰动实验（§6）必须先于投稿完成；K-Radar 与私有数据是加分项而非门槛。完整实验设计见**附录 A**。

### 8.10 数字汇报红线

1. 主表一律 mean±std（≥3 seed：42 / 666 / 2023，主实验固定 666）；
2. **禁止**挑选最优 checkpoint 汇报；best-ckpt 只用于定性可视化并声明；
3. test 集（若获得）仅最终评估一次；
4. 引用他人数字标注 reported / reproduced；MAFF-Net 标注蒸馏非公平；
5. 4090/5090 数值与 A4000 不可直接对比，效率以相对延迟或注明硬件；
6. 消融每项 ≥2 run；
7. **跨域数字**：相对 Drop 汇报时必须同时给源域 AP 与目标域 AP 原值，禁止只报百分比；私有数据集须声明 proprietary 并给出统计特征（见附录 A.8）。

---

## 附录 A：跨域 Drop 与跨数据集实验全案

> **附录定位**：§8.9 判断"真正机会是跨域相对 Drop 缩小"，但截至 r0 该方向**尚未启动**（TJ4DRadSet 适配代码不存在、K-Radar 未开始）。本附录把该判断落成可执行方案：度量定义与意义、机制分析、实验设计、数据集选型、TJ4DRadSet 适配清单、K-Radar 提取规则、私有/自采数据协议模板。对应收敛方案二 §7、Cycle 3 任务 3.3 / 3.6。

### A.1 含义与度量定义

**跨域 Drop（Cross-domain Drop / 相对性能下降）**：在**源域**（训练数据集/环境）上训练的模型，直接迁移到**目标域**（不同数据集/传感器/天气/场景）后的相对性能损失。

$$
\text{Drop}=\frac{AP_{\text{source}}-AP_{\text{target}}}{AP_{\text{source}}}
$$

- $AP_{\text{source}}$：模型在自身训练集对应测试集上的精度；
- $AP_{\text{target}}$：**同一模型、不再训练**，直接迁到目标域的精度（zero-shot transfer）；
- **Drop 越小，泛化/鲁棒性越好**。

三条理解要点：

1. **不看目标域绝对 mAP**——不同数据集难度、标注协议、类别集合不同，绝对值无法公平比较；相对 Drop 用模型自己做归一化，绕开了"目标域天生更难"的陷阱。
2. **必须是 zero-shot 迁移**——一旦在目标域上微调，Drop 衡量的是"适应能力"而非"泛化能力"，二者要在论文里分开表述（微调可作补充实验，见 A.4.1 的 CS-7）。
3. **Drop 缩小 ≠ 目标域 SOTA**——论文主张是"相对基线，PRISM 的跨域退化更小"，不是"在目标域刷到第一"。

### A.2 为什么是"真正机会"（意义）

1. **更容易过显著性检验**：VoD 单域已饱和（44.9→54.6），单模块增益收缩到 +0.5~+2，而 seed 间 σ≈1 mAP，+1 以下难过 bootstrap；相对 Drop 衡量"抗退化能力"，受 seed 方差影响更小、效应量更易做出（assessment §0、§5.3）。
2. **是 PRISM 物理正确性的间接证明**：DAUT 建模的是 Doppler 可观性这一**跨雷达共性**而非数据集统计捷径，STER 在目标域噪声更多时作用更大——跨域 Drop 缩小反过来支撑"抓住了物理本质"的主张，使创新一、二超越刷点工具。
3. **真实部署刚需**：跨地区/跨天气/跨雷达型号运行是自动驾驶常态，单数据集高分模型无部署价值。
4. **Q1 期刊隐性门槛**：T-IV/RA-L 尤其看重多数据集与鲁棒性（assessment §2.2）。

**量化预期**（assessment §5.3，仅作规划目标，不得提前写入摘要）：

| 跨域场景 | 预期相对 Drop 缩小 |
|---|---|
| VoD → TJ4DRadSet | 10–25% |
| K-Radar 恶劣天气 | 15–30% |

### A.3 域偏移来源与 PRISM 作用机制

| 域偏移来源 | 具体表现 | PRISM 的对应机制 |
|---|---|---|
| 传感器差异 | 雷达型号、天线数、点云生成算法、点密度、RCS 标定 | DAUT 依赖物理可观性而非特定密度分布；RAPR 证据质量门自适应低密度 |
| 场景差异 | 城市/高速、道路结构、目标尺寸分布 | CRLF 门控在证据不足时退化单帧，避免场景误配 |
| 天气 | 雨/雾/雪引入杂波、ghost、衰减 | STER 抑制无支持回波；RADAR_AUG 显式模拟 |
| 统计差异 | 点密度/RCS/Doppler 噪声分布 | RADAR_AUG + 一致性损失 $L_{inv}$ 学习不变表征 |

**叙事逻辑**：跨域实验不只是"多测一个数据集"，而是回答"PRISM 的建模是否物理泛化"。因此论文应把**跨域 Drop 缩小**与**鲁棒性扰动实验（§6）**互相印证：前者是真实域偏移，后者是受控域偏移，两条证据链共同支撑可靠性主张。

### A.4 跨域 Drop 实验设计方案

#### A.4.1 核心实验矩阵（跨传感器，VoD ↔ TJ4DRadSet）

| ID | Train | Test | 模型 | 作用 |
|----|-------|------|------|------|
| CS-1 | VoD | VoD | RadarPillars-5f | 源域基线 |
| CS-2 | VoD | VoD | PRISM-S | 源域方法 |
| CS-3 | VoD | TJ4D | RadarPillars-5f | 基线跨传感器 Drop |
| CS-4 | VoD | TJ4D | PRISM-S | **方法跨传感器 Drop（核心主张）** |
| CS-5 | TJ4D | TJ4D | RadarPillars-5f | 目标域自训练对照（证明目标域可学） |
| CS-6 | TJ4D | TJ4D | PRISM-S | 方法在目标域有效性（非 VoD 过拟合） |
| CS-7（可选） | VoD→TJ4D 微调 | TJ4D | PRISM-S | 微调上界，与 zero-shot 分开表述 |

**核心论断**：$\Delta\text{Drop}=\text{Drop}_{\text{CS-3}}-\text{Drop}_{\text{CS-4}}>0$，即 PRISM 的跨域退化显著小于基线。CS-5/CS-6 用于排除"目标域本身太难"的混淆。

#### A.4.2 天气矩阵（K-Radar，若启用）

| ID | Train | Test | 说明 |
|----|-------|------|------|
| W-1 | normal | normal | 天气内对照 |
| W-2/3/4 | normal | rain / fog / snow | 单天气迁移 |
| W-5 | all-weather | adverse-weather | 全天气→恶劣天气 |

每个 W 单元同时跑 RadarPillars 与 PRISM-S，比较 Drop。

#### A.4.3 指标体系

| 指标 | 定义 | 用途 |
|---|---|---|
| Relative Drop（主） | $(AP_s-AP_t)/AP_s$ | 主汇报 |
| Absolute Drop | $AP_s-AP_t$ | 辅助 |
| **ΔDrop**（核心） | $\text{Drop}_{base}-\text{Drop}_{PRISM}$ | 论文主张量 |
| 分类别 Drop | 按 Car/Ped/Cyc 分别计算 | 分析哪类受益 |
| 分距离 Drop | 近/中/远分段 | 远距通常 Drop 更大，PRISM 应更明显 |

#### A.4.4 协议对齐规则（红线）

1. **只在交集类别**评估（Car/Ped/Cyc 共同部分）；
2. IoU 阈值、BEV 范围、最大距离尽量统一；无法统一时**分别报告两数据集各自官方协议**，禁止混算；
3. 归一化参数只用训练集统计；
4. 不得在目标域上调参/选 checkpoint；
5. 若协议无法严格统一，全文称 **cross-sensor transfer study**，**禁用 domain generalization** 措辞。

#### A.4.5 统计检验

- 每个矩阵单元 ≥3 seed（42/666/2023）；
- 目标域逐帧 AP → 1000 次 bootstrap → Drop 的 95% CI；
- 对 $\Delta\text{Drop}$ 做配对检验（基线 vs PRISM 同 seed 配对），$p<0.05$ 判定显著；
- 与主实验一致：mean±std，禁止挑最优 ckpt。

#### A.4.6 产出物

| 产出 | 载体 |
|---|---|
| TJ4DRadSet 跨传感器结果 | Table 6（收敛方案二 §4） |
| K-Radar 天气鲁棒性 | Table 7 / 鲁棒性表 |
| 跨域 Drop 对比柱状图（基线 vs PRISM，按场景） | Figure（见 §8.7 图⑨） |
| Drop–距离曲线 | Figure / 补充材料 |

### A.5 数据集选型与推荐

| 数据集 | 规模 | 天气 | 数据形式 | 与 PRISM 适配 | 推荐 |
|---|---|---|---|---|---|
| **View-of-Delft (VoD)** | 8682 帧 | 良好 | 点云 | 已用，主开发集 | —（已选） |
| **TJ4DRadSet** | 7757 帧 / 44 序列 | 良好 | 点云 + **track ID** | 首选第二数据集；track ID 可做时序稳定性 | ⭐⭐⭐⭐⭐ **必做** |
| **K-Radar** | ~35k 帧 | **7 种天气** | **4D radar tensor** | 天气鲁棒性关键；需自建 tensor→点管线 | ⭐⭐⭐⭐ 强烈推荐（可选） |
| MAN TruckScenes | 中 | 良好 | 6 雷达 360° | 多雷达/重卡，处理复杂 | ⭐⭐ 后续工作 |
| Astyx HiRes2019 | 小 | 良好 | 点云 | 仅单帧，不适合时序 | ⭐ 不用于跨域 |
| RadarScenes | 大 | — | 经典雷达（非 4D 成像） | 与 4D pillar 路线不符 | ⭐ 不适用 |

**分层结论**：维持收敛方案二 §7 的"TJ4DRadSet 必做 + K-Radar 可选"。
- **TJ4DRadSet**：点云格式、有 track ID、规模适中、适配成本最低，且可复现性好——优先做；
- **K-Radar**：独一无二的天气维度，是鲁棒性叙事最有力证据，但 tensor 预处理工程负担重——建议 TJ4D 跑通且投稿目标为 T-IV/TITS 时再上；
- **自采/私有数据**：见 A.8，定位为差异化加分项而非主实验集。

### A.6 TJ4DRadSet 适配方案

> 目标：把 TJ4DRadSet 接入 OpenPCDet 数据管线，产出与 VoD 一致的 `infos + gt_database`，支撑 CS-3~CS-7。以下为实施清单；具体坐标约定/类别表/扫描频率以 TJ4DRadSet 官方文档为准，落地前逐项核验。

**① 目录与文件**

```text
pcdet/datasets/tj4dradset/
├── __init__.py
├── tj4dradset_dataset.py      # OpenPCDet Dataset 子类
├── tj4dradset_utils.py        # 读取/坐标变换/类别映射
└── sequence_loader 集成        # 复用 VoD 时序加载逻辑（真实 Δt、ego 对齐）

tools/cfgs/dataset_configs/tj4dradset_dataset_radar.yaml
tools/cfgs/tj4d_models/prism_pillars_rf.yaml
```

**② 坐标系统一**：把 TJ4D 原始坐标系变换到项目统一系（前 x、左 y、上 z）。先核对官方坐标约定，再写刚体变换（旋转 + 可能的轴交换），并用可视化 + `tests` 断言验证。

**③ 类别映射表**（只在交集类别评估）：

```text
TJ4DRadSet 原始类别 → 统一类别
  Car / Vehicle      → Car
  Pedestrian         → Pedestrian
  Cyclist            → Cyclist
  其余（如 Tricycle 等）→ 评估时剔除，映射表存档
```

**④ 真实 Δt 与序列**：按 TJ4D 扫描频率把帧号换算为秒级 Δt；沿用 sequence 级划分，禁止随机按帧划分；track ID 建立连续帧索引（仅用于时序稳定性实验，主实验用原始标签）。

**⑤ 点云范围 / 体素**：TJ4D 检测范围与 VoD 不同，`POINT_CLOUD_RANGE`、`VOXEL_SIZE` 按 TJ4D 官方范围重设，不照搬 VoD 的 `[0,-25.6,-3,51.2,25.6,2]`。

**⑥ info 与 gt-database**：生成 `tj4dradset_infos_{train,val,test}.pkl` + `tj4dradset_dbinfos_train.pkl`；主实验同样**禁用 gt_sampling**（时序公平）。

**⑦ 验收 checklist**：

- [ ] 坐标系变换后点云与标注框可视化对齐；
- [ ] train/val 序列不重叠（因果、无泄漏）；
- [ ] Δt 符号通过 `tests/test_time_sign.py` 同类断言；
- [ ] 1 epoch 冒烟训练 → 评估 → AP 输出全链路通过；
- [ ] 类别映射表与剔除规则写入配置并存档。

### A.7 K-Radar tensor→点固定提取规则设计

K-Radar 主数据为 **4D radar tensor**（range × azimuth × elevation × Doppler 张量），与点云型 RadarPillar 不同，必须先固定"tensor→点"提取规则，且**该规则一经确定不得按天气调整**，否则污染 domain shift 实验。

**提取管线（固定一次，全天气共用）**：

```text
4D radar tensor
   │  (1) 固定 CFAR / 峰值检测（阈值一次设定）
   ▼
候选检测点 (range, azimuth, elevation, Doppler, power)
   │  (2) 固定功率/信噪比下限（全天气同一阈值）
   ▼
球坐标 → 车体直角坐标 (x, y, z)
   │  (3) 统一点格式 [x,y,z,log_rcs,v_rel,v_comp,delta_t,range,sin_az,cos_az,...]
   ▼
统一点云（接入 PRISM 数据管线）
```

**规则红线**：

1. CFAR 阈值、功率下限、最大距离、角度范围**只在一套固定参数下运行**，禁止按天气/序列单独调；
2. 优先复用 K-Radar 官方已提供的点云表示（若可用），避免自研提取引入不可比因素；自研则须把提取代码随论文开源；
3. 保留 power、Doppler、range、azimuth、elevation 原始量，供后续复算；
4. 天气划分按 K-Radar 官方 sequence 标签，禁止同一录制序列跨 train/test；
5. 提取后抽样可视化核对（雨/雾/雪下点云形态合理，无明显伪点堆积）。

### A.8 私有/自采数据集实验协议模板

自采/私有数据可成为论文独特卖点（受控域偏移、别人没有的数据），但须遵守与公开数据相同的协议。三条现实路径：

| 路径 | 适用 | 价值 | 风险 |
|---|---|---|---|
| **A 自采**（实验室/合作车装 4D 雷达） | 有车有设备 | 可主动设计受控域偏移（同雷达跨场景 / 同场景跨雷达），归因最干净 | 标注贵、周期长、≥1000 帧才有统计意义 |
| **B 私有合作数据**（车企/雷达厂商） | 有合作关系 | 常能获得不同型号雷达同场景数据，天然构成跨传感器对，性价比最高 | 不可公开，需 proprietary 声明 + 统计特征背书 |
| **C 仿真合成**（CARLA 等） | 无硬件 | 成本最低 | 4D 雷达仿真保真度不足，仅作 RADAR_AUG 延伸验证，不作主跨域证据 |

**协议模板（落地时逐项填写，与 A.4.4 红线一致）**：

```text
[数据集名称 / 代号]：__________________________
[来源路径]：A 自采 / B 私有合作 / C 仿真
[雷达型号与参数]：__________________________
[帧数 / 序列数]：________ / ________
[场景与天气]：__________________________
[坐标系约定 → 统一系变换]：__________________________
[标注方式与类别映射]：__________________________
[真实 Δt 来源（时间戳/频率）]：__________________________
[train/val/test 序列级划分]：__________________________
[归一化参数（仅训练集统计）]：__________________________
[是否公开 / proprietary 声明]：__________________________
[域偏移实验角色]：受控跨传感器 / 受控跨场景 / 天气 / 其他
```

**论文表述要求**：

- 私有数据写明 "proprietary dataset, details cannot be released"，并给出统计特征（点密度、RCS 分布、目标类别占比）让审稿人信服；
- 私有/自采结果**必须与至少一个公开数据集结果并存**，保证可复现背书；
- 若自采数据构成"受控域偏移"（同雷达跨场景或同场景跨雷达），应明确这是公开数据集无法提供的归因证据，作为差异化亮点突出。

---

*维护说明：本文档随实验推进更新；r0 之后的每次重要训练结果应回填 §7 并复核 §7.2 的可靠性分级。附录 A 随 TJ4DRadSet/K-Radar 适配推进更新，A.6/A.7 的 checklist 完成后在对应项打勾并记录实测协议参数。*
