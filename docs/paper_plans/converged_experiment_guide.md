# PRISM-Pillars-RF 收敛方案（二）：实验开展指南与后续路线

<div align="center">

**当前状态 · 分阶段实验协议 · 消融设计 · 统计与效率协议 · 风险预案**

版本 1.0 · 2026-08-09 · 收敛自 `paper_plans/` 六份方案文档与 `Train_reports/` 三份训练报告

</div>

> **文档地位**：本文档是实验执行层面的唯一收敛参考，与 `converged_paper_plan.md`（贡献与方法）配套。
> 验收阈值的权威来源是 `project_constraints.md`；训练/评估/部署命令的权威来源是 `Train_guide.md`；远端环境与平台信息见 `remote_environment.md`。本文档与上述文档冲突时，以上述文档为准。

---

## 目录

- [1. 当前状态快照](#1-当前状态快照)
- [2. 实验体系总览与依赖关系](#2-实验体系总览与依赖关系)
- [3. 分阶段协议 P0–P8](#3-分阶段协议-p0p8)
- [4. 主实验表设计（Table 1–6）](#4-主实验表设计table-16)
- [5. 消融实验矩阵](#5-消融实验矩阵)
- [6. 鲁棒性实验](#6-鲁棒性实验)
- [7. 跨数据集实验](#7-跨数据集实验)
- [8. 统计显著性协议](#8-统计显著性协议)
- [9. 效率测量协议与预算](#9-效率测量协议与预算)
- [10. 模块退出标准速查](#10-模块退出标准速查)
- [11. 近期行动路线（Cycle 收敛）](#11-近期行动路线cycle-收敛)
- [12. 风险清单与缓解](#12-风险清单与缓解)

---

## 1. 当前状态快照（截至 2026-08-09）

### 1.1 已完成

| 事项 | 结果 | 备注 |
|------|------|------|
| 双平台环境（5090 / 4090D） | ✅ 30 项单元测试通过，算子编译完成 | 详见 `remote_environment.md` |
| VoD 数据部署（8682 帧，train 5139 / val 1296） | ✅ infos 与 GT 库已生成 | test 2247 帧无标签 |
| RadarPillar 基线训练（60 ep, bs=16, 5090） | ✅ bbox R40 mAP 58.07；3D R40 47.21 | `round1_report.md`；非严格 P0 协议（未跑 1/3/5 帧变体与 3 seed） |
| PRISM-Pillars-RF-S 完整训练 r0（80 ep, bs=4, 4090D） | ✅ 3D mAP 53.29（std）/ 52.21（R40）@ ep79 | 四 PRISM 模块 + RepDWC + Lite-MDFEN 联合；超预测上界 |
| Cycle 1 代码 | ✅ cdist 分块、雷达增强、分阶段训练、AMP bf16、bs=8、每 10 epoch 评估 | commits ec92570 → cb78f4b，**尚未训练验证** |

### 1.2 未完成（按优先级）

| 事项 | 性质 | 优先级 |
|------|------|--------|
| **同协议公平对照**（基线 80 ep、同增强、3 seed） | 论文主表前提 | 🔴 P0 |
| **P1 时序基线判决链**（全方案生死线） | 创新一成立的核心证据 | 🔴 P0 |
| 逐模块消融隔离（Table 2 完整链） | 贡献归因 | 🔴 P0 |
| Cycle 1 重训（bs=8 + 雷达增强 + 分阶段） | 增益挖掘 | 🟡 P1 |
| 可靠性诊断（Spearman、q 分布、坍缩监控） | Table 5 素材 | 🟡 P1 |
| 鲁棒性扰动实验 | Table / 曲线 | 🟡 P1 |
| 效率基准 + RepDWC 部署转换与等价性测试 | 验收阶段 D | 🟡 P1 |
| TJ4DRadSet 数据集开发（`pcdet/datasets/tj4dradset/` 不存在） | Table 6 | 🟡 P2 |
| 统计显著性（3 seed + bootstrap） | 验收阶段 C | 🔴 随主实验 |
| C 版（C=48 + CenterHead + 5 帧）与联合微调 | 性能上限 | ⚪ P3 |
| K-Radar 恶劣天气 | 可选扩展 | ⚪ 可选 |

---

## 2. 实验体系总览与依赖关系

```
P0 基线复现 ──> P1 时序基线判决链 ──> P2 CRLF 单验 ──> P3 STER ──> P4 可学习Σ
   (硬门槛)        (硬门槛·生死线)                                    │
                                                                      ▼
                    P5 RepDWC ──> P6 Lite-MDFEN ──> P7 检测头 ──> P8 联合微调
                    (可与 P2-P4 并行)
```

- **P0、P1 是硬门槛**：不通过则整个方案停止或转向（§3 失败预案）。
- P5–P7 每个借鉴模块带独立退出标准（§10），触发退出是**预期行为**，照实写入论文。
- 现状修正：r0 以联合训练方式"跳级"跑通了 P2–P4 + P5–P6 的全组合，**但判决链与消融证据缺失**——后续实验的首要任务不是继续堆模块，而是**补齐证据链**。

---

## 3. 分阶段协议 P0–P8

### P0：RadarPillars 严格复现（部分完成，需补严格版）

| 项 | 内容 |
|----|------|
| 配置 | `tools/cfgs/vod_models/vod_radarpillar.yaml` |
| 帧数变体 | 1 / 3 / 5 帧（`--set DATA_CONFIG.NUM_SWEEPS {1,3,5}`） |
| 训练 | bs=16，60 ep，`--fix_random_seed`（主 seed 666）；5 帧变体补 seed 42 / 2023 |
| 成功标准 | 5 帧 mAP 与复现锚点 48.76（或论文 50.70）偏差 ≤ 1.0；记录实测 3-seed σ |
| 失败处理 | 偏差 >1.5 → 排查数据管线（增强、类别映射、IoU 阈值、NMS 0.10、R40 协议）后重跑，不得继续 |

**公平对照补充（新增，r0 暴露的需求）**：为与 PRISM-S 的 80 ep 结果直接可比，基线需追加一组 **80 ep、禁用 gt_sampling、同几何增强** 的 5 帧训练（3 seed）。这是主表 Table 1 中 RadarPillars 行的来源。

### P1：时序基线判决性实验（全方案生死线，未执行）

固定 q=1、固定 σr=0.10 / σt=0.50、无 temporal attention、标准 RadarPillars 骨干：

| # | 方法 | 实现要点 | 预期 |
|---|------|---------|------|
| 1 | naive 累积 | 历史帧直接拼接 | 基线 |
| 2 | ego-motion 对齐 | 历史点经 pose 变换到当前坐标系 | > #1 |
| 3 | 确定性 Doppler 补偿 | μ = p + Δt·v_comp·u，hard assignment | > #2 |
| 4 | 各向同性高斯路由 | Σ = σ²I 软路由 | > #3 |
| 5 | 固定各向异性路由 | Σ = σr²uuᵀ + σt²nnᵀ，σt>σr | > #4 |

**成功标准**：mAP 严格单调 #1 < #2 < #3 < #4 < #5（允许并列，不允许倒挂超过 seed 噪声 0.3）。每个变体 ≥2 seed；#3 与 #5 各 3 seed（直接进论文 Table 2）。

**判决与失败预案**：

- 单调成立 → 核心假设成立，论文主张一落地；
- #3 ≤ #2（确定性补偿无效）→ VoD 上运动错位不是主要矛盾：转查数据（历史帧是否已预补偿、Δt 符号——先跑 `tests/test_time_sign.py`）；
- #5 ≤ #4（各向异性无效）→ 创新一空心化：降级为"可靠性 + 局部融合"主线重估，或终止方案；
- 注意：即使 r0 全模型结果好，P1 仍必须跑——它是论文论证链（Table 2）而非调参环节。

### P2：CRLF 独立验证（消融补做）

```bash
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.RELIABILITY.ENABLED False MODEL.DOPPLER_TUBE.LEARNABLE False \
            MODEL.LITE_MDFEN.ENABLED False \
    --extra_tag p2_crlf_only
```

对照组：同配置再关 `TEMPORAL_FUSION.ENABLED False`（≈ P1#5 水平）。成功标准：ΔmAP ≥ 0.5 且延迟增量 ≤ 15%，否则触发 CRLF 退出（回退朴素拼接）。

### P3：STER 可靠性（消融补做）

分阶段训练见收敛方案一 §6.2。验证指标（Table 5 素材）：Spearman(q, s)、GT 内/外平均 q、motion-tail 平均 q、q 分桶后的 FP 变化、mean(q) 坍缩监控。退出条件：训练不稳定或 ΔmAP < 0.3 → 回退解析可靠性 $q=\exp(-\eta_1 d-\eta_2 e)$。

**必做负对照**：q=1、random q、无 L_rel 的 learned q、仅 BCE、BCE+rank、+ghost aug、以前景概率替代可靠性。

### P4：可学习 Σ（r0 已含，补对照）

r0 直接启用 learned σ 且稳定——但缺"固定 σ vs learned σ"对照，无法归因。补一组固定 σ（0.10/0.50）全模型对照即可。监控 σ 触界频率；退出条件：不稳定或频繁触界 → 回退解析增长 $s_r=0.10+0.15|\Delta t|,\ s_t=0.50+0.50|\Delta t|$。

### P5–P7：借鉴模块验证（带退出标准）

| 阶段 | 内容 | 对照 | 退出条件 | 回退 |
|------|------|------|---------|------|
| P5 | RepDWC 骨干 | Dense Conv / 普通 DWConv | ΔmAP < −0.3 | BaseBEVBackbone |
| P6 | Lite-MDFEN（单 DCNv3 + raw bypass） | 无 MDFEN；DCN 位置消融（高/中/低分辨率、2×DCN、无 bypass） | ΔmAP < 0.5 或 ΔLatency > +10% | 仅保留 RepDWC |
| P7 | CenterHead（+IoU/dIoU） | AnchorHeadSingle | ΔmAP < 0.5 或 ΔLatency > +10% | AnchorHead |

P5 必须附部署态等价性测试（`tests/test_rep_parameterization.py`，max_diff < 1e-4；FP16 放宽 1e-3）。

### P8：联合微调与最终模型

- 全模块联合微调 10–20 epoch，base LR ≤ 1e-4，backbone 0.5×。
- 产出两个投稿配置：
  - **PRISM-Pillars-RF-S**：C=32、3 历史帧、AnchorHead —— 公平性与效率证明；
  - **PRISM-Pillars-RF-C**：C=48、5 历史帧、CenterHead + IoU/dIoU —— 性能上限。
- 主表同时报告 S 与 C。

---

## 4. 主实验表设计（Table 1–6）

| 表 | 内容 | 来源阶段 |
|----|------|---------|
| **Table 1** | VoD 主结果：PointPillars / RadarPillars 1/3/5f（含 80ep 公平版）/ RadarNeXt / MAFF-Net*（引用，注明蒸馏）/ SGE-Flow / PRISM-S / PRISM-C；列含 Car/Ped/Cyc、EAA mAP、Params、GFLOPs、FPS、P95、显存 | P0+P8 |
| **Table 2** | 时序证据建模判决链：P1 五变体 + learned Σ + reliability + full PRISM；按静态/动态、近/中/远分段报告 | P1–P4 |
| **Table 3** | Backbone/Neck/Head 组合矩阵（Dense/DWConv/RepDWC × 有无 MDFEN × Anchor/Center） | P5–P7 |
| **Table 4** | MDFEN 消融：FPN/PAN/无 DCN/DCN 位置（高中低）/双 DCN/无 bypass/final | P6 |
| **Table 5** | 可靠性消融（§5.2 负对照全集）+ Spearman、ECE、GT 内外 q、FP | P3 |
| **Table 6** | TJ4DRadSet 第二数据集（分别训练与迁移） | §7 |

---

## 5. 消融实验矩阵

### 5.1 时序证据链（对应 Table 2，论证骨架）

必须依次证明：

$$
\text{确定性补偿} < \text{各向异性概率路由} < \text{概率路由+可靠性} < \text{完整 PRISM}
$$

### 5.2 可靠性负对照（对应 Table 5）

```text
q = 1 ｜ random q ｜ learned q without L_rel ｜ BCE only ｜ BCE + ranking
｜ BCE + ranking + ghost aug ｜ foreground probability instead of reliability
```

### 5.3 Routing 消融

```text
hard assignment ｜ isotropic Gaussian ｜ anisotropic Gaussian
｜ remove sigma_t >= sigma_r ｜ use v_rel instead of v_comp
｜ fixed sigma ｜ learned sigma ｜ 两步归一化 vs 单步归一化（证明修正必要）
```

### 5.4 Attention 消融

```text
direct addition ｜ 3x3 conv fusion ｜ global attention ｜ local attention
｜ feature-only score ｜ + Mahalanobis bias ｜ + reliability bias
｜ + time decay ｜ + gate（完整）
```

### 5.5 骨干/颈/头组合（对应 Table 3/4）

见 §3 P5–P7；每项 ≥2 run。

---

## 6. 鲁棒性实验

| 扰动 | 强度梯度 |
|------|---------|
| 随机点丢失 | 10% / 30% / 50% |
| Doppler 噪声 | 0.1 / 0.3 / 0.5 m/s |
| 自车速度偏置 | 0.2 / 0.5 / 1.0 m/s |
| 历史帧丢失 | 1 / 2 / 3 sweeps |
| RCS 缩放 | 0.8 / 1.0 / 1.2 |
| Ghost 注入 | 5% / 10% / 20% |

对比对象：RadarPillars-5f、PRISM-S（可选 RadarNeXt 引用值）。
产出曲线：扰动强度–mAP / Recall / FP；历史帧数–mAP / Latency。

评估在**训练时未见扰动**上进行；雷达增强开启的模型与关闭的模型各报一组，以分离"增强带来的鲁棒性"与"概率建模自带的鲁棒性"。

---

## 7. 跨数据集实验

### 7.1 TJ4DRadSet（必做，Table 6）

前置开发（当前不存在）：`pcdet/datasets/tj4dradset/`——坐标系统一（前 x、左 y、上 z）、track ID 索引、真实 Δt、类别映射（Vehicle/Pedestrian/Cyclist）、OpenPCDet info 与 gt-database。

实验：① 从头训练（验证模块非 VoD 过拟合）；② VoD→TJ4D 迁移；③ 利用 track ID 做同目标跨帧预测稳定性。

**禁止**因类别协议差异直接声称严格跨域泛化；只在交集类别评估。无法统一协议时称为 **cross-sensor transfer study**。

### 7.2 K-Radar（可选扩展）

Normal → Rain / Fog / Snow；all-weather → adverse-weather。主数据为 4D radar tensor，需固定 tensor→点提取规则（不得按天气分别调阈值，否则污染 domain shift 实验）。报告相对 Drop =（AP_source − AP_target）/ AP_source。

### 7.3 VoD track-id 标签使用约束

`label_2_with_track_ids`（6435 帧）仅用于时序一致性/track 稳定性实验；**主实验与 P0 一律使用原始 label_2**，禁止混用。

---

## 8. 统计显著性协议

1. 基线与 PRISM 各 ≥3 seed（42 / 666 / 2023，主实验固定 666）；
2. val 集逐帧 AP 差异 → 1000 次 bootstrap → ΔmAP 95% CI；**判据：CI 下限 ≥ 0.5 且 p < 0.05**；
3. 3 seed mAP 标准差目标 ≤ 0.30；若实测基线 σ≈1（领域观察值），论文以实测值为准并加大 seed 数到 5；
4. 消融每项 ≥2 run，报告均值±标准差；
5. **禁止**挑选最优 checkpoint 汇报；test 集（若获得）仅最终评估一次；
6. 验收口径（project_constraints §3.4）：主指标为 EAA、R40、3 次运行均值。

> r0 的 53.29 目前是单 seed、仅 ep70–80 评估——在按本协议复证之前，不得作为论文主表数字。

---

## 9. 效率测量协议与预算

### 9.1 测量规范

- 工具：`tools/benchmark_latency.py`；CUDA sync、100 warmup、≥1000 iter；FP32 主报告 + FP16 补充；batch 1 与 4；
- 分模块计时：数据加载 / 自车对齐 / VFE+Attention / STER+DAUT+RAPR / CRLF / RepDWC / MDFEN / Head / 后处理；
- 部署态：先 `tools/convert_to_deploy.py --validate` 再测延迟；
- DCNv3 若需自定义 ONNX/TensorRT 算子，论文不得仅凭 PyTorch FPS 宣称边缘部署；
- 4090/5090 数值与 A4000 不可直接对比，论文注明硬件或以相对延迟表述。

### 9.2 预算与目标（验收口径，详见 project_constraints §4）

| 指标 | 最低要求 | 目标 |
|------|---------|------|
| FPS（A4000 当量，FP32, bs=1, 部署态） | ≥ 60.4（RadarNeXt 的 90%） | ≥ 64.0 |
| P95 延迟 | ≤ 23.0 ms | ≤ 21.0 ms |
| 参数量（S 版部署态） | ≤ 0.40M | ≤ 0.35M |
| GFLOPs（S 版） | ≤ 3.0 | ≤ 2.5 |
| 时序模块（STER+DAUT+RAPR+CRLF）延迟预算 | ≤ 7.0 ms 合计 | — |

延迟优化备选：TOPK 16→8（时序加速 ~15%，mAP 损失 <0.5）；`LITE_MDFEN.ENABLED: false`（省 ~2ms）。

---

## 10. 模块退出标准速查（权威版本在 project_constraints §6）

| 模块 | 退出条件 | 回退方案 | 验证阶段 |
|------|---------|---------|---------|
| RepDWC | ΔmAP < −0.3 vs Conv2D 骨干 | BaseBEVBackbone | P5 |
| Lite-MDFEN | ΔmAP < 0.5 或 ΔLatency > +10% | 仅 RepDWC | P6 |
| STER | 不稳定或 ΔmAP < 0.3 | q=1 或解析 q | P3 |
| DAUT 可学习 Σ | 不稳定或 σ 频繁触界 | 固定/解析 σ | P4 |
| RAPR | ΔmAP < 0.3 vs 确定性最近邻路由 | 确定性 Pillar 分配 | P2 |
| CRLF | ΔmAP < 0.5 或 ΔLatency > +15% | 朴素拼接/平均 | P2 |
| CenterHead | ΔmAP < 0.5 或 ΔLatency > +10% | AnchorHead | P7 |

判定流程：带/不带目标模块各 3 seed → ΔmAP 均值−标准差 < 阈值或延迟超标 → 触发退出 → 回退重训 → 记录决策（论文照实写）。

---

## 11. 近期行动路线（Cycle 收敛）

> 以 r0 报告 §6 的 Cycle 计划为基础，按"证据链优先于刷点"原则重排。

### Cycle 1（进行中）：公平对照 + 判决链 + 增强验证

| # | 任务 | 产出 | 预计 GPU 时（4090D） |
|---|------|------|---------------------|
| 1.1 | 基线公平对照：RadarPillars 5f @ 80 ep、禁 gt_sampling、seed 666/42/2023 | Table 1 基线行；核验 r0 增益真实性 | ~6 h |
| 1.2 | **P1 判决链**五变体（q=1, 固定 σ），#3/#5 各 3 seed、其余 2 seed | Table 2 骨架；创新一生死判决 | ~24 h |
| 1.3 | Cycle 1 重训：PRISM-S @ bs=8 + 雷达增强 + 分阶段训练（代码已就绪） | 与 r0 对比，量化增强与分阶段增益 | ~10 h |
| 1.4 | r0 复证：PRISM-S @ bs=8 无增强联合训练，3 seed | 主表 PRISM-S 行候选 | ~30 h |

### Cycle 2：消融隔离 + 诊断

| # | 任务 | 产出 |
|---|------|------|
| 2.1 | Table 2 完整链（fixed σ / learned σ / +reliability / +fusion） | 贡献归因 |
| 2.2 | Table 5 可靠性负对照全集 + Spearman/ECE/q 分布诊断 | 创新二证据 + 可视化素材 |
| 2.3 | Routing/Attention 消融（§5.3/5.4）+ 超参扫描（OneCycle PCT_START/DIV_FACTOR、λ_rel/λ_σ/λ_inv） | 敏感性说明 |
| 2.4 | 固定 σ vs learned σ 全模型对照（P4 归因） | Table 2 补充 |

### Cycle 3：架构升级 + 鲁棒性 + 第二数据集

| # | 任务 | 产出 |
|---|------|------|
| 3.1 | P5–P7 骨干/颈/头消融（Table 3/4）+ 部署转换与等价性测试 | 工程组件验收 |
| 3.2 | 鲁棒性扰动实验（§6） | 鲁棒性曲线 |
| 3.3 | TJ4DRadSet 数据集开发 + 训练/迁移（Table 6） | 跨传感器证据 |
| 3.4 | 效率基准（§9） | Table 1 效率列 |
| 3.5 | C 版（C=48 + CenterHead + 5 帧）+ P8 联合微调 | 性能上限 |
| 3.6 | （可选）K-Radar 恶劣天气 | 扩展验证 |

**GPU 预算**：合计约 150–180 个 4090-小时（≈ assessment §22 的 170 h 估算，其中 P0/部分 P5-P8 已由 r0 与 round1 覆盖一部分）；按周节奏 W1 跑完 Cycle 1 的 1.1+1.2（判决链优先），其余顺次推进。

---

## 12. 风险清单与缓解

| # | 风险 | 概率 | 缓解 |
|---|------|------|------|
| 1 | 提升不过 seed 方差（显著性失败） | 中 | ≥3–5 seed + bootstrap；叙事转跨域 Drop 与鲁棒性；聚焦动态/远距子集报告 |
| 2 | P1 单调链倒挂（创新一空心化） | 中 | §3 P1 判决树；先验单元测试（Δt 符号、协方差正定、概率守恒）；倒挂时降级为"可靠性+局部融合"主线 |
| 3 | r0 增益被证明部分来自协议差异（80ep/bs/评估期） | **中高** | Cycle 1 的 1.1/1.4 公平对照先行；论文主表只用同协议结果 |
| 4 | HyperDet 叙事撞车 | 高 | 收敛方案一 §8.4/8.8 应对；加速投稿时间窗 |
| 5 | 无 test split | 确定 | val 开发评估 + 声明；后续官方渠道补测 |
| 6 | autodl-tmp 磁盘不足 / 实例回收 | 低-中 | 解压后删 zip（−14.6 GB）；output 定期清理 ckpt（保留 best+last）；重要 ckpt 拉回本机 |
| 7 | 单卡训练中断 | 低 | 关键 run 用 `--ckpt` 断点恢复 |
| 8 | DCNv3 部署链（ONNX/TRT）不可用 | 中 | 论文效率主张限于 PyTorch + 部署态 PyTorch，边缘部署列为 future work |

---

## 附录：常用命令速查

完整命令见 `Train_guide.md` 与 `remote_environment.md` §D。模板：

```bash
# 训练（远端 4090D）
cd /root/PRISM-Pillars
/root/miniconda3/bin/python tools/train.py \
    --cfg_file tools/cfgs/vod_models/<cfg>.yaml \
    --batch_size <B> --epochs <E> --fix_random_seed --extra_tag <tag>

# 评估
/root/miniconda3/bin/python tools/test.py \
    --cfg_file tools/cfgs/vod_models/<cfg>.yaml --batch_size 8 \
    --ckpt output/cfgs/vod_models/<model>/<tag>/ckpt/checkpoint_epoch_<N>.pth

# 批量评估（学习曲线）
/root/miniconda3/bin/python tools/test.py --cfg_file <cfg> --eval_all \
    --ckpt_dir output/.../ckpt --start_epoch 10

# 部署转换与等价性验证
/root/miniconda3/bin/python tools/convert_to_deploy.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt <ckpt> --output deploy_models/prism_s_deploy.pth --validate
```

---

*维护说明：本文档随实验推进更新；每个 Cycle 结束后在 §11 对应行回填实测结果与决策记录，并按 §10 登记任何触发的模块退出。*
