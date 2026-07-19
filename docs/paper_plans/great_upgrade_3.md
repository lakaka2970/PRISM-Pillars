# 一、整合后的项目定位

建议将完整项目命名为：

## **PRISM-Pillars-RF**

**Physics-Guided Reliable Temporal Evidence Fusion with Re-parameterized Foreground Refinement**

推荐论文题目：

> **PRISM-Pillars: Physics-Guided Reliable Temporal Evidence Fusion with Efficient Foreground Refinement for 4D Radar 3D Object Detection**

这里的 **RF** 表示：

* **Reliable temporal evidence Fusion**
* **Re-parameterized Foreground refinement**

最终项目不是简单的：

```text
PRISM + RadarNeXt
```

而是围绕一个明确的方法论构建：

# **先纠正时序证据，再增强空间表征**

即：

[
\text{Evidence Correction}
\rightarrow
\text{Temporal Fusion}
\rightarrow
\text{Efficient Representation Refinement}
]

核心观点为：

> 历史雷达回波一旦经过错误的确定性补偿并进入 BEV，后续再强的卷积或可变形特征增强也无法可靠恢复其真实几何。因此，模型必须先利用 Doppler 可观性、时序支持度和运动不确定性，对历史回波进行概率化修正；随后再使用可重参数化主干和保留原始特征的轻量可变形 Neck，对融合后的残余稀疏前景进行高效增强。

这条主线使 PRISM 和 RadarNeXt 的作用层级清晰分开：

| 阶段          | 解决的问题              | 采用方法                 |
| ----------- | ------------------ | -------------------- |
| 点级历史证据      | 回波错位、ghost、切向速度不可观 | PRISM 概率证据建模         |
| Pillar 时序融合 | 历史信息是否值得使用         | 可靠性感知局部检索            |
| BEV 空间特征    | 前景稀疏、不规则、断裂        | Lite-MDFEN           |
| 高效部署        | 新增时序计算导致延迟上升       | RepDWC 重参数化 Backbone |

RadarPillars 已经证明了补偿径向速度分解、PillarAttention 和均匀通道缩放的有效性，因此这些应作为基础架构继承，而不是重新包装成本文创新。 原始模型采用 (C=32)、约 0.27 M 参数和 1.99 GFLOPs，说明整合方案必须继续控制实际推理开销。

PRISM 原方案已经把历史点定义为“带可靠性权重和 Doppler 各向异性运动不确定性的时序概率证据”，这一点继续作为论文的理论核心。

RadarNeXt 则通过 RepDWC 和 MDFEN 将雷达特征的稀疏噪声处理推迟到多尺度融合阶段，并在 VoD 和 TJ4DRadSet 上展示了精度与实时性平衡。其正式版本报告 VoD 50.48 mAP、TJ4D 32.30 mAP，以及 RTX A4000 上 67.10 FPS、Jetson AGX Orin 上 29.03 FPS。([Springer Nature][1])

---

# 二、最终模型的科学问题

整篇论文围绕三个主要科学问题和一个工程问题展开。

## Q1：历史雷达点能否被当成确定点？

不能。

自车位姿只能补偿传感器运动；Doppler 只直接约束径向运动。历史点在切向方向存在天然不可观性，因此确定性补偿容易产生：

* 动态目标拖影；
* 错误 Pillar 分配；
* 远距离目标结构断裂；
* ghost 和多径回波积累。

本文将历史点建模为各向异性概率证据，而不是确定点。

## Q2：所有历史回波是否同等可信？

不是。

历史点是否值得用于当前检测，取决于：

* 当前帧是否存在空间支持；
* Doppler 是否一致；
* RCS 和局部密度是否异常；
* 历史时间是否过长；
* 自车补偿残差是否过大。

因此需要点级时序可靠性 (q_i)。

## Q3：历史证据应该怎样进入当前帧？

不应直接拼接，也不应使用全部 Pillar 的全局注意力。

当前帧 Pillar 应作为 Query，只从局部历史概率证据中检索，并综合考虑：

* 特征相似度；
* Mahalanobis 几何一致性；
* 可靠性；
* 证据质量；
* 时间衰减。

## Q4：加入概率时序模块后如何保持实时性？

采用：

* RepDWC 可重参数化 Backbone；
* 单 DCNv3 的轻量多路径 Neck；
* 固定局部 Top-K；
* 只对历史点执行概率 Scatter；
* 当前帧继续使用原始高效 Pillar 编码。

RadarNeXt 的 RepDWC 在训练时使用多分支深度卷积、点卷积与 BN 分支，在部署时折叠为单路径结构。([Springer Nature][2]) MDFEN 则只在部分路径上使用 DCNv3，并保留未经修改的原始特征，以降低增强过程中的信息损失。([Springer Nature][2])

---

# 三、最终总体架构

```text
                          ┌──────────────────────────────┐
Current radar frame P_t ─►│ Current Evidence Encoder     │
                          │ Velocity components           │
                          │ PillarVFE                      │
                          │ PillarAttention                │
                          └──────────────┬───────────────┘
                                         │ Current Pillars
                                         │ Query
                                         ▼
Historical frames P_t-k ─► Ego-motion alignment
              │
              ├─ Shared point embedding
              ├─ Self-supervised Reliability Estimator
              ├─ Doppler Anisotropic Uncertainty Tube
              ├─ Reliability-weighted Probabilistic Routing
              └─ Historical Evidence BEV
                                         │ Key / Value
                                         ▼
                          Causal Local Temporal Fusion
                                         │
                                         ▼
                              Fused Current Pillars
                                         │
                              PointPillar Scatter
                                         │
                                         ▼
                          Re-parameterizable RepDWC
                              Three-stage Backbone
                                         │
                                         ▼
                         Single-DCNv3 Lite-MDFEN
                          + untouched raw-feature path
                                         │
                                         ▼
                       AnchorHead / CenterHead candidate
                                         │
                                         ▼
                                  3D detections
```

## 两项架构纪律

### 当前帧不做概率扩散

当前帧坐标是直接观测，不需要因历史运动产生的空间扩散。当前帧保留 RadarPillars 原始编码，防止目标轮廓被平滑。

### DCNv3 不进入物理证据建模之前

正确顺序是：

[
\text{点级物理修正}
\rightarrow
\text{概率时序融合}
\rightarrow
\text{BEV 可变形增强}
]

不能先用 DCNv3 改变空间响应，再把它解释成物理位置或运动不确定性。

---

# 四、模块 A：雷达特征预处理

统一点格式：

```text
[
    x, y, z,
    log_rcs,
    v_rel,
    v_comp,
    delta_t,
    range,
    sin_azimuth,
    cos_azimuth,
    sweep_idx,
    source_sensor_id
]
```

视线单位向量：

[
\mathbf u_i=
\frac{[x_i,y_i]^\top}
{\sqrt{x_i^2+y_i^2}+\epsilon}
]

继承 RadarPillars 的速度分解：

[
v_{x,i}=v^{comp}*{r,i}u*{x,i}
]

[
v_{y,i}=v^{comp}*{r,i}u*{y,i}
]

最终点特征：

```text
x, y, z
log_rcs
v_rel, v_comp
v_x, v_y
delta_t
range
sin_azimuth, cos_azimuth
local_density
local_rcs_mean
local_doppler_std
ego_comp_residual
```

RadarPillars 的消融表明，补偿径向速度及其 (x/y) 分量非常关键；只加入补偿速度分量即可带来显著提升，而简单 Pillar 内速度偏移没有稳定收益。

---

# 五、模块 B：自监督时序证据可靠性

建议模块名：

## **STER：Self-Supervised Temporal Evidence Reliability**

输出：

[
q_i\in[0,1]
]

它不是目标前景概率，而是：

> 历史点在补偿后是否能够作为当前帧的有效时序证据。

## 5.1 网络结构

```python
class TemporalReliabilityEstimator(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        return torch.sigmoid(self.net(features))
```

## 5.2 时序支持度

对历史点的预测位置 (\boldsymbol\mu_i)，在当前帧点云中寻找支持：

[
s_i=
\max_{j\in P_t}
\exp
\left[
-\frac12
(\mathbf p_j-\boldsymbol\mu_i)^\top
\bar{\mathbf\Sigma}_i^{-1}
(\mathbf p_j-\boldsymbol\mu_i)
\right]
]

这里的 (\bar{\mathbf\Sigma}_i) 使用：

* 固定协方差；或者
* 从学习协方差 `detach()` 得到的值。

这样可以避免 Reliability 网络和 Uncertainty 网络共同扩大协方差，使所有点都得到虚假的高支持。

## 5.3 伪标签

[
y_i=
\begin{cases}
1,&s_i>0.6\
0,&s_i<0.2\
\text{ignore},&\text{otherwise}
\end{cases}
]

可靠性损失：

[
\mathcal L_{\mathrm{rel}}
=========================

\mathcal L_{\mathrm{FocalBCE}}
+
0.2\mathcal L_{\mathrm{rank}}
]

排序损失：

[
\mathcal L_{\mathrm{rank}}
==========================

\max(0,m-q_i^++q_i^-)
]

推荐：

[
m=0.2
]

上传的 PRISM 指南已经提出基于当前帧支持度构造自监督可靠性标签。 最终实现需要补上停止梯度、忽略区间和防坍缩机制。

---

# 六、模块 C：Doppler 各向异性不确定性管

建议模块名：

## **DAUT：Doppler-Aware Uncertainty Tube**

历史点经过自车位姿变换后，其当前坐标系 BEV 位置为：

[
\mathbf p_i=[x_i,y_i]^\top
]

径向方向：

[
\mathbf u_i=[u_x,u_y]^\top
]

切向方向：

[
\mathbf n_i=[-u_y,u_x]^\top
]

确定性径向预测均值：

[
\boldsymbol\mu_i
================

\mathbf p_i+
\Delta t_i v^{comp}_{r,i}\mathbf u_i
]

实际代码必须通过数据集时间戳单元测试确认正负号。

## 6.1 时间相关协方差

建议不要直接预测最终位置标准差，而是预测速度不确定性：

[
s_{r,i}
=======

\sigma_{p,r}
+
|\Delta t_i|\sigma_{v,r,i}
]

[
s_{t,i}
=======

\sigma_{p,t}
+
|\Delta t_i|\sigma_{v,t,i}
]

协方差：

[
\mathbf\Sigma_i=
s_{r,i}^2\mathbf u_i\mathbf u_i^\top+
s_{t,i}^2\mathbf n_i\mathbf n_i^\top+
\sigma_0^2\mathbf I
]

物理约束：

[
s_{t,i}\geq s_{r,i}
]

因为 Doppler 直接约束径向运动，而切向运动不可充分观测。

## 6.2 有界参数化

[
\sigma_{v,r}
============

\sigma_{r,\min}
+
(\sigma_{r,\max}-\sigma_{r,\min})
\operatorname{sigmoid}(a_r)
]

[
\sigma_{v,t}
============

\sigma_{v,r}
+
(\sigma_{t,\max}-\sigma_{v,r})
\operatorname{sigmoid}(a_t)
]

其中：

[
[a_r,a_t]
=========

\operatorname{MLP}
[
r_i,
\log RCS_i,
|v^{comp}_{r,i}|,
|\Delta t_i|,
d_i
]
]

初始配置：

```yaml
DOPPLER_TUBE:
  SIGMA_POSITION_BASE: 0.03
  SIGMA_R_MIN: 0.03
  SIGMA_R_MAX: 0.60
  SIGMA_T_MAX: 2.00
  FIXED_SIGMA_R_POSITION: 0.10
  FIXED_SIGMA_T_POSITION: 0.50
```

开发时先使用固定 (0.10/0.50) m，再切换到可学习值。

---

# 七、模块 D：可靠性加权概率 Pillar 路由

建议模块名：

## **RAPR：Reliability-Aware Probabilistic Routing**

对历史点 (i)，在其均值附近搜索 (K_r\times K_r) 个 Pillar。

几何概率：

[
g_{ij}
======

\exp
\left[
-\frac12
(\mathbf c_j-\boldsymbol\mu_i)^\top
\mathbf\Sigma_i^{-1}
(\mathbf c_j-\boldsymbol\mu_i)
\right]
]

先只归一化几何概率：

[
\pi_{ij}
========

\frac{g_{ij}}
{\sum_{j'\in\mathcal N(i)}g_{ij'}+\epsilon}
]

再乘点可靠性：

[
w_{ij}=q_i\pi_{ij}
]

这个顺序很重要。如果在归一化之前把 (q_i) 同时放入分子和分母，它会被抵消。

历史 Pillar 特征：

[
\mathbf H_j
===========

\frac{
\sum_iw_{ij}\phi(\mathbf z_i)
}{
\sum_iw_{ij}+\epsilon
}
]

证据质量：

[
m_j=\sum_iw_{ij}
]

平均可靠性：

[
\bar q_j=
\frac{
\sum_iw_{ij}q_i
}{
m_j+\epsilon
}
]

证据门控：

[
\widetilde{\mathbf H}_j
=======================

(1-e^{-m_j})\mathbf H_j
]

最终输出：

```text
history_feature
evidence_mass
pillar_reliability
pillar_covariance
mean_delta_t
```

推荐参数：

```yaml
PROBABILISTIC_ROUTING:
  NEIGHBOR_SIZE: 5
  MIN_RELIABILITY: 0.05
  USE_EVIDENCE_MASS_GATE: true
  MAX_HISTORY_POINTS: 2048
```

原 PRISM 文档提出了将历史点软路由到邻近 Pillar 的总体思路。

---

# 八、模块 E：因果局部时序 Pillar 融合

建议模块名：

## **CRLF：Causal Reliability-Aware Local Fusion**

当前帧有效 Pillar：

[
\mathbf F_i^t
]

历史概率 Pillar：

[
\widetilde{\mathbf H}_j
]

当前 Pillar 作为 Query，只搜索半径 (R) 内的有效历史 Pillar，并保留 Top-(K_t)。

## 8.1 注意力打分

[
e_{ij}
======

\frac{
(\mathbf W_Q\mathbf F_i)
(\mathbf W_K\mathbf H_j)^\top
}{\sqrt d}
+
b^{geo}_{ij}
+
\alpha\log(\bar q_j+\epsilon)
+
\gamma\log(1+m_j)
-----------------

\beta|\Delta t_j|
]

几何偏置：

[
b^{geo}_{ij}
============

-\frac12
(\mathbf c_i-\mathbf c_j)^\top
(\mathbf\Sigma_j+\sigma_c^2\mathbf I)^{-1}
(\mathbf c_i-\mathbf c_j)
]

注意力：

[
a_{ij}=\operatorname{softmax}*j(e*{ij})
]

历史特征：

[
\widehat{\mathbf H}_i
=====================

\sum_ja_{ij}\mathbf W_V\mathbf H_j
]

## 8.2 门控融合

注意力熵：

[
E_i=-\sum_ja_{ij}\log(a_{ij}+\epsilon)
]

融合门：

[
g_i=
\operatorname{sigmoid}
\left[
\operatorname{MLP}
(
\mathbf F_i,
\widehat{\mathbf H}*i,
\max_ja*{ij},
E_i,
\bar q_i,
m_i
)
\right]
]

输出：

[
\mathbf F_i^{out}
=================

\mathbf F_i+
g_i\widehat{\mathbf H}_i
]

推荐配置：

```yaml
TEMPORAL_FUSION:
  HIDDEN_DIM: 64
  NUM_HEADS: 4
  LOCAL_RADIUS: 3
  TOPK: 16
  RELIABILITY_ALPHA: 1.0
  EVIDENCE_MASS_GAMMA: 0.5
  TIME_DECAY_BETA: 1.0
  USE_MAHALANOBIS_BIAS: true
  USE_GATE: true
```

该设计的复杂度接近：

[
O(pK_t)
]

而不是全局注意力的 (O(p^2))。

---

# 九、模块 F：RepDWC 可重参数化 Backbone

建议模块名：

## **RBB：Re-parameterizable BEV Backbone**

继续保留 RadarPillars 的三个阶段和均匀通道策略：

```text
Stage 1：3 blocks，stride 1
Stage 2：5 blocks，首层 stride 2
Stage 3：5 blocks，首层 stride 2
```

首选通道：

```text
[32, 32, 32]
```

精度候选：

```text
[48, 48, 48]
```

不应直接照搬 RadarNeXt 的 (C=64)，因为 RadarPillars 已经证明极稀疏雷达数据适合更小的均匀通道网络。

## 9.1 训练态 RepDWC

深度卷积阶段：

[
\mathbf y'=
\operatorname{Act}
\left[
BN(D_{3\times3}(\mathbf x))
+
BN(D_{1\times1}(\mathbf x))
\right]
]

点卷积阶段：

[
\mathbf y=
\operatorname{Act}
\left[
BN(P_{3\times3}(\mathbf y'))
+
BN(P_{1\times1}(\mathbf y'))
+
BN(\mathbf y')
\right]
]

RadarNeXt 正式架构采用这种深度卷积和点卷积的多分支训练结构，并在推理时进行结构重参数化。([Springer Nature][2])

## 9.2 部署态融合

将 Conv+BN 转换为：

[
\mathbf W'=
\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\mathbf W
]

[
\mathbf b'=
\beta-
\frac{\gamma\mu}{\sqrt{\sigma^2+\epsilon}}
+
\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}\mathbf b
]

然后：

* (1\times1) kernel 补零到 (3\times3)；
* identity BN 转为中心为 1 的卷积核；
* 所有对应分支 kernel 和 bias 相加；
* 删除训练分支。

## 9.3 等价性测试

部署转换前后必须满足：

[
\max
|f_{\mathrm{train}}(x)-f_{\mathrm{deploy}}(x)|
<10^{-4}
]

FP16 可放宽至 (10^{-3})。

需要覆盖：

* stride 1；
* stride 2；
* 有 identity；
* 无 identity；
* batch size 1 和 8。

---

# 十、模块 G：单 DCNv3 的 Lite-MDFEN

建议名称：

## **SR-MDFEN：Single-Deformable Raw-Bypass MDFEN**

这里不是完整复制 RadarNeXt 的所有 MDFEN 变体，而是保留两个经过其消融支持的原则：

1. 仅使用一个 DCNv3；
2. 将增强特征与未修改的原始特征融合。

RadarNeXt 检验了多种 DCNv3 数量和位置，结论是单个 DCNv3 在适当位置具有更好的精度—效率平衡；增加多个 DCNv3 不保证提升，一些配置还会因过度特征处理而降低准确率。([Springer Nature][2])

## 10.1 尺度定义

设 Backbone 输出：

[
F_1\in\mathbb R^{B\times C\times H\times W}
]

[
F_2\in\mathbb R^{B\times C\times H/2\times W/2}
]

[
F_3\in\mathbb R^{B\times C\times H/4\times W/4}
]

## 10.2 推荐融合拓扑

先进行高层语义上采样：

[
T_2=
\operatorname{RepDWC}
\left[
\operatorname{Concat}
(F_2,\operatorname{Up}(F_3))
\right]
]

只对高分辨率原始特征建立增强分支：

[
E_1=\operatorname{DCNv3}(F_1)
]

保留原始旁路并融合：

[
T_1=
\operatorname{RepDWC}
\left[
\operatorname{Concat}
(F_1,E_1,\operatorname{Up}(T_2))
\right]
]

再进行一次自底向上反馈：

[
B_2=
\operatorname{RepDWC}
\left[
\operatorname{Concat}
(T_2,\operatorname{Down}(T_1))
\right]
]

最终统一分辨率：

[
F_{neck}
========

\operatorname{Concat}
\left[
T_1,
\operatorname{Up}(B_2),
\operatorname{Up}(F_3)
\right]
]

再使用 (1\times1) Conv 压缩通道。

## 10.3 推荐配置

```yaml
LITE_MDFEN:
  ENABLED: true
  CHANNELS: 32
  USE_SINGLE_DCNV3: true
  DCN_KERNEL_SIZE: 3
  DCN_GROUPS: 4
  DCN_PATH: HIGH_RES_RAW_FEATURE
  PRESERVE_RAW_BYPASS: true
  FUSION_BLOCK: RepDWC
  OUTPUT_CHANNELS: 96
```

RadarNeXt 的正式 MDFEN 使用多路径层级融合，保留原始雷达特征，并通过 RepDWC 完成通道融合；官方消融中，单个 DCNv3 的选定位置取得了较好的准确率和效率平衡。([Springer Nature][2])

---

# 十一、检测头策略

不建议在第一轮同时更换检测头。

## 公平模型：PRISM-Pillars-RF-S

使用原始 `AnchorHeadSingle`。

用途：

* 与 RadarPillars 进行严格对比；
* 证明收益来自概率时序建模和高效空间增强；
* 避免检测头变化干扰消融。

## 精度模型：PRISM-Pillars-RF-C

使用 CenterHead：

* Center heatmap；
* (x,y,z) offset；
* (l,w,h)；
* yaw；
* 可选速度；
* IoU quality。

损失：

[
\mathcal L_{\mathrm{head}}
==========================

\mathcal L_{\mathrm{focal}}
+
\lambda_1\mathcal L_{\mathrm{L1}}
+
\lambda_i\mathcal L_{\mathrm{IoU}}
+
\lambda_d\mathcal L_{\mathrm{dIoU}}
]

可选加入 corner heatmap。

RadarNeXt 使用 Anchor-free CenterHead，并结合 Focal、L1、IoU 和 dIoU 等监督。([Springer Nature][2])

CenterHead 只有满足以下准入条件才进入最终主模型：

[
\Delta mAP\geq0.5
]

且：

[
\Delta Latency\leq10%
]

---

# 十二、最终损失函数

公平版总损失：

[
\mathcal L=
\mathcal L_{\mathrm{det}}
+
\lambda_{\mathrm{rel}}\mathcal L_{\mathrm{rel}}
+
\lambda_{\sigma}\mathcal L_{\sigma}
+
\lambda_{\mathrm{inv}}\mathcal L_{\mathrm{inv}}
]

推荐：

[
\lambda_{\mathrm{rel}}=0.20
]

[
\lambda_{\sigma}=0.01
]

[
\lambda_{\mathrm{inv}}=0.05
]

## 不确定性正则

[
\mathcal L_{\sigma}
===================

\operatorname{mean}
\left[
\max(0,s_r-s_{r,\max})
+
\max(0,s_t-s_{t,\max})
+
\max(0,s_r-s_t)
\right]
]

## 跨增强一致性

仅在 GT 或高置信前景区域 (\Omega)：

[
\mathcal L_{\mathrm{inv}}
=========================

\frac1{|\Omega|}
\sum_{j\in\Omega}
\left|
\operatorname{norm}(F_j^a)
--------------------------

\operatorname{stopgrad}
\left[
\operatorname{norm}(F_j^b)
\right]
\right|_2^2
]

第一版不加入单独的 temporal consistency loss，避免损失项过多。

---

# 十三、代码项目目录

```text
PRISM-Pillars-RF/
├── pcdet/
│   ├── datasets/
│   │   ├── vod/
│   │   │   ├── vod_dataset.py
│   │   │   └── sequence_loader.py
│   │   ├── tj4dradset/
│   │   │   ├── tj4dradset_dataset.py
│   │   │   └── sequence_loader.py
│   │   └── augmentor/
│   │       ├── data_augmentor.py
│   │       └── radar_process_augmentor.py
│   │
│   ├── models/
│   │   ├── radar_evidence/
│   │   │   ├── radar_point_embedding.py
│   │   │   ├── temporal_reliability.py
│   │   │   ├── doppler_uncertainty_tube.py
│   │   │   ├── probabilistic_pillar_router.py
│   │   │   └── temporal_support_builder.py
│   │   │
│   │   ├── backbones_3d/
│   │   │   └── vfe/
│   │   │       ├── pillar_vfe.py
│   │   │       └── prism_pillar_vfe.py
│   │   │
│   │   ├── map_to_bev/
│   │   │   ├── pointpillar_scatter.py
│   │   │   └── history_evidence_scatter.py
│   │   │
│   │   ├── temporal/
│   │   │   ├── local_candidate_retriever.py
│   │   │   ├── mahalanobis_bias.py
│   │   │   └── causal_local_pillar_fusion.py
│   │   │
│   │   ├── backbones_2d/
│   │   │   ├── base_bev_backbone.py
│   │   │   ├── rep_dwc.py
│   │   │   ├── rep_bev_backbone.py
│   │   │   ├── reparameterize.py
│   │   │   ├── dcnv3_wrapper.py
│   │   │   └── lite_mdfen.py
│   │   │
│   │   ├── dense_heads/
│   │   │   ├── anchor_head_single.py
│   │   │   └── prism_center_head.py
│   │   │
│   │   └── detectors/
│   │       ├── radar_pillars.py
│   │       └── prism_pillars_rf.py
│   │
│   └── utils/
│       ├── radar_geometry.py
│       ├── covariance_2d.py
│       ├── latency_profiler.py
│       └── deployment_validator.py
│
├── tools/
│   ├── cfgs/
│   │   ├── vod_models/
│   │   │   ├── radar_pillars.yaml
│   │   │   ├── prism_pillars.yaml
│   │   │   ├── prism_pillars_rf_s.yaml
│   │   │   └── prism_pillars_rf_c.yaml
│   │   └── tj4d_models/
│   │       └── prism_pillars_rf.yaml
│   ├── train.py
│   ├── test.py
│   ├── convert_to_deploy.py
│   └── benchmark_latency.py
│
└── tests/
    ├── test_time_sign.py
    ├── test_covariance.py
    ├── test_probability_conservation.py
    ├── test_reliability_weight.py
    ├── test_rep_parameterization.py
    ├── test_dcn_shape.py
    └── test_causal_sequence.py
```

---

# 十四、关键数据接口

每个 batch 返回：

```python
batch_dict = {
    "current_points": current_points,
    "history_points": history_points,

    "history_delta_t": delta_t,
    "history_sweep_idx": sweep_idx,
    "history_pose_to_current": poses,

    "current_voxel_coords": current_coords,
    "gt_boxes": gt_boxes,
    "sequence_id": sequence_id,
    "frame_id": frame_id
}
```

历史点格式：

```text
[
    batch_idx,
    x, y, z,
    rcs,
    v_rel,
    v_comp,
    delta_t,
    sweep_idx
]
```

必须保证：

* 训练和验证按 sequence 划分；
* 同一 sequence 不跨 split；
* 只读取过去帧；
* 保留真实时间差；
* 先做自车位姿转换，再做目标径向预测。

---

# 十五、Detector 前向过程

```python
def forward(self, batch_dict):
    current = batch_dict["current_points"]
    history = batch_dict["history_points"]

    # -------------------------------------------------
    # 1. Current-frame deterministic RadarPillars branch
    # -------------------------------------------------
    current_point_features = self.point_embedding(current)

    current_pillars, current_coords = self.current_vfe(
        current,
        current_point_features
    )

    current_pillars = self.pillar_attention(
        current_pillars,
        current_coords
    )

    # -------------------------------------------------
    # 2. Historical probabilistic evidence branch
    # -------------------------------------------------
    history_point_features = self.point_embedding(history)

    reliability = self.reliability_estimator(
        history,
        history_point_features
    )

    mu, covariance, sigma_r, sigma_t = self.doppler_tube(
        history,
        history_point_features
    )

    history_evidence = self.probabilistic_router(
        point_features=history_point_features,
        mean=mu,
        covariance=covariance,
        reliability=reliability,
        batch_idx=history[:, 0].long()
    )

    # -------------------------------------------------
    # 3. Causal local temporal fusion
    # -------------------------------------------------
    fused_pillars = self.temporal_fusion(
        current_features=current_pillars,
        current_coords=current_coords,
        history_features=history_evidence.features,
        history_coords=history_evidence.coords,
        evidence_mass=history_evidence.mass,
        reliability=history_evidence.reliability,
        covariance=history_evidence.covariance,
        delta_t=history_evidence.delta_t
    )

    # -------------------------------------------------
    # 4. Current BEV construction
    # -------------------------------------------------
    spatial_features = self.current_scatter(
        fused_pillars,
        current_coords
    )

    # -------------------------------------------------
    # 5. RepDWC multi-scale backbone
    # -------------------------------------------------
    multi_scale_features = self.rep_bev_backbone(
        spatial_features
    )

    # -------------------------------------------------
    # 6. Single-DCNv3 raw-bypass foreground refinement
    # -------------------------------------------------
    neck_features = self.lite_mdfen(
        multi_scale_features
    )

    # -------------------------------------------------
    # 7. Detection
    # -------------------------------------------------
    predictions = self.dense_head(
        neck_features
    )

    if self.training:
        return self.compute_losses(
            predictions,
            reliability,
            covariance,
            batch_dict
        )

    return self.post_processing(
        predictions,
        batch_dict
    )
```

---

# 十六、配置文件推荐

```yaml
MODEL:
  NAME: PRISMPillarsRF

  POINT_FEATURES:
    USE_RCS: true
    USE_RELATIVE_VELOCITY: true
    USE_COMPENSATED_VELOCITY: true
    USE_VELOCITY_COMPONENTS: true
    USE_DELTA_T: true
    USE_AZIMUTH_ENCODING: true
    OUTPUT_DIM: 32

  CURRENT_VFE:
    NAME: PillarVFE
    NUM_FILTERS: [32]

  PILLAR_ATTENTION:
    ENABLED: true
    HIDDEN_DIM: 32

  RELIABILITY:
    ENABLED: true
    HIDDEN_DIM: 32
    POS_THRESHOLD: 0.60
    NEG_THRESHOLD: 0.20
    RANK_MARGIN: 0.20
    MIN_ROUTING_Q: 0.05

  DOPPLER_TUBE:
    ENABLED: true
    LEARNABLE: true
    SIGMA_POSITION_BASE: 0.03
    SIGMA_R_MIN: 0.03
    SIGMA_R_MAX: 0.60
    SIGMA_T_MAX: 2.00

  PROBABILISTIC_ROUTING:
    NEIGHBOR_SIZE: 5
    USE_RELIABILITY: true
    USE_EVIDENCE_MASS_GATE: true
    MAX_HISTORY_POINTS: 2048

  TEMPORAL_FUSION:
    ENABLED: true
    HIDDEN_DIM: 64
    NUM_HEADS: 4
    LOCAL_RADIUS: 3
    TOPK: 16
    RELIABILITY_ALPHA: 1.0
    EVIDENCE_MASS_GAMMA: 0.5
    TIME_DECAY_BETA: 1.0
    USE_MAHALANOBIS_BIAS: true
    USE_GATE: true

  BACKBONE_2D:
    NAME: RepBEVBackbone
    NUM_FILTERS: [32, 32, 32]
    LAYER_NUMS: [3, 5, 5]
    LAYER_STRIDES: [1, 2, 2]
    DEPLOY_MODE: false

  LITE_MDFEN:
    ENABLED: true
    USE_SINGLE_DCNV3: true
    DCN_KERNEL_SIZE: 3
    DCN_GROUPS: 4
    PRESERVE_RAW_BYPASS: true
    OUTPUT_CHANNELS: 96

  DENSE_HEAD:
    NAME: AnchorHeadSingle
    CLASS_AGNOSTIC: false

DATA_CONFIG:
  NUM_SWEEPS: 3
  HISTORY_ONLY: true
  USE_TRUE_DELTA_T: true
  SEQUENCE_LEVEL_SPLIT: true

LOSS:
  LAMBDA_REL: 0.20
  LAMBDA_SIGMA: 0.01
  LAMBDA_INV: 0.05

OPTIMIZATION:
  BATCH_SIZE_PER_GPU: 8
  NUM_EPOCHS: 80
  OPTIMIZER: AdamW
  START_LR: 0.0003
  MAX_LR: 0.003
  WEIGHT_DECAY: 0.01
  GRAD_NORM_CLIP: 10
  USE_AMP: true
```

学习率和 OneCycle 设置可先沿用 RadarPillars，以保证基线比较公平。

---

# 十七、开发与训练顺序

不能从第一天就把全部模块联合训练。

## 阶段 P0：RadarPillars 严格复现

完成：

* 1-frame；
* 3-frame；
* 5-frame；
* 原速度分量；
* PillarAttention；
* uniform scaling；
* 参数量、GFLOPs、FPS。

成功标准：

[
|\Delta mAP|\leq0.5\sim1.0
]

---

## 阶段 P1：建立全部时序基线

依次实现：

1. naive accumulation；
2. ego-motion accumulation；
3. deterministic Doppler compensation；
4. isotropic Gaussian routing；
5. fixed anisotropic routing。

固定：

```text
q = 1
sigma_r = 0.10 m
sigma_t = 0.50 m
无 temporal attention
普通 RadarPillars backbone
```

只有满足：

[
\text{anisotropic}

>

\text{isotropic}

>

\text{deterministic}
]

才继续开发 Reliability。

---

## 阶段 P2：加入局部时序融合

此时：

* (q=1)；
* (\sigma) 固定；
* 训练 CRLF；
* Backbone 暂时保持原 RadarPillars。

目的：

> 单独验证局部时序检索是否有效。

---

## 阶段 P3：加入 Reliability

训练前 5 epoch：

```text
q = 1
lambda_rel = 0
```

第 6—15 epoch：

```text
lambda_rel 从 0 线性增加到 0.2
sigma 保持固定
```

随后启用：

* BCE；
* ranking；
* ghost augmentation；
* sweep dropout。

---

## 阶段 P4：启用可学习不确定性

先冻结 Reliability 5 个 epoch，只训练 sigma MLP。

随后联合解冻：

```text
Reliability LR = 0.5 × base LR
Sigma LR       = 1.0 × base LR
Temporal LR    = 0.5 × base LR
RadarPillars   = 0.2 × base LR
```

---

## 阶段 P5：替换 RepDWC Backbone

对比：

```text
PRISM + Dense Conv
PRISM + ordinary DWConv
PRISM + RepDWC train structure
PRISM + RepDWC deploy structure
```

初期不加入 MDFEN。

RepDWC 准入标准：

[
\Delta mAP\geq-0.3
]

同时满足至少一个：

[
Latency\下降\geq15%
]

或：

[
相同延迟下mAP提升\geq0.5
]

---

## 阶段 P6：加入 Lite-MDFEN

依次测试：

1. multi-path without DCN；
2. single DCN high-resolution；
3. single DCN middle-resolution；
4. single DCN low-resolution；
5. two DCNs；
6. single DCN without raw bypass；
7. final single DCN raw-bypass。

MDFEN 准入标准：

[
\Delta mAP\geq0.5
]

且：

[
Latency增长\leq10%
]

或弱小、远距离目标 AP 提升：

[
\geq1.0
]

---

## 阶段 P7：检测头实验

最后测试：

* AnchorHead；
* CenterHead；
* CenterHead + IoU/dIoU；
* CenterHead + corner auxiliary。

主消融始终保留 AnchorHead。

---

## 阶段 P8：整体微调

最终联合微调 10—20 epoch：

```text
base LR = 1e-4
Backbone LR multiplier = 0.5
PRISM modules = 1.0
CenterHead = 1.0
```

---

# 十八、必须完成的单元测试

## 1. 时间方向测试

人工构造：

* 点在雷达前方；
* 正径向速度；
* 已知 (\Delta t)。

确认均值移动方向正确。

## 2. 协方差测试

检查：

[
\lambda_{\min}(\Sigma)>0
]

以及：

[
\sigma_t\geq\sigma_r
]

## 3. 概率守恒

对每个历史点：

[
\sum_j\pi_{ij}\approx1
]

可靠性加权后：

[
\sum_jw_{ij}\approx q_i
]

## 4. Reliability 生效测试

同一几何概率下：

[
q_1>q_2
\Rightarrow
\sum_jw_{1j}>
\sum_jw_{2j}
]

## 5. 因果性测试

修改未来帧输入，当前预测不得变化。

## 6. RepDWC 等价性测试

部署前后输出误差满足规定阈值。

## 7. DCNv3 shape 测试

检查 NCHW/NHWC 转换，禁止在循环中重复 permute。

---

# 十九、主实验设计

## Table 1：VoD 主结果

列：

```text
Method
Frames
Entire-area mAP
Car AP
Pedestrian AP
Cyclist AP
Driving-corridor mAP
Params
GFLOPs
FPS
P95 latency
GPU memory
```

模型：

```text
PointPillars
RadarPillars 1-frame
RadarPillars 3-frame
RadarPillars 5-frame
RadarNeXt
PRISM-Pillars
PRISM-Pillars + RepDWC
PRISM-Pillars + Lite-MDFEN
PRISM-Pillars-RF-S
PRISM-Pillars-RF-C
```

RadarNeXt 的正式实验在 VoD 与 TJ4D 上同时报告准确率和边缘设备速度，可以作为精度—效率对比基准。([Springer Nature][1])

---

## Table 2：时序证据建模

| 模型                    | 目的       |
| --------------------- | -------- |
| 1-frame               | 单帧基线     |
| Naive accumulation    | 简单增加点数   |
| Ego-motion            | 仅补偿自车    |
| Deterministic Doppler | 确定性径向补偿  |
| Isotropic Gaussian    | 排除普通平滑   |
| Fixed anisotropic     | 物理各向异性   |
| Learned anisotropic   | 自适应不确定性  |
| + Reliability         | 抑制错误历史证据 |
| + Local fusion        | 完整 PRISM |

---

## Table 3：Backbone 与 Neck

| PRISM | Dense | DWConv | RepDWC | MDFEN | Head   |
| ----: | ----: | -----: | -----: | ----: | ------ |
|     ✓ |     ✓ |        |        |       | Anchor |
|     ✓ |       |      ✓ |        |       | Anchor |
|     ✓ |       |        |      ✓ |       | Anchor |
|     ✓ |     ✓ |        |        |     ✓ | Anchor |
|     ✓ |       |        |      ✓ |     ✓ | Anchor |
|     ✓ |       |        |      ✓ |     ✓ | Center |

---

## Table 4：MDFEN 消融

```text
FPN
PAN
Multi-path without DCN
Single DCN high-res
Single DCN mid-res
Single DCN low-res
Two DCNs
Single DCN without raw bypass
Final SR-MDFEN
```

RadarNeXt 的消融说明，无 DCNv3 的多路径网络虽然快但精度较低；两个或更多 DCNv3 的收益不稳定，部分配置反而损失精度，因此本项目必须保留“单 DCN + 原始旁路”的最小设计。([Springer Nature][2])

---

## Table 5：Reliability 消融

```text
q = 1
random q
learned q without L_rel
BCE only
BCE + ranking
BCE + ranking + ghost augmentation
foreground probability instead of reliability
```

报告：

* Spearman((q,s))；
* GT 内平均 (q)；
* GT 外平均 (q)；
* motion-tail 平均 (q)；
* ECE；
* false positives。

---

## Table 6：第二数据集

推荐使用 TJ4DRadSet：

```text
RadarPillars
RadarNeXt
PRISM
PRISM-RF-S
PRISM-RF-C
```

需要分别训练和测试，不能因类别协议不同直接声称严格跨域泛化。

---

# 二十、鲁棒性实验

至少加入：

| 扰动              | 强度              |
| --------------- | --------------- |
| 随机点丢失           | 10%、30%、50%     |
| Doppler 噪声      | 0.1、0.3、0.5 m/s |
| Ego-speed bias  | 0.2、0.5、1.0 m/s |
| 历史帧丢失           | 1、2、3 sweeps    |
| RCS scale       | 0.8、1.0、1.2     |
| Ghost injection | 5%、10%、20%      |

比较：

```text
RadarPillars 5-frame
RadarNeXt
PRISM
PRISM-RF
```

画出：

* 扰动强度—mAP；
* 扰动强度—Recall；
* 扰动强度—FP；
* 历史帧数—mAP；
* 历史帧数—Latency。

---

# 二十一、效率评估

需要拆分计时：

```text
Data loading
Ego alignment
Point embedding
Reliability
Uncertainty Tube
Probabilistic Routing
Temporal candidate retrieval
Temporal attention
RepDWC Backbone
MDFEN
Detection Head
Post-processing
Total
```

同时报告：

* 平均延迟；
* P95 延迟；
* batch size 1；
* warm-up 后 500—2000 次迭代；
* FP32；
* FP16；
* RepDWC 训练态；
* RepDWC 部署态。

目标值建议：

| 指标                   |           目标 |
| -------------------- | -----------: |
| 参数量                  |     低于 1.5 M |
| GFLOPs               |         低于 6 |
| 桌面 GPU               |    大于 40 FPS |
| AGX Orin             | 大于 18–20 FPS |
| PRISM-RF 相对 PRISM 延迟 |      不增加或略下降 |
| 相对 RadarPillars 总延迟  |     不超过约 2 倍 |

由于 DCNv3 可能需要自定义 ONNX/TensorRT 算子，论文中不能只报告 PyTorch FPS后直接宣称完成边缘部署。

---

# 二十二、论文创新点最终表述

论文只主张三个原创贡献。

## 创新一：各向异性概率历史证据

> We propose a Doppler-aware anisotropic evidence model that represents historical radar returns as spatial probability distributions rather than deterministic compensated points, explicitly distinguishing Doppler-observable radial motion from poorly observable tangential motion.

## 创新二：自监督时序可靠性

> We introduce a self-supervised temporal evidence reliability estimator that learns to suppress unsupported historical returns without additional point-level annotations.

## 创新三：因果局部概率融合

> We develop a causal reliability-aware local pillar fusion mechanism in which current pillars selectively retrieve historical evidence according to feature similarity, anisotropic motion uncertainty, evidence reliability, evidence mass, and temporal distance.

RepDWC 和 MDFEN 不应写入原创贡献列表，而应写成工程实现：

> To retain practical efficiency, the proposed temporal evidence model is implemented with a re-parameterizable BEV backbone and a single-deformable raw-bypass foreground refinement neck.

---

# 二十三、论文中心思想陈述

推荐全文反复围绕这句话：

> **Reliable radar detection requires correcting uncertain temporal evidence before enhancing spatial representations.**

中文：

> **可靠的多帧雷达检测，应先修正不确定的历史证据，再增强融合后的空间表征。**

更完整的版本：

> 多帧雷达检测的主要问题并不只是点数不足，而是历史点在不完整运动观测和噪声条件下被错误地确定性融合。PRISM-Pillars-RF 首先利用 Doppler 各向异性运动不确定性和自监督可靠性，将历史回波构造为概率证据；随后通过因果局部检索将其融合到当前 Pillar，并使用可重参数化主干和单可变形多尺度 Neck 对残余稀疏前景进行高效细化。

---

# 二十四、论文结构

## Abstract

逻辑顺序：

1. 4D 雷达全天候但稀疏、噪声大；
2. 多帧提高密度，但确定性补偿产生错位；
3. Doppler 只约束径向运动；
4. 提出 PRISM-Pillars-RF；
5. 各向异性概率历史证据；
6. 自监督可靠性；
7. 因果局部 Pillar 融合；
8. 高效重参数化空间细化；
9. 数据集结果和端侧速度。

## Introduction

### Paragraph 1

4D 雷达优势与稀疏问题。

### Paragraph 2

RadarPillars、RadarNeXt 等工作的进展。

### Paragraph 3

多帧累积的作用和确定性补偿缺陷。

### Paragraph 4

核心洞察：

> Historical returns are uncertain evidence, not deterministic geometry.

### Paragraph 5

Correct-then-Refine 方法论。

### Paragraph 6

三个贡献。

## Related Work

```text
2.1 Radar-only 4D object detection
2.2 Pillar-based efficient radar detection
2.3 Multi-frame radar aggregation
2.4 Motion uncertainty and reliability
2.5 Re-parameterizable and deformable feature extraction
```

## Method

```text
3.1 Overview and Correct-then-Refine principle
3.2 Radar feature encoding
3.3 Self-supervised temporal reliability
3.4 Doppler anisotropic uncertainty tube
3.5 Reliability-aware probabilistic routing
3.6 Causal local temporal pillar fusion
3.7 Re-parameterizable BEV backbone
3.8 Single-deformable raw-bypass MDFEN
3.9 Detection head and loss
```

## Experiments

```text
4.1 Datasets and protocols
4.2 Implementation details
4.3 Main comparison
4.4 Temporal evidence analysis
4.5 Reliability and uncertainty analysis
4.6 Backbone and MDFEN analysis
4.7 Robustness experiments
4.8 Efficiency and deployment
4.9 Qualitative results
4.10 Failure cases
```

---

# 二十五、论文图表

建议主文包含：

1. **总体架构图**：Correct → Fuse → Refine；
2. **Doppler 各向异性 Tube 图**；
3. **概率路由图**；
4. **局部时序检索图**；
5. **RepDWC 训练态—部署态图**；
6. **单 DCNv3 raw-bypass Neck 图**；
7. **历史帧数—精度/延迟曲线**；
8. **可靠性和不确定性可视化**；
9. **Naive、PRISM、PRISM-RF 定性对比**；
10. **扰动鲁棒性曲线**。

---

# 二十六、模块退出机制

不存在能够事先“确保”性能提升的组合，因此每个借鉴模块都应设置退出标准。

## RepDWC 不达标时

保留普通 RadarPillars Backbone，论文主线不受影响。

## MDFEN 不达标时

只保留 RepDWC，不把 MDFEN放入最终模型。

## CenterHead 不达标时

主模型继续使用 AnchorHead。

## 可学习 Sigma 不稳定时

使用固定或按时间解析增长的 Sigma：

[
s_r=0.10+0.15|\Delta t|
]

[
s_t=0.50+0.50|\Delta t|
]

## Reliability 不稳定时

先用解析可靠性：

[
q_i=
\exp(
-\eta_1 d_i^{support}
-\eta_2|e_i^{doppler}|
)
]

作为过渡基线。

---

# 二十七、最终投稿模型建议

最终保留两种配置。

## PRISM-Pillars-RF-S

目标：公平、轻量、易复现。

```text
C = 32
3 historical frames
fixed or learned uncertainty
Top-K = 16
RepDWC Backbone
single DCNv3 Lite-MDFEN
AnchorHead
```

## PRISM-Pillars-RF-C

目标：最高精度。

```text
C = 48
5 historical frames
learned uncertainty
Top-K = 16
RepDWC Backbone
single DCNv3 Lite-MDFEN
CenterHead + IoU/dIoU
```

论文主表同时报告两种：

* `S` 证明效率和公平性；
* `C` 展示性能上限。

---

# 最终定稿

整合后的项目不应被描述为“在 RadarPillars 上增加 PRISM、RepDWC、DCNv3 和 CenterHead”，而应描述为：

> **一个以 Correct-then-Refine 为指导原则的分层雷达检测框架。模型首先在历史回波层面解决不完整运动观测、错误补偿和证据可靠性问题；随后在当前 BEV 层面使用因果局部检索完成时序融合；最后通过可重参数化主干和保留原始特征的单可变形多尺度 Neck，对融合后的稀疏前景进行高效增强。**

最终方法链为：

[
\boxed{
\text{Doppler物理不确定性}
\rightarrow
\text{自监督可靠性}
\rightarrow
\text{概率证据路由}
\rightarrow
\text{因果局部融合}
\rightarrow
\text{RepDWC高效编码}
\rightarrow
\text{单DCN残余前景细化}
}
]

原创性集中在前三个时序概率模块；RadarNeXt 相关结构作为经过已有实验支持的效率与表征组件，用来增强工程完成

[1]: https://link.springer.com/article/10.1186/s13634-025-01271-2?utm_source=chatgpt.com "Radarnext: lightweight and real-time 3D object detector based on 4D mmWave imaging radar | Journal on Advances in Signal Processing | Springer Nature Link"
[2]: https://link.springer.com/article/10.1186/s13634-025-01271-2 "Radarnext: lightweight and real-time 3D object detector based on 4D mmWave imaging radar | Journal on Advances in Signal Processing | Springer Nature Link"
