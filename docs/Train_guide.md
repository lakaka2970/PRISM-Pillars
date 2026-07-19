# PRISM-Pillars-RF 训练指南

<div align="center">

**PRISM-Pillars-RF 完整训练与评估手册**

*Physics-Guided Reliable Temporal Evidence Fusion with Re-parameterized Foreground Refinement*

</div>

---

## 目录

- [1. 模型架构概览](#1-模型架构概览)
- [2. 环境搭建](#2-环境搭建)
- [3. 数据集准备](#3-数据集准备)
- [4. 配置系统说明](#4-配置系统说明)
- [5. 训练流程](#5-训练流程)
- [6. 训练命令](#6-训练命令)
- [7. 模型评估](#7-模型评估)
- [8. 部署转换](#8-部署转换)
- [9. 延迟基准测试](#9-延迟基准测试)
- [10. 分阶段开发协议](#10-分阶段开发协议)
- [11. 超参数参考](#11-超参数参考)
- [12. 损失函数详解](#12-损失函数详解)
- [13. 监控与可视化](#13-监控与可视化)
- [14. 常见问题排查](#14-常见问题排查)
- [15. 可复现性检查清单](#15-可复现性检查清单)

---

## 1. 模型架构概览

### 1.1 方法论：Correct-then-Refine（先纠正，后增强）

PRISM-Pillars-RF 遵循严格的三层 **Correct-then-Refine** 方法论：

```
┌──────────────────────────────────────────────────────────────────┐
│ 第一层：点级物理修正                                              │
│   多普勒各向异性不确定性 → 自监督可靠性 → 概率证据路由             │
├──────────────────────────────────────────────────────────────────┤
│ 第二层：概率时序融合                                              │
│   因果局部 Pillar 注意力 + 马氏距离偏置 + 可靠性先验              │
│   + 证据质量奖励 + 时间衰减惩罚                                   │
├──────────────────────────────────────────────────────────────────┤
│ 第三层：高效空间增强                                              │
│   RepDWC 可重参数化骨干网络 → 单 DCNv3 原始旁路多尺度颈部          │
└──────────────────────────────────────────────────────────────────┘
```

核心观点：**历史雷达回波是不确定的概率证据，而非确定的几何点。** 必须先利用 Doppler 可观性、时序支持度和运动不确定性对历史回波进行概率化修正，再使用可重参数化主干和轻量可变形 Neck 对融合后的残余稀疏前景进行高效增强。

### 1.2 数据流

```
当前帧 P_t                        历史帧 P_{t-k}
      │                                │
      ▼                                ▼
 PillarVFE +                    自车运动补偿对齐
 速度分量分解                          │
      │                        共享点特征编码
      ▼                                │
 PillarAttention           ┌──────────┼──────────┐
 (自注意力)                │          │          │
      │                    ▼          ▼          │
      │              STER (q_i)   DAUT (μ, Σ)    │
      │                    │          │          │
      │                    └────┬─────┘          │
      │                         ▼                │
      │              RAPR (概率路由)              │
      │                         │                │
      │                   历史 Pillar 特征        │
      │                         │                │
      ▼                         ▼                │
 ┌─────────────────────────────────────┐          │
 │   因果局部时序融合 (CRLF)            │          │
 │   • 局部候选检索                    │          │
 │   • 多头注意力 + 五类先验           │          │
 │     (马氏几何/可靠性/证据量/         │          │
 │      时间衰减/特征相似度)           │          │
 │   • 门控残差融合                    │          │
 └─────────────────────────────────────┘          │
                    │                             │
                    ▼                             │
           PointPillar Scatter → BEV               │
                    │                             │
                    ▼                             │
         RepDWC 骨干网络 (三阶段)                   │
         Blocks: [3, 5, 5], C: [32, 32, 32]       │
                    │                             │
                    ▼                             │
         Lite-MDFEN 颈部                           │
         • 单 DCNv3 作用于高分辨率原始特征           │
         • 保留原始特征旁路                         │
         • 自上而下 + 自下而上双路径                │
                    │                             │
                    ▼                             │
              检测头                               │
         AnchorHeadSingle / PRISMCenterHead        │
                    │                             │
                    ▼                             │
              3D 检测结果                           │
```

### 1.3 模块清单

| # | 模块 | 缩写 | 论文章节 | 输入 | 输出 | 参数量 |
|---|------|------|---------|------|------|--------|
| 1 | 雷达点特征编码 | — | §4 | 原始雷达点云 | 32维点特征 | ~1K |
| 2 | 时序可靠性估计器 | STER | §5 | 点特征 | q_i ∈ [0,1] | ~3K |
| 3 | 时序支持度构建器 | — | §5.2 | μ, Σ, 当前帧点 | 伪标签 s_i | 0 (固定) |
| 4 | 多普勒不确定性管 | DAUT | §6 | 点特征, Δt | μ, Σ (2×2协方差) | ~4K |
| 5 | 概率 Pillar 路由器 | RAPR | §7 | μ, Σ, q, 特征 | 历史 Pillar 特征 | 0 (无参数) |
| 6 | 因果局部 Pillar 融合 | CRLF | §8 | 当前+历史 Pillar | 融合后 Pillar 特征 | ~80K |
| 7 | RepDWC 骨干网络 | — | §9 | BEV (B,C,H,W) | 多尺度 BEV 特征 | ~30K (部署) |
| 8 | 轻量 MDFEN 颈部 | SR-MDFEN | §10 | 多尺度特征 | 增强 BEV 特征 | ~25K |
| 9 | AnchorHead 检测头 | — | §11 | BEV 特征 | 类别+框+方向 | ~30K |
| 10 | CenterHead 检测头 | — | §11 | BEV 特征 | 热力图+偏移+尺寸+偏航+速度+IoU | ~50K |

**总参数量 (PRISM-Pillars-RF-S, C=32)：** ~0.5M (训练态) / ~0.35M (部署态)
**总参数量 (RadarPillars 基线, C=32)：** ~0.27M

### 1.4 两条架构纪律

1. **当前帧不做概率扩散** —— 当前帧坐标是直接观测，保留 RadarPillars 原始确定性编码，防止目标轮廓被平滑。

2. **DCNv3 不进入物理证据建模之前** —— 可变形卷积会改变空间响应，必须放在物理引导的时序融合之后。正确顺序：点级物理修正 → 概率时序融合 → BEV 可变形增强。

---

## 2. 环境搭建

### 2.1 系统要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **操作系统** | Ubuntu 18.04 / Windows 10+ | Ubuntu 20.04+ |
| **Python** | 3.8+ | 3.10 |
| **PyTorch** | 2.0+ | 2.4+ |
| **CUDA** | 11.8+ | 12.1+ |
| **GPU 显存** | 8 GB (batch_size=4) | 24 GB (batch_size=16) |
| **内存** | 16 GB | 32 GB |
| **硬盘** | 50 GB | 100 GB (多数据集) |

### 2.2 安装步骤

#### 第一步：创建 Python 虚拟环境

```bash
# 使用 venv
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 升级 pip
python -m pip install -U pip setuptools wheel
```

#### 第二步：安装 PyTorch

```bash
# CUDA 12.1 版本
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 版本
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 第三步：安装 SparseConv (spconv)

spconv 是 PillarVFE 体素化的必需依赖：

```bash
# CUDA 12.x + PyTorch 2.4
pip install spconv-cu121

# CUDA 11.x + PyTorch 2.4
pip install spconv-cu118

# 如果没有预编译包，从源码编译：
# git clone https://github.com/traveller59/spconv.git
# cd spconv && python setup.py bdist_wheel && pip install dist/*.whl
```

#### 第四步：安装项目依赖

```bash
# 核心依赖
pip install -r requirements.txt

# 以开发模式安装 pcdet
python setup.py develop
```

#### 第五步：可选依赖

```bash
# WandB 实验跟踪
pip install wandb

# DCNv3 支持 (通过 MMCV，获取完整 DCNv3 精度)
# 注意：如果 MMCV 不可用，DCNv3Wrapper 会自动降级为标准 Conv2d (~1-2 mAP 损失)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# TensorBoard (通常随 PyTorch 一起安装)
pip install tensorboard
```

### 2.3 验证安装

```bash
# 验证 PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU数量: {torch.cuda.device_count()}')"

# 验证 spconv
python -c "import spconv; print(f'spconv {spconv.__version__}')"

# 验证 pcdet
python -c "from pcdet.config import cfg; print('pcdet OK')"

# 运行单元测试
python tests/test_time_sign.py
python tests/test_covariance.py
```

---

## 3. 数据集准备

### 3.1 View-of-Delft (VoD) —— 主要数据集

View-of-Delft 数据集提供带有 Doppler 速度的 4D 雷达点云，用于自动驾驶场景下的 3D 目标检测。

#### 3.1.1 数据下载与组织

从 [VoD 官方站点](https://github.com/tudelft-iv/view-of-delft-dataset) 下载数据集，按以下结构组织：

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
├── ImageSets/
│   ├── train.txt          # 训练序列 ID
│   ├── val.txt            # 验证序列 ID
│   └── test.txt           # 测试序列 ID
├── training/
│   ├── velodyne/          # 雷达点云 (.bin 文件)
│   ├── label_2/           # 3D 边界框标签
│   ├── calib/             # 标定文件
│   └── image_2/           # 相机图像 (可选，仅用于可视化)
└── testing/
    └── velodyne/          # 测试集雷达点云
```

#### 3.1.2 生成数据信息文件

```bash
# 生成训练信息文件和 GT 数据库
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

此命令会生成：
- `vod_infos_train.pkl` —— 所有训练帧的元数据
- `vod_infos_val.pkl` —— 所有验证帧的元数据
- `vod_dbinfos_train.pkl` —— GT 数据库（用于 GT 采样增强）

#### 3.1.3 多帧序列加载

对于 PRISM-Pillars-RF 时序训练，`sequence_loader.py` 模块负责：

- **序列级数据划分**：确保同一序列的帧不会同时出现在训练集和验证集中（防止时序泄露）
- **自车运动对齐**：补偿帧间的传感器运动
- **因果时间差**：计算历史帧与当前帧之间的 Δt
- **可配置历史帧数**：通过 `NUM_SWEEPS` 控制加载的历史帧数量

模型配置中的关键字段：
```yaml
DATA_CONFIG:
    NUM_SWEEPS: 3              # 历史帧数量
    HISTORY_ONLY: true         # 是否加载历史帧
    USE_TRUE_DELTA_T: true     # 使用真实时间戳计算 Δt
    SEQUENCE_LEVEL_SPLIT: true # 序列级划分防止数据泄露
```

#### 3.1.4 点云特征格式

每个雷达点包含 7 个特征：
```
[x, y, z, RCS, v_r, v_r_comp, timestamp]
```
其中：
- `x, y, z` —— 传感器坐标系下的 3D 坐标
- `RCS` —— 雷达散射截面（反射强度）
- `v_r` —— 相对径向速度（Doppler 测量值）
- `v_r_comp` —— 补偿径向速度（去除自车运动）
- `timestamp` —— 点的时间戳，用于计算 Δt

### 3.2 TJ4DRadSet（后续支持）

```
data/TJ4DRadSet/
├── ImageSets/
├── training/
│   ├── velodyne/
│   └── label_2/
└── testing/
    └── velodyne/
```

### 3.3 Astyx HiRes2019（仅支持单帧）

```bash
python -m pcdet.datasets.astyx.astyx_dataset create_astyx_infos \
    tools/cfgs/dataset_configs/astyx_dataset_radar.yaml
```

---

## 4. 配置系统说明

### 4.1 配置文件层级

PRISM-Pillars-RF 继承 OpenPCDet 的分层 YAML 配置系统：

```
tools/cfgs/
├── dataset_configs/
│   └── vod_dataset_radar.yaml       # 数据集定义 (基础配置)
└── vod_models/
    ├── vod_radarpillar.yaml         # RadarPillars 基线模型
    └── prism_pillars_rf_s.yaml      # PRISM-Pillars-RF-S 模型
```

模型配置通过 `_BASE_CONFIG_` 引用数据集配置：
```yaml
DATA_CONFIG:
    _BASE_CONFIG_: tools/cfgs/dataset_configs/vod_dataset_radar.yaml
    # ... 模型特定的覆写项
```

### 4.2 配置文件结构

一个完整的模型配置文件包含以下顶层段落：

| 段落 | 用途 |
|------|------|
| `CLASS_NAMES` | 检测类别列表 |
| `DATA_CONFIG` | 数据集路径、点云范围、数据处理、数据增强 |
| `MODEL` | 完整模型架构定义 |
| `OPTIMIZATION` | 训练超参数（学习率、批次大小、训练轮数、调度器） |

### 4.3 MODEL 段详解

```yaml
MODEL:
    NAME: PRISMPillarsRF              # 检测器类名

    # 当前帧编码 (继承 RadarPillars)
    POINT_FEATURES: {...}             # 点特征配置 (§4)
    VFE: {...}                        # PillarVFE + 速度分解
    BACKBONE_3D: {...}                # PillarAttention (自注意力)
    MAP_TO_BEV: {...}                 # PointPillarScatter

    # PRISM 时序证据模块
    RELIABILITY: {...}                # STER: 可靠性估计器 (§5)
    DOPPLER_TUBE: {...}               # DAUT: 不确定性管 (§6)
    PROBABILISTIC_ROUTING: {...}     # RAPR: 概率路由 (§7)
    TEMPORAL_FUSION: {...}           # CRLF: 因果局部融合 (§8)

    # 空间增强
    BACKBONE_2D: {...}               # RepBEVBackbone (§9)
    LITE_MDFEN: {...}                # Lite-MDFEN 颈部 (§10)

    # 检测头
    DENSE_HEAD: {...}                # AnchorHeadSingle 或 PRISMCenterHead (§11)

    # 损失权重
    LOSS:
        LAMBDA_REL: 0.20             # 可靠性损失权重
        LAMBDA_SIGMA: 0.01           # 不确定性正则化权重
        LAMBDA_INV: 0.05             # 跨增强一致性损失权重

    POST_PROCESSING: {...}           # NMS、分数阈值、评估指标
```

### 4.4 命令行覆写配置

使用 `--set` 参数可在运行时覆写任意配置项：

```bash
# 覆写批次大小和训练轮数
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set OPTIMIZATION.BATCH_SIZE_PER_GPU 4 OPTIMIZATION.NUM_EPOCHS 100

# 禁用某个模块
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.LITE_MDFEN.ENABLED False
```

---

## 5. 训练流程

### 5.1 训练循环架构

```
┌──────────────────────────────────────────────────────────────┐
│                        训练循环                               │
├──────────────────────────────────────────────────────────────┤
│  1. 数据加载                                                  │
│     - 序列加载器获取当前帧 + 历史帧                            │
│     - 自车运动对齐应用于历史点                                 │
│     - 数据增强 (翻转、旋转、缩放)                              │
│                                                               │
│  2. 前向传播 (见 §1.2 数据流)                                  │
│                                                               │
│  3. 损失计算                                                  │
│     L = L_det + λ_rel·L_rel + λ_sigma·L_sigma + λ_inv·L_inv  │
│                                                               │
│  4. 反向传播                                                  │
│     - 梯度计算 (PyTorch autograd 自动完成)                     │
│     - 梯度裁剪 (max_norm=10)                                  │
│                                                               │
│  5. 优化器更新                                                │
│     - Adam OneCycle 调度器                                    │
│     - 学习率 warmup → 峰值 → 余弦衰减                         │
│                                                               │
│  6. 日志记录与检查点保存                                       │
│     - TensorBoard 标量记录                                    │
│     - 可选 WandB 集成                                         │
│     - 每 N 轮保存检查点                                       │
│     - 早停 + 最佳模型保存                                     │
│                                                               │
│  7. 定期评估                                                  │
│     - 每隔 eval_interval 轮在验证集上评估                      │
│     - 多 IoU 阈值下计算各类别 3D AP                            │
│     - 按加权 mAP 跟踪最佳模型                                  │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 损失函数（论文 §12）

训练时的总损失为：

```
L = L_det + λ_rel · L_rel + λ_sigma · L_sigma + λ_inv · L_inv
```

| 损失项 | 权重 | 说明 | 何时生效 |
|--------|------|------|---------|
| **L_det** | 1.0 | 检测损失：Focal Loss + SmoothL1 + 方向分类 | 始终 |
| **L_rel** | 0.20 | 自监督可靠性：FocalBCE + 0.2·排序损失 | 存在历史帧且启用可靠性模块时 |
| **L_sigma** | 0.01 | 不确定性正则化：防止 σ 膨胀，强制 s_t ≥ s_r | 启用 DAUT 模块时 |
| **L_inv** | 0.05 | 跨增强特征一致性（仅前景区域） | 双重增强数据加载器可用时 |

### 5.3 优化器与学习率调度

默认优化器配置沿袭 RadarPillars 以保证公平比较：

```yaml
OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 8
    NUM_EPOCHS: 80
    OPTIMIZER: adam_onecycle      # Adam + OneCycle 调度
    LR: 0.003                     # 峰值学习率
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9
    MOMS: [0.95, 0.85]           # OneCycle 动量上下界
    PCT_START: 0.4                # 预热占整个周期的比例
    DIV_FACTOR: 10                # 起始 LR = 峰值 LR / 10
    DECAY_STEP_LIST: [35, 45]     # 阶梯衰减的轮次
    LR_DECAY: 0.1                 # 每次衰减因子
    LR_CLIP: 0.0000001            # 最低学习率
    GRAD_NORM_CLIP: 10            # 梯度裁剪最大范数
```

**OneCycle 学习率曲线示意：**

```
LR
│
│     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
│    ╱                              ╲
│   ╱                                ╲___________╲___________
│  ╱                                                   ╲
│ ╱                                                     ╲
└────────────────────────────────────────────────────────────→ Epoch
0     预热(40%)          峰值                   阶梯衰减
```

### 5.4 数据增强

```yaml
DATA_AUGMENTOR:
    DISABLE_AUG_LIST: ['gt_sampling']  # 公平对比中禁用 GT 采样
    AUG_CONFIG_LIST:
        - NAME: random_world_flip
          ALONG_AXIS_LIST: ['x']       # 随机 X 轴翻转 (50% 概率)

        - NAME: random_world_rotation
          WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]  # ±45° 旋转

        - NAME: random_world_scaling
          WORLD_SCALE_RANGE: [0.95, 1.05]  # ±5% 缩放扰动
```

**注意：** PRISM-Pillars-RF-S 配置中禁用了 GT 采样增强，以保证时序公平对比。原始 RadarPillars 基线配置中包含该项增强。

### 5.5 输出目录结构

训练完成后，输出文件组织如下：

```
output/vod_models/prism_pillars_rf_s/default/
├── ckpt/
│   ├── checkpoint_epoch_1.pth
│   ├── checkpoint_epoch_10.pth
│   ├── ...
│   └── checkpoint_epoch_80.pth
├── tensorboard/
│   └── events.out.tfevents.*
├── eval/
│   ├── eval_during_train/       # 训练过程中的定期评估
│   │   └── epoch_N/
│   └── eval_with_train/         # 最后 10 个检查点的最终评估
│       └── epoch_N/
├── log_train_YYYYMMDD-HHMMSS.txt
└── prism_pillars_rf_s.yaml     # 使用的配置副本
```

---

## 6. 训练命令

### 6.1 基线训练：RadarPillars

首先，训练基线模型以验证环境：

```bash
# 单 GPU 训练 (基线)
python tools/train.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16 \
    --epochs 60 \
    --extra_tag baseline

# 使用 WandB 跟踪
python tools/train.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16 \
    --use_wandb \
    --extra_tag baseline_wandb
```

**预期基线结果 (RadarPillars, C=32, VoD)：**
- Car 3D AP@0.50: ~36%
- Pedestrian 3D AP@0.25: ~41%
- Cyclist 3D AP@0.25: ~69%
- mAP: ~48.7%

### 6.2 PRISM-Pillars-RF-S 训练（公平对比版）

```bash
# 单 GPU 训练
python tools/train.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --epochs 80 \
    --extra_tag prism_s_v1

# 从检查点恢复训练
python tools/train.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --ckpt output/vod_models/prism_pillars_rf_s/default/ckpt/checkpoint_epoch_40.pth \
    --extra_tag resume

# 从预训练权重启动 (如 RadarPillars 骨干网络)
python tools/train.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --pretrained_model output/vod_models/vod_radarpillar/default/ckpt/checkpoint_epoch_60.pth \
    --extra_tag finetune
```

### 6.3 多 GPU 分布式训练

```bash
# 使用 torch.distributed (2 GPUs)
python tools/train.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --launcher pytorch \
    --tcp_port 18888

# 使用 torch.distributed (4 GPUs)
# 创建 scripts/dist_train.sh:
#   #!/bin/bash
#   python -m torch.distributed.launch --nproc_per_node=$1 tools/train.py \
#       --cfg_file $2 --launcher pytorch --batch_size $3

bash scripts/dist_train.sh 4 tools/cfgs/vod_models/prism_pillars_rf_s.yaml 8
```

### 6.4 训练参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cfg_file` | str | 必填 | 模型配置 YAML 文件路径 |
| `--batch_size` | int | 配置值 | 每 GPU 批次大小 |
| `--epochs` | int | 配置值 | 总训练轮数 |
| `--workers` | int | 8 | DataLoader 工作进程数 |
| `--extra_tag` | str | `default` | 实验标签（用于输出目录命名） |
| `--ckpt` | str | None | 恢复训练的检查点路径 |
| `--pretrained_model` | str | None | 预训练权重路径 |
| `--launcher` | str | `none` | `none` / `pytorch` / `slurm` |
| `--sync_bn` | flag | False | 分布式训练时启用 SyncBatchNorm |
| `--fix_random_seed` | flag | False | 固定所有随机种子为 666 |
| `--ckpt_save_interval` | int | 1 | 每 N 轮保存一次检查点 |
| `--max_ckpt_save_num` | int | 30 | 最多保留的检查点数量 |
| `--use_wandb` | flag | False | 启用 Weights & Biases 日志 |
| `--set` | list | None | 覆写配置项 (如 `--set OPTIMIZATION.LR 0.001`) |

### 6.5 训练配置汇总

| 场景 | 配置文件 | 每卡 Batch | GPU 数 | 总 Batch | 学习率 | 轮数 | 预计耗时 (A4000) |
|------|---------|-----------|--------|---------|--------|------|-----------------|
| RadarPillars 基线 | `vod_radarpillar.yaml` | 16 | 1 | 16 | 0.01 | 60 | ~3 小时 |
| PRISM-S (单 GPU) | `prism_pillars_rf_s.yaml` | 8 | 1 | 8 | 0.003 | 80 | ~8 小时 |
| PRISM-S (4 GPU) | `prism_pillars_rf_s.yaml` | 8 | 4 | 32 | 0.003 | 80 | ~3 小时 |

---

## 7. 模型评估

### 7.1 单检查点评估

```bash
# 评估指定检查点
python tools/test.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --ckpt output/vod_models/prism_pillars_rf_s/default/ckpt/checkpoint_epoch_80.pth

# 将详细结果保存到文件
python tools/test.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --ckpt output/vod_models/prism_pillars_rf_s/default/ckpt/checkpoint_epoch_80.pth \
    --save_to_file
```

### 7.2 批量评估所有检查点

```bash
# 评估目录下所有检查点（用于绘制学习曲线）
python tools/test.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --eval_all \
    --ckpt_dir output/vod_models/prism_pillars_rf_s/default/ckpt \
    --start_epoch 10
```

此命令会对每个保存的检查点运行评估，并将结果记录到 TensorBoard 中以绘制性能曲线。

### 7.3 多 GPU 评估

```bash
# 分布式评估
python tools/test.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --batch_size 8 \
    --ckpt output/vod_models/prism_pillars_rf_s/default/ckpt/checkpoint_epoch_80.pth \
    --launcher pytorch
```

### 7.4 评估指标说明

VoD 评估协议计算以下指标：

| 指标 | IoU 阈值 | 说明 |
|------|---------|------|
| **Car 3D AP** | 0.50 | 汽车类在 IoU=0.5 下的平均精度 |
| **Pedestrian 3D AP** | 0.25 | 行人类在 IoU=0.25 下的平均精度 |
| **Cyclist 3D AP** | 0.25 | 骑行者类在 IoU=0.25 下的平均精度 |
| **mAP** | — | 三类的平均 AP |

额外报告：
- **EAA** (Entire Annotated Area)：全传感器范围内的评估
- **R40**：40 个召回位置（KITTI 标准）
- 多个召回阈值下的各类 AP：R30、R50、R70

### 7.5 早停配置

基线配置启用了带自定义指标权重的早停：

```yaml
OPTIMIZATION:
    early_stop:
        enabled: True
        metrics: ['Car_3d/moderate_R40', 'Pedestrian_3d/moderate_R40', 'Cyclist_3d/moderate_R40']
        metric_reducer: weighted_mean
        metric_weights: [0.2, 0.3, 0.5]  # Cyclist > Pedestrian > Car
        mode: max
        patience: 30                      # 30 轮无提升则停止
        min_delta: 0.0
        start_epoch: 10                   # 第 10 轮之后才考虑早停
        save_best: True                   # 自动保存最佳模型
```

---

## 8. 部署转换

### 8.1 RepDWC 训练态 → 部署态转换

PRISM-Pillars-RF 使用 RepDWC 模块，具有**训练态多分支结构**（深度卷积 + 逐点卷积 + BatchNorm + 恒等分支）和**部署态融合单路径结构**。转换工具将所有分支折叠为等效的单一卷积：

```bash
# 转换并验证等价性 (推荐)
python tools/convert_to_deploy.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt output/vod_models/prism_pillars_rf_s/default/ckpt/checkpoint_epoch_80.pth \
    --output deploy_models/prism_s_deploy.pth \
    --validate

# 使用 FP16 精度转换
python tools/convert_to_deploy.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt output/vod_models/prism_pillars_rf_s/default/ckpt/checkpoint_epoch_80.pth \
    --output deploy_models/prism_s_deploy_fp16.pth \
    --fp16 --validate
```

### 8.2 等价性验证

转换工具通过对随机输入比较训练态和部署态的输出差异来验证等价性：

```
=== 等价性验证 ===
  B=1, stride=1, id=True: max_diff=0.000032 [通过]
  B=8, stride=1, id=True: max_diff=0.000045 [通过]
  B=1, stride=2, id=False: max_diff=0.000028 [通过]
  B=8, stride=2, id=False: max_diff=0.000051 [通过]

总结果: 全部通过
```

| 精度 | 容差 | 期望最大差异 |
|------|------|-------------|
| FP32 | 1e-4 | < 5e-5 |
| FP16 | 2e-3 | < 1e-3 |

### 8.3 部署转换收益

| 指标 | 训练态 | 部署态 | 提升 |
|------|--------|--------|------|
| 参数量 | ~0.5M | ~0.35M | 约 30% 减少 |
| 每块卷积分支数 | 4-5 | 1 | 计算图简化 |
| BN 融合 | 否 | 是 | 消除运行时 BN |
| 推理速度 | 基线 | +15-25% | A4000 实测 |

---

## 9. 延迟基准测试

### 9.1 分模块计时

```bash
# 标准基准测试 (1000 次迭代, 100 次预热)
python tools/benchmark_latency.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt deploy_models/prism_s_deploy.pth \
    --iterations 1000 --warmup 100

# FP16 基准测试
python tools/benchmark_latency.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt deploy_models/prism_s_deploy_fp16.pth \
    --iterations 1000 --warmup 100 --fp16

# 自定义批次大小
python tools/benchmark_latency.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt deploy_models/prism_s_deploy.pth \
    --batch_size 4
```

### 9.2 预期延迟分解 (RTX A4000, batch_size=1)

| 阶段 | 平均 (ms) | P95 (ms) | 占比 |
|------|----------|----------|------|
| 数据加载 + 自车对齐 | ~2.0 | ~3.5 | ~14% |
| 点特征编码 + VFE | ~1.5 | ~2.0 | ~10% |
| Pillar Attention (3D 骨干) | ~1.0 | ~1.5 | ~7% |
| 时序融合 (PRISM) | ~3.0 | ~4.5 | ~21% |
| RepDWC 骨干网络 | ~2.5 | ~3.0 | ~17% |
| MDFEN 颈部 | ~2.0 | ~3.0 | ~14% |
| 检测头 | ~2.5 | ~3.5 | ~17% |
| **总计** | **~14.5** | **~21.0** | **100%** |

目标 FPS：~69 FPS (RTX A4000) / ~30 FPS (Jetson AGX Orin)

### 9.3 延迟优化建议

1. **部署态转换**：先转换 RepDWC 再测试 —— 约 15-25% 加速
2. **FP16 推理**：使用 `--fp16` 参数 —— 额外约 30-40% 加速
3. **批量推理**：batch_size=4 相比 batch_size=1 吞吐量提升约 2.5 倍
4. **减少 TOPK**：将 `TEMPORAL_FUSION.TOPK` 设为 8（原 16）—— 时序融合加速约 15%，mAP 损失 < 0.5
5. **禁用 MDFEN**：如果延迟预算紧张，设置 `LITE_MDFEN.ENABLED: false` —— 节省约 2ms

---

## 10. 分阶段开发协议

论文规定了严格的分阶段开发协议（论文 §17）。每个阶段必须通过其成功标准后才能进入下一阶段。

### 10.1 阶段总览

```
P0 ──→ P1 ──→ P2 ──→ P3 ──→ P4 ──→ P5 ──→ P6 ──→ P7 ──→ P8
 │      │      │      │      │      │      │      │      │
基线   时序   局部   可靠性  可学习  RepDWC  MDFEN  检测头  联合
       基线   融合            Σ                           微调
```

### 10.2 P0：RadarPillars 严格复现

**目标**：复现 RadarPillars 基线，误差在 ±0.5-1.0 mAP 以内。

**训练命令**：使用 `tools/cfgs/vod_models/vod_radarpillar.yaml`。

```bash
# 1 帧基线
python tools/train.py --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml --batch_size 16 --epochs 60
```

**多帧配置覆写**：
```bash
# 3 帧 (朴素累积)
python tools/train.py --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --set DATA_CONFIG.NUM_SWEEPS 3

# 5 帧 (朴素累积)
python tools/train.py --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --set DATA_CONFIG.NUM_SWEEPS 5
```

**成功标准**：|ΔmAP| ≤ 0.5-1.0（与 RadarPillars 论文报告的 50.7 mAP 相比）。

### 10.3 P1：建立全部时序基线

**目标**：建立从朴素到各向异性的全频谱时序累积方法。

**所有测试使用**：q=1, 固定 σ_r=0.10, σ_t=0.50, 无时序注意力, 标准 RadarPillars 骨干。

| # | 方法 | 预期 mAP 趋势 |
|---|------|-------------|
| 1 | 朴素累积（无自车补偿） | 基线 |
| 2 | 自车运动对齐 | > #1 |
| 3 | 确定性 Doppler 补偿 | > #2 |
| 4 | 各向同性高斯路由 | > #3 |
| 5 | 固定各向异性路由 | > #4 |

**成功标准**：各向异性 > 各向同性 > 确定性（严格排序）。

### 10.4 P2：局部时序融合

**目标**：验证局部时序检索 (CRLF) 独立有效性。

**配置**：q=1, σ 固定, 训练 CRLF, 保持原始 RadarPillars 骨干。

```bash
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.RELIABILITY.ENABLED False \
    --set MODEL.DOPPLER_TUBE.LEARNABLE False \
    --set MODEL.BACKBONE_2D.NAME BaseBEVBackbone \
    --set MODEL.LITE_MDFEN.ENABLED False
```

### 10.5 P3：加入可靠性估计 (STER)

**目标**：训练并验证自监督可靠性估计器。

**训练计划**：

```
第 1-5 轮:   q = 1 (固定), λ_rel = 0 (预热阶段)
第 6-15 轮:  λ_rel 从 0 线性增加到 0.20, σ 保持固定
第 16+ 轮:   完整的可靠性训练 (BCE + ranking + ghost 增强)
```

**关键配置**：
```yaml
RELIABILITY:
    ENABLED: true
    HIDDEN_DIM: 32
    POS_THRESHOLD: 0.60      # s_i > 0.6 → 正样本伪标签
    NEG_THRESHOLD: 0.20      # s_i < 0.2 → 负样本伪标签
    RANK_MARGIN: 0.20        # 排序损失边距
```

### 10.6 P4：启用可学习不确定性

**目标**：通过有界参数化启用可学习的 Doppler 不确定性。

**训练计划**：

```
第 1-5 轮:   冻结可靠性模块，仅训练 σ MLP
第 6+ 轮:    联合解冻，使用分阶段学习率：
             - 可靠性 LR = 0.5 × 基础 LR
             - Sigma LR   = 1.0 × 基础 LR
             - 时序 LR    = 0.5 × 基础 LR
             - RadarPillars = 0.2 × 基础 LR
```

**关键配置**：
```yaml
DOPPLER_TUBE:
    ENABLED: true
    LEARNABLE: true
    SIGMA_POSITION_BASE: 0.03
    SIGMA_R_MIN: 0.03         # σ_r ∈ [0.03, 0.60]
    SIGMA_R_MAX: 0.60
    SIGMA_T_MAX: 2.00         # σ_t ∈ [σ_r, 2.00]
```

### 10.7 P5：替换 RepDWC 骨干网络

**目标**：用 RepDWC 替换 Conv2D 骨干，验证无性能退化。

**消融对比**：
```
PRISM + Dense Conv (基线)
PRISM + Depthwise Conv
PRISM + RepDWC (训练态)
PRISM + RepDWC (部署态)
```

```bash
# 使用 RepDWC 骨干训练
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.BACKBONE_2D.NAME RepBEVBackbone \
    --set MODEL.LITE_MDFEN.ENABLED False
```

**成功标准**：ΔmAP ≥ -0.3 且 (延迟降低 ≥ 15% 或 相同延迟下 mAP 提升 ≥ 0.5)。

### 10.8 P6：加入 Lite-MDFEN 颈部

**目标**：添加前景细化颈部，验证精度增益是否值得其延迟开销。

**7 种消融配置**（论文 §17, P6）：
```
1. 多路径无 DCNv3 (基线)
2. 单 DCNv3 作用于高分辨率特征
3. 单 DCNv3 作用于中分辨率特征
4. 单 DCNv3 作用于低分辨率特征
5. 两个 DCNv3 层
6. 单 DCNv3 无原始旁路
7. 最终版：单 DCNv3 + 原始旁路 ← 推荐
```

```bash
# 完整 PRISM-Pillars-RF-S (所有模块)
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml
```

**成功标准**：ΔmAP ≥ 0.5 且 ΔLatency ≤ 10%（或小/远距离目标 AP 提升 ≥ 1.0）。

### 10.9 P7：检测头实验

**目标**：对比 AnchorHead 与 CenterHead。

```bash
# CenterHead (精度版)
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.DENSE_HEAD.NAME PRISMCenterHead
```

**配置方案**：
- AnchorHead (公平基线)
- CenterHead
- CenterHead + IoU 质量分支
- CenterHead + dIoU 回归
- CenterHead + corner 辅助损失

**成功标准**：ΔmAP ≥ 0.5 且 ΔLatency ≤ 10%。

### 10.10 P8：联合微调

**目标**：所有模块端到端联合优化。

```bash
# 加载 P7 最佳检查点进行微调
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt output/vod_models/prism_pillars_rf_s/p7_centerhead/ckpt/checkpoint_epoch_80.pth \
    --epochs 20 \
    --extra_tag p8_finetune \
    --set OPTIMIZATION.LR 0.0001
```

**分阶段学习率乘数**（需手动实现）：
| 组件 | LR 乘数 |
|------|--------|
| 骨干网络 (RepDWC) | 0.5× |
| PRISM 模块 (STER, DAUT, RAPR, CRLF) | 1.0× |
| 检测头 | 1.0× |

基础 LR = 1e-4，最终联合微调 10-20 轮。

---

## 11. 超参数参考

### 11.1 关键超参数

| 参数 | 位置 | RadarPillars | PRISM-S | 增大影响 |
|------|------|-------------|---------|---------|
| **C (通道数)** | `BACKBONE_2D.NUM_FILTERS` | [32,32,32] | [32,32,32] | 更多容量，更多延迟 |
| **NUM_SWEEPS** | `DATA_CONFIG` | 5 (朴素) | 3 (概率) | 更多历史 → 更丰富时序信息，更多计算 |
| **TOPK** | `TEMPORAL_FUSION` | — | 16 | 更多候选 → 更好召回，O(p·K_t) 代价 |
| **LOCAL_RADIUS** | `TEMPORAL_FUSION` | — | 3 | 更大半径 → 更多上下文，更多计算 |
| **LR** | `OPTIMIZATION` | 0.01 | 0.003 | 更快收敛但有不稳定风险 |
| **BATCH_SIZE_PER_GPU** | `OPTIMIZATION` | 16 | 8 | 更大批次 → 更稳定梯度，更多显存 |
| **λ_rel** | `LOSS` | — | 0.20 | 更强的可靠性监督 |
| **λ_sigma** | `LOSS` | — | 0.01 | 更紧的不确定性约束 |
| **λ_inv** | `LOSS` | — | 0.05 | 更强的跨增强一致性 |

### 11.2 可靠性估计器超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HIDDEN_DIM` | 32 | MLP 隐藏层维度 |
| `POS_THRESHOLD` | 0.60 | 正样本伪标签的支持度阈值 |
| `NEG_THRESHOLD` | 0.20 | 负样本伪标签的支持度阈值 |
| `RANK_MARGIN` | 0.20 | 成对排序损失中的边距 |
| `MIN_ROUTING_Q` | 0.05 | 路由的最小可靠性（低于此值的点被丢弃） |

### 11.3 多普勒不确定性管超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SIGMA_POSITION_BASE` | 0.03 | 基础位置不确定性 σ_p,base (米) |
| `SIGMA_R_MIN` | 0.03 | 最小径向不确定性 (米) |
| `SIGMA_R_MAX` | 0.60 | 最大径向不确定性 (米) |
| `SIGMA_T_MAX` | 2.00 | 最大切向不确定性 (米) |
| `FIXED_SIGMA_R_POSITION` | 0.10 | 固定 σ_r (LEARNABLE=false 时使用) |
| `FIXED_SIGMA_T_POSITION` | 0.50 | 固定 σ_t (LEARNABLE=false 时使用) |

### 11.4 概率路由超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NEIGHBOR_SIZE` | 5 | K_r × K_r Pillar 搜索窗口 |
| `MAX_HISTORY_POINTS` | 2048 | 最大处理的历史点数量（内存限制） |
| `MIN_RELIABILITY` | 0.05 | 点过滤的可靠性阈值 |
| `USE_EVIDENCE_MASS_GATE` | true | 是否使用 (1 - e^{-m_j}) 证据质量门控 |

### 11.5 时序融合超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HIDDEN_DIM` | 64 | 注意力投影维度 |
| `NUM_HEADS` | 4 | 多头注意力头数 |
| `LOCAL_RADIUS` | 3 | 候选搜索的 Pillar 网格半径 |
| `TOPK` | 16 | 每个当前 Pillar 的最大历史候选数 |
| `RELIABILITY_ALPHA` | 1.0 | 注意力中可靠性对数先验的权重 |
| `EVIDENCE_MASS_GAMMA` | 0.5 | 注意力中证据量奖励的权重 |
| `TIME_DECAY_BETA` | 1.0 | 注意力中时间衰减惩罚的权重 |
| `USE_MAHALANOBIS_BIAS` | true | 是否包含马氏几何偏置 |
| `USE_GATE` | true | 是否使用可学习门控残差融合 |

---

## 12. 损失函数详解

### 12.1 检测损失 (L_det)

由检测头计算（AnchorHeadSingle 或 PRISMCenterHead）：

**AnchorHeadSingle**：
```
L_det = L_cls + λ_loc·L_loc + λ_dir·L_dir
```
- L_cls：Sigmoid Focal Loss (α=0.25, γ=2.0)，用于分类
- L_loc：加权 Smooth L1 Loss (β=1/9)，用于边界框回归
- L_dir：交叉熵损失，用于方向分类
- λ_loc=2.0, λ_dir=0.2

**PRISMCenterHead**：
```
L_det = L_heatmap + λ_off·L_offset + λ_size·L_size + λ_yaw·L_yaw + λ_vel·L_vel + λ_iou·L_iou + λ_diou·L_diou
```
- L_heatmap：Sigmoid Focal Loss
- L_offset, L_size：L1 Loss
- L_yaw：方向分箱交叉熵
- L_vel：L1 Loss (vx, vy)
- L_iou：L1 Loss (IoU 质量)
- L_diou：Distance-IoU 正则化

### 12.2 可靠性损失 (L_rel)

```
L_rel = FocalBCE(q, 伪标签) + 0.2 × RankingLoss(q, 伪标签)
```

**伪标签生成流程**：
- 通过马氏距离匹配计算当前帧最近点的支持度 s_i
- s_i > POS_THRESHOLD (0.60) → 标签 = 1 (可靠)
- s_i < NEG_THRESHOLD (0.20) → 标签 = 0 (不可靠)
- 其他情况 → 标签 = -1 (忽略，模糊区域)

**FocalBCE**：带 α=0.25, γ=2.0 的 Focal 二元交叉熵
**RankingLoss**：成对边距排序损失 max(0, margin - q⁺ + q⁻)

### 12.3 不确定性正则化 (L_sigma)

```
L_sigma = mean[max(0, s_r - s_r_max) + max(0, s_t - s_t_max) + max(0, s_r - s_t)]
```

三项约束：
1. s_r ≤ s_r_max (0.60)：径向不确定性不能爆炸
2. s_t ≤ s_t_max (2.00)：切向不确定性有上界
3. s_r ≤ s_t：物理约束 —— 切向不确定性始终大于径向

### 12.4 跨增强一致性损失 (L_inv)

```
L_inv = 1/|Ω| · Σ_j ||norm(F_j^a) - stopgrad(norm(F_j^b))||²
```

仅应用于前景区域。需要双重增强数据加载器（同一帧，不同增强种子）。如果数据加载器未提供 `spatial_features_2d_aug2`，此损失将被静默跳过。

---

## 13. 监控与可视化

### 13.1 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir output/vod_models/prism_pillars_rf_s/default/tensorboard --port 6006
```

**需要重点关注的关键指标**：

| 指标组 | 具体指标 | 健康范围 |
|--------|---------|---------|
| **检测** | `det_loss_rpn`, `det_loss_cls`, `det_loss_loc` | 持续下降 |
| **可靠性** | `rel_focal_loss`, `rel_ranking_loss` | 持续下降，ranking → 0 |
| **不确定性** | `loss_sigma` | 稳定，< 0.1 |
| **一致性** | `loss_inv` | 持续下降 |
| **总体** | `loss_total` | 平滑下降 |
| **学习率** | LR 曲线 | 遵循 OneCycle 调度 |

### 13.2 WandB 集成

```bash
# 安装 WandB
pip install wandb

# 登录 (仅首次)
wandb login

# 带 WandB 跟踪训练
python tools/train.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --use_wandb \
    --extra_tag experiment_1
```

WandB 自动记录：
- 所有 TensorBoard 标量
- 模型配置
- 系统指标（GPU 利用率、显存使用）
- 梯度直方图（如启用）

### 13.3 Demo 可视化

```bash
# 对单个点云进行推理可视化
python tools/demo.py \
    --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --ckpt output/vod_models/prism_pillars_rf_s/default/ckpt/checkpoint_epoch_80.pth \
    --data_path data/VoD/view_of_delft_PUBLIC/radar_5frames/training/velodyne/000000.bin
```

---

## 14. 常见问题排查

### 14.1 常见问题

#### CUDA 显存不足 (OOM)

**现象**：`RuntimeError: CUDA out of memory.`

**解决方案**（按顺序尝试）：
```bash
# 1. 减小批次大小
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --batch_size 4

# 2. 减少历史点数量上限
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.PROBABILISTIC_ROUTING.MAX_HISTORY_POINTS 1024

# 3. 减少最大体素数量
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set DATA_CONFIG.DATA_PROCESSOR.2.MAX_NUMBER_OF_VOXELS.train 8000

# 4. 使用梯度累积 (需手动实现) 或启用 FP16
```

#### spconv 导入错误

**现象**：`ImportError: cannot import name 'spconv'`

**解决方案**：
```bash
# 检查 CUDA 版本是否与 spconv 匹配
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配的 spconv
pip uninstall spconv
pip install spconv-cu121  # 替换为你的 CUDA 版本

# 或从源码编译
git clone https://github.com/traveller59/spconv.git
cd spconv
python setup.py bdist_wheel
pip install dist/*.whl
```

#### DCNv3 不可用

**现象**：日志中出现警告 "DCNv3 not available, falling back to standard Conv2d"

**影响**：约 1-2 mAP 下降。训练仍可正常进行。

**解决方案**：
```bash
# 安装带 DCNv3 的 MMCV
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# 或接受降级 —— Lite-MDFEN 将使用无 DCN 的多路径方案
```

#### 训练 Loss 为 NaN

**现象**：若干次迭代后 Loss 变为 NaN。

**原因与解决方案**：
1. **学习率过高**：降低 `OPTIMIZATION.LR`（如 0.003 → 0.001）
2. **梯度爆炸**：降低 `GRAD_NORM_CLIP`（如 10 → 5）
3. **协方差数值不稳定**：增大 `SIGMA_POSITION_BASE`（如 0.03 → 0.05）
4. **FP16 溢出**：禁用 AMP / 使用 FP32

#### 可靠性损失始终为零

**现象**：训练过程中 `rel_focal_loss = 0.0`。

**原因**：所有支持度分数落在忽略区间 [0.2, 0.6] 内，未生成任何伪标签。

**解决方案**：
```bash
# 放宽正/负样本阈值
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml \
    --set MODEL.RELIABILITY.POS_THRESHOLD 0.50 \
    --set MODEL.RELIABILITY.NEG_THRESHOLD 0.30
```

#### 数据加载速度慢

**现象**：GPU 利用率 < 80%，训练速度低于预期。

**解决方案**：
```bash
# 增加 DataLoader 工作进程数
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --workers 12

# 减少 MAX_HISTORY_POINTS 以加速序列加载
--set MODEL.PROBABILISTIC_ROUTING.MAX_HISTORY_POINTS 1024

# 固定内存 (OpenPCDet dataloader 已默认启用)
```

### 14.2 模块退出机制

如果某个模块导致性能下降，按退出标准禁用：

| 问题 | 操作 |
|------|------|
| RepDWC 导致 mAP 下降 > 0.3 | 设置 `BACKBONE_2D.NAME: BaseBEVBackbone` (退回 Conv2D) |
| MDFEN 增加延迟 > 10% 或 mAP 提升 < 0.5 | 设置 `LITE_MDFEN.ENABLED: false` |
| CenterHead 不稳定或过慢 | 使用 `DENSE_HEAD.NAME: AnchorHeadSingle` |
| 可学习 σ 不稳定 | 设置 `DOPPLER_TUBE.LEARNABLE: false` (使用固定 σ) |
| 可学习 q 不稳定 | 设置 `RELIABILITY.ENABLED: false` (使用解析 q 公式) |

---

## 15. 可复现性检查清单

论文级可复现结果需逐项验证：

### 训练前

- [ ] 随机种子已固定：`--fix_random_seed` 参数或 `OPTIMIZATION.FIX_RANDOM_SEED: True`
- [ ] 配置文件已存档（自动复制到输出目录）
- [ ] 数据集信息文件已从同一数据版本重新生成
- [ ] 依赖版本已锁定（执行 `pip freeze > requirements_frozen.txt` 记录）
- [ ] GPU 型号和 CUDA 版本已记录

### 训练中

- [ ] 批次大小已记录（所有 GPU 的有效总 batch）
- [ ] 所有超参数已记录（配置 YAML 为唯一真相来源）
- [ ] 数据增强设置已验证（公平对比中禁用 GT 采样）
- [ ] FP32 训练（除非显式声明，否则不使用 AMP）
- [ ] 序列级划分已启用（`SEQUENCE_LEVEL_SPLIT: true`）
- [ ] 无测试集信息泄露

### 训练后

- [ ] 最佳检查点按验证集指标选择（非测试集）
- [ ] 所有评估指标按标准 IoU 阈值报告
- [ ] 各类别 AP 逐一报告（而非仅 mAP）
- [ ] 延迟在相同硬件上使用部署态模型测量
- [ ] 参数量和 GFLOPs 已计算
- [ ] RepDWC 部署转换等价性已验证（max_diff < 1e-4）

### 论文报告

- [ ] 主结果：PRISM-Pillars-RF-S vs RadarPillars 基线
- [ ] 消融实验：每个模块的贡献（ΔmAP, ΔLatency）
- [ ] 各类别详细结果（Car / Pedestrian / Cyclist）
- [ ] 参考硬件上的推理速度 (FPS)
- [ ] 模型规模（参数量、FLOPs）

---

## 附录 A：命令速查卡

### 训练一行命令

```bash
# 基线
python tools/train.py --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml --batch_size 16 --epochs 60

# PRISM-Pillars-RF-S
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --batch_size 8 --epochs 80

# PRISM-Pillars-RF-S + WandB
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --batch_size 8 --use_wandb

# 恢复训练
python tools/train.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --ckpt <path> --extra_tag resume
```

### 评估一行命令

```bash
# 单检查点
python tools/test.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --batch_size 8 --ckpt <path>

# 全量检查点 (学习曲线)
python tools/test.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --eval_all --start_epoch 10
```

### 部署一行命令

```bash
# 转换为部署态
python tools/convert_to_deploy.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --ckpt <path> --output deploy.pth --validate

# 基准测试
python tools/benchmark_latency.py --cfg_file tools/cfgs/vod_models/prism_pillars_rf_s.yaml --ckpt deploy.pth --iterations 1000
```

### 测试一行命令

```bash
# 全部测试
python -m pytest tests/ -v

# 单独测试
python tests/test_time_sign.py
python tests/test_covariance.py
python tests/test_rep_parameterization.py
```

---

## 附录 B：配置对比表

| 配置项 | RadarPillars 基线 | PRISM-Pillars-RF-S |
|--------|------------------|-------------------|
| **模型类** | PointPillar | PRISMPillarsRF |
| **点特征** | x, y, z, RCS, v_r, v_r_comp, time | + v_x, v_y, Δt, range, sin/cos 方位角, 局部统计 |
| **VFE** | PillarVFE (C→32) | PillarVFE (C→32) |
| **3D 骨干** | PillarAttention (1 head) | PillarAttention (1 head) |
| **可靠性** | — | STER: 3层 MLP → q_i |
| **不确定性** | — | DAUT: 有界可学习 σ_r, σ_t |
| **路由方式** | 确定性 | RAPR: 概率加权 |
| **时序融合** | 朴素拼接 | CRLF: 五先验局部注意力 |
| **2D 骨干** | BaseBEVBackbone (Conv2D, 3阶段) | RepBEVBackbone (RepDWC, 3阶段) |
| **颈部** | — | Lite-MDFEN (单 DCNv3 + 原始旁路) |
| **检测头** | AnchorHeadSingle | AnchorHeadSingle |
| **损失函数** | 仅 L_det | L_det + L_rel + L_sigma + L_inv |
| **参数量 (~)** | 0.27M | 0.5M (训练) / 0.35M (部署) |
| **历史帧数** | 5 (朴素) | 3 (概率) |
| **学习率** | 0.01 | 0.003 |
| **训练轮数** | 60 | 80 |

---

## 附录 C：关键文件索引

| 文件 | 用途 |
|------|------|
| [tools/train.py](../tools/train.py) | 训练入口 |
| [tools/test.py](../tools/test.py) | 评估入口 |
| [tools/convert_to_deploy.py](../tools/convert_to_deploy.py) | RepDWC 部署转换 |
| [tools/benchmark_latency.py](../tools/benchmark_latency.py) | 分模块延迟测试 |
| [tools/demo.py](../tools/demo.py) | 推理可视化 |
| [tools/cfgs/vod_models/vod_radarpillar.yaml](../tools/cfgs/vod_models/vod_radarpillar.yaml) | 基线配置 |
| [tools/cfgs/vod_models/prism_pillars_rf_s.yaml](../tools/cfgs/vod_models/prism_pillars_rf_s.yaml) | PRISM-S 配置 |
| [tools/cfgs/dataset_configs/vod_dataset_radar.yaml](../tools/cfgs/dataset_configs/vod_dataset_radar.yaml) | VoD 数据集配置 |
| [pcdet/models/detectors/prism_pillars_rf.py](../pcdet/models/detectors/prism_pillars_rf.py) | 主检测器实现 |
| [pcdet/utils/loss_utils.py](../pcdet/utils/loss_utils.py) | 损失函数实现 |
| [docs/paper_plans/great_upgrade_3.md](paper_plans/great_upgrade_3.md) | 完整论文计划 |

---

> **文档版本**: 1.1
> **最后更新**: 2026-07-19
> **维护者**: PRISM-Pillars-RF 团队
