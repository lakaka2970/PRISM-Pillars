# 远程算力设备档案（PRISM-Pillars）

<div align="center">

**连接方式 · 目录结构 · 运行环境 · 部署记录**

创建：2026-08-06 · 最后更新：2026-08-09（新增 RTX 4090D 平台支持）

</div>

> ⚠️ **安全提示**：本文档包含算力租用实例的登录密码，且位于 git 跟踪目录。实例为 seetacloud 租用设备（重启后端口变化、可随时销毁），密码随实例生命周期失效。请勿将此文档推送到任何公开仓库。

---

## 0. 平台概览

项目可在两类 GPU 平台上训练与部署，**源码完全一致**（仓库中不存在平台相关算子）；差异仅在**运行环境**（torch 版本、CUDA 编译参数、spconv/cumm 版本）。

| 项 | 平台 A：RTX 5090 | 平台 B：RTX 4090D |
|----|------------------|-------------------|
| GPU 架构 | Blackwell **sm_120** | Ada **sm_89** |
| 显存 | 32 GB | 24 GB |
| 驱动 | 595.71.05（支持 CUDA 13.2 运行时） | 595.71.05 |
| CUDA Toolkit | 12.1（nvcc V12.1.105） | 12.1（cu121） |
| **CUDA 策略** | **PTX-JIT**（编译 compute_90 PTX → 驱动 JIT 到 sm_120） | 原生 sm_89 内核 |
| `TORCH_CUDA_ARCH_LIST` | `"9.0+PTX"` | `"8.9"`（或省略） |
| torch | **2.7.1+cu128** | **2.1.2+cu121** |
| spconv-cu121 | 2.3.8 | 2.3.6 |
| cumm-cu121 | 0.7.11 | 0.4.11 |
| Python | 3.10.20（conda env `prism5090`） | 3.10.8（base miniconda3） |
| numpy / numba | 1.26.4 / **0.65.1** | 1.26.4 / **0.65.1** |
| conda nvidia channel | ⛔ 被屏蔽 → PTX-JIT | ✅ 可用 |
| 环境构建脚本 | 位于服务器 `/root/autodl-tmp/*.sh` | 位于服务器 `/root/autodl-tmp/*.sh` |

> **如何判断当前所在平台**：
> ```bash
> nvidia-smi --query-gpu=name,memory.total --format=csv   # 型号
> python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
> # (9, 0) → 5090 / Blackwell；capability (8, 9) → 4090D / Ada
> ```

---

## A 部分：RTX 5090 平台（PRISM-5090）

### A.1 连接方式

| 项 | 值 |
|----|----|
| 服务商 | seetacloud（AutoDL 容器） |
| 主机名 | `connect.weste.seetacloud.com` |
| 端口 | **17355**（实例重启后会变化，见 §C.4） |
| 用户 | `root` |
| 密码 | `vL4WhPdDqt6o` |

```bash
ssh -p 17355 root@connect.weste.seetacloud.com
```

### A.2 硬件配置

| 项 | 值 |
|----|----|
| GPU | NVIDIA GeForce RTX 5090 ×1（**32 GB**，Blackwell sm_120） |
| 驱动 | 595.71.05（运行时支持 CUDA 13.2） |
| CUDA Toolkit | 12.1（`/usr/local/cuda-12.1`，nvcc V12.1.105） |
| CPU | 128 核（宿主机共享，容器配额以实测为准） |
| 内存 | 503 GB（宿主机共享） |
| 系统 | Ubuntu 22.04 容器（Linux 5.15.0-94-generic） |

#### A.2.1 CUDA 策略：PTX-JIT（关键）

由于 conda nvidia channel 被屏蔽（JSON decode error），无法安装 CUDA 12.8 toolkit。采用 **PTX 前向兼容 JIT** 策略：

| 环节 | 编译端 | 运行时 |
|------|--------|--------|
| 代码 | nvcc 12.1 → `compute_90` PTX | 驱动 595 → JIT 编译到 sm_120 |
| 设置 | `TORCH_CUDA_ARCH_LIST="9.0+PTX"` | 透明（驱动自动处理） |

**已验证**：iou3d_nms 及其他 4 个 CUDA 算子均在 RTX 5090 上正确执行。

#### A.2.2 磁盘布局

| 挂载点 | 容量 | 用途 | 当前占用 |
|--------|------|------|---------|
| `/root`（overlay） | **30 GB** | 仅系统与 conda 环境，**勿放大文件** | ~10 GB |
| `/root/autodl-tmp`（/dev/md0） | **80 GB** | 数据集、训练输出、checkpoint | ~42 GB |
| `/autodl-pub` | 20 TB（AutoFS） | AutoDL 公共数据集（无 VoD） | — |

### A.3 软件环境（已验证，2026-08-08）

Python：`/root/miniconda3/envs/prism5090/bin/python`（Python 3.10.20，conda env `prism5090`）

| 包 | 版本 | 说明 |
|----|------|------|
| torch | **2.7.1+cu128** | `cuda.is_available() = True`，sm_120 matmul 正常 |
| numpy | **1.26.4** | 必须锁定 <2（opencv/torch 兼容性） |
| numba / llvmlite | **0.65.1** / 0.47.0 | 必须锁定：0.66 破坏 rotate_iou 签名，≤0.60 段错误 |
| cumm-cu121 | **0.7.11** | spconv 2.3.8 的依赖（注意：cumm 版本线 ≠ spconv 版本线） |
| spconv-cu121 | **2.3.8** | PointToVoxel CPU 体素化（不执行 GPU conv kernel） |
| opencv-python-headless | 4.10.0.84 | 勿升级 5.x（要求 numpy≥2） |
| pcdet | 0.3.0+65202f3（develop 模式） | 5 个 CUDA 算子已按 compute_90+PTX 编译，JIT 到 sm_120 |

#### A.3.1 激活环境

```bash
export PATH=/root/miniconda3/envs/prism5090/bin:$PATH
# 或
/root/miniconda3/envs/prism5090/bin/python <script>
```

#### A.3.2 验证结果（2026-08-08）

- CUDA 算子导入：iou3d_nms / roiaware_pool3d / roipoint_pool3d / pointnet2_stack / pointnet2_batch ✅
- GPU 执行：iou3d_nms GPU 计算、torch matmul 2048×2048 均正确 ✅
- 单元测试：`python -m pytest tests/ -q` → **30 passed in 1.67s** ✅
- **1 epoch 冒烟训练（vod_radarpillar.yaml, bs=16）**：训练 → checkpoint → 评估 → AP 输出全链路通过 ✅（~65s）

---

## B 部分：RTX 4090D 平台（PRISM-4090D）

### B.1 连接方式

| 项 | 值 |
|----|----|
| 服务商 | seetacloud（AutoDL 容器） |
| 主机名 | `connect.westb.seetacloud.com` |
| 端口 | **20801**（实例重启后会变化，见 §C.4） |
| 用户 | `root` |
| 密码 | `tdARGYdowKIm` |

```bash
ssh -p 20801 root@connect.westb.seetacloud.com
```

### B.2 硬件配置

| 项 | 值 |
|----|----|
| GPU | NVIDIA GeForce RTX 4090 D ×1（**24564 MiB**，Ada sm_89） |
| 驱动 | 595.71.05（支持 CUDA 12.1 运行时） |
| CUDA Toolkit | 12.1（cu121） |
| 系统 | Ubuntu 22.04 容器 |

#### B.2.1 CUDA 策略：原生 sm_89

4090D 使用原生 Ada 架构内核，无需 PTX-JIT。torch 2.1.2+cu121 自带 sm_89 二进制内核；编译算子时用：

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
# 或省略该变量（nvcc 默认含 compute_89）
```

> 与 5090 相反：**不要在 4090D 上设置 `9.0+PTX`**（会生成不匹配的 PTX，虽可 JIT 但非最优）。

#### B.2.2 磁盘布局

| 挂载点 | 容量 | 用途 | 当前占用 |
|--------|------|------|---------|
| `/root`（overlay） | **30 GB** | 仅系统与 conda 环境，**勿放大文件** | ~10 GB |
| `/root/autodl-tmp`（共享文件系统） | 约数百 GB | 数据集、训练输出、checkpoint | — |
| `/autodl-pub/data` | 20 TB | AutoDL 公共数据集（无 VoD） | — |

### B.3 软件环境（已验证，2026-08-09）

Python：`/root/miniconda3/bin/python`（Python 3.10.8，**base** 环境，无独立 conda env）

| 包 | 版本 | 说明 |
|----|------|------|
| torch | **2.1.2+cu121** | `cuda.is_available() = True`，sm_89 matmul 正常 |
| numpy | **1.26.4** | 必须锁定 <2 |
| numba / llvmlite | **0.65.1** / 0.47.0 | 必须锁定（同 5090） |
| cumm-cu121 | **0.4.11** | spconv 2.3.6 的依赖 |
| spconv-cu121 | **2.3.6** | PointToVoxel CPU 体素化 |
| pcdet | 0.3.0+65202f3（develop 模式） | 5 个 CUDA 算子已按 sm_89 编译 |

#### B.3.1 激活环境

```bash
export PATH=/root/miniconda3/bin:$PATH
# 或直接用
/root/miniconda3/bin/python <script>
```

#### B.3.2 验证结果（2026-08-09）

- CUDA 算子导入：iou3d_nms / roiaware_pool3d / roipoint_pool3d / pointnet2_stack / pointnet2_batch ✅
- GPU 执行：iou3d_nms GPU 计算、torch matmul 2048×2048 均正确 ✅
- 单元测试：`python -m pytest tests/ -q` → **30 passed in 2.66s** ✅
- 数据集：8682 帧 + 5 个 pkl + ImageSets + gt_database 齐全，符号链接完好 ✅
- 缺失（不阻塞）：sklearn / shapely —— 项目源码未引用（grep 确认），无需安装

---

## C 部分：公共信息

### C.1 目录结构

```
/root/PRISM-Pillars/                    # 项目代码（与本机 e:\Work\FT\PRISM-Pillars 同步）
├── pcdet/                              # 已编译安装（setup.py develop）
├── tools/                              # train.py / test.py / cfgs/
├── tests/                              # 单元测试（30 项，全部通过）
├── docker/bootstrap_vod.sh             # VoD 部署脚本存档
├── data/VoD/view_of_delft_PUBLIC  ->   # 软链至数据盘
└── output                         ->   /root/autodl-tmp/outputs   # 训练输出（勿放根分区）

/root/autodl-tmp/
├── datasets/vod/
│   ├── view_of_delft_PUBLIC/           # 解压后数据（8682 帧，symlink 完好）
│   │   ├── radar_5frames/training/velodyne/   # 8682 个 .bin（5帧累积点云）
│   │   ├── radar/training/             # 单帧雷达 + calib
│   │   ├── lidar/training/             # label_2 / pose / image_2 / ImageSets
│   │   └── radar_5frames/ImageSets -> ../lidar/ImageSets/
│   └── label_2_with_track_ids/label_2/ # 6435 帧带 track ID 标签（仅时序实验用）
├── outputs/                            # 训练输出根目录（与 /root/PRISM-Pillars/output 软链）
├── *.whl                               # torch wheel 副本（可删除以释放空间）
├── *.sh                                # 环境构建脚本（install_torch, setup_env_fast, compile_pcdet）
└── *.log                               # 构建与训练日志
```

### C.2 数据集状态（VoD）

- 训练点云 **8682 帧**；官方 ImageSets：train **5139** / val **1296** / test 2247（test 帧在包内但**无标签**）
- infos 已生成：`vod_infos_train/val/test/trainval.pkl` + `vod_dbinfos_train.pkl`（GT 库：Car 15608 / Ped 16143 / Cyc 6685）
- track-id 标签独立存放，主实验禁用（详见 `paper_plans/assessment_and_experiment_plan.md` §20.2）
- CRC 校验通过；符号链接由 Linux unzip 自动还原
- 两平台数据均已就位（5090 与 4090D 各一份）

### C.3 代码同步流程

1. 本机为开发主体：改动在本机完成
2. `scp` 同步至远端（家庭上行上限 ≈5.4 MB/s；用原生 scp，勿用 paramiko SFTP，仅 1.1 MB/s）
3. 远端执行 `sed -i 's/\r$//' <files>` 去 Windows 行尾
4. git 仓库位于 `/root/PRISM-Pillars`，两平台与本地均基于 main 分支 `65202f3`

### C.4 注意事项

1. **实例重启后端口会变**：登录 seetacloud 控制台查看新端口 → 更新本文档对应平台 §1 的 Port → 首次连接确认 host key。
2. **根分区仅 30 GB**：一切大文件（数据、输出、缓存）必须落 `/root/autodl-tmp`；`output/` 已软链。
3. **计费**：实例运行即计费，实验间隙建议在控制台关机（关机保留磁盘，端口变化；释放则数据全丢）。
4. **PyPI 镜像**：优先使用 Tsinghua（`https://pypi.tuna.tsinghua.edu.cn/simple`，~2.9 MB/s），aliyun 在 5090 实例上行极慢（~70 KB/s）。
5. **CUDA 算子重编译（5090）**：`TORCH_CUDA_ARCH_LIST="9.0+PTX"`；**（4090D）**：`TORCH_CUDA_ARCH_LIST="8.9"` 或省略。其余步骤相同：`pip install --no-build-isolation -e .`、更新前 `find pcdet/ops -name "*.so" -delete`。
6. **torch 加载遗留 checkpoint**：torch ≥2.6 默认 `weights_only=True`，需要在 `torch.load` 处加 `weights_only=False`（见 §C.5 修复 #10）。5090（torch 2.7）**必需**；4090D（torch 2.1）无影响但代码已统一添加。

### C.5 环境搭建过程中的修复记录（复现必读）

| # | 问题 | 修复 | 位置 | 适用平台 |
|---|------|------|------|---------|
| 1 | pip 自动升级 numpy 至 2.x，与 torch 2.1 冲突 | 锁定 `numpy==1.26.4` | 环境 | 全部 |
| 2 | numba 0.66 报 `Signature mismatch`（rotate_iou 导入期） | 锁定 `numba==0.65.1`（0.61–0.65 均可，≤0.60 段错误） | 环境 | 全部 |
| 3 | PRISM 6 个模块相对导入越界（`....utils` 应为 `...utils`） | 已修复并同步 | `pcdet/models/radar_evidence/*.py`、`pcdet/models/temporal/{mahalanobis_bias,causal_local_pillar_fusion}.py` | 全部（源码） |
| 4 | py3.10 移除 `collections.Iterable` | 改为 `collections.abc.Iterable` | `tools/train_utils/optimization/fastai_optim.py` | 全部（源码） |
| 5 | VoD 评估函数缺失（`get_vod_eval_result` 在 eval1.py） | `vod_dataset.py` 改导入 `eval1` | `pcdet/datasets/vod/vod_dataset.py` | 全部（源码） |
| 6 | **RTX 5090 (sm_120) 上 torch 2.1.2 matmul 报 "no kernel image"** | 升级 torch 至 2.7.1+cu128（CUDA 12.8 运行时自带 sm_120 内核） | 环境 | **仅 5090** |
| 7 | **conda nvidia channel 被屏蔽**（JSON decode error），无法装 CUDA 12.8 toolkit | PTX-JIT：nvcc 12.1 编译 `compute_90` PTX，驱动 595 JIT 到 sm_120 | `compile_pcdet.sh`: `TORCH_CUDA_ARCH_LIST="9.0+PTX"` | **仅 5090** |
| 8 | **pypi aliyun 镜像极慢**（~70 KB/s） | 改用 Tsinghua pypi mirror（~2.9 MB/s） | pip `-i https://pypi.tuna.tsinghua.edu.cn/simple` | 全部 |
| 9 | **cumm-cu121 版本线混淆**：set `cumm-cu121==2.3.8` 但 cumm 最新只有 0.8.x | 正确 pin：`cumm-cu121==0.7.11`（spconv 2.3.8 要求 `cumm-cu121<0.8.0,>=0.7.11`） | `setup_env_fast.sh` | 5090（4090D 用 cumm 0.4.11） |
| 10 | **torch 2.7 `weights_only=True` 默认值**导致 checkpoint 加载失败（`numpy.core.multiarray.scalar` 被拒） | `torch.load(filename, map_location=loc_type, weights_only=False)` | `pcdet/models/detectors/detector3d_template.py`、`tools/convert_to_deploy.py`、`tools/benchmark_latency.py` | 全部（5090 必需，4090D 无害） |
| 11 | pcdet `pip install -e .` 构建隔离失败（iso env 无 torch） | `pip install --no-build-isolation -e .` | `compile_pcdet.sh` | 全部 |
| 12 | 缺少 tqdm 导致训练 dataset 加载失败 | `pip install tqdm` | 环境 | 全部 |

#### C.5.1 torch wheel 下载经验（5090 专属）

- **正确路径**：`https://download.pytorch.org/whl/cu128/<filename>`（flat path，非 `/whl/cu128/torch/`）
- download.pytorch.org 约 1 MB/s（单连接）；aria2 多连接初始 ~920 KB/s 但随着连接老化退化
- **推荐**：使用 `curl -r <range>` 8 段并行下载，合计 ~1 MB/s（单 IP 限速，多连接不叠加）
- 下载到 `/root/autodl-tmp/`（80 GB 数据盘），切勿放在 `/root/`（仅 30 GB overlay）
- sha256: `d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3`

---

## D 部分：常用命令速查

### D.1 RTX 5090

```bash
# 登录
ssh -p 17355 root@connect.weste.seetacloud.com

# 激活环境
export PATH=/root/miniconda3/envs/prism5090/bin:$PATH

# 训练（示例：P0 基线）
cd /root/PRISM-Pillars
/root/miniconda3/envs/prism5090/bin/python tools/train.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16 --epochs 60 --fix_random_seed --extra_tag p0_5f_seed666

# 评估
/root/miniconda3/envs/prism5090/bin/python tools/test.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml --batch_size 8 \
    --ckpt output/cfgs/vod_models/vod_radarpillar/<tag>/ckpt/checkpoint_epoch_60.pth

# 单元测试
/root/miniconda3/envs/prism5090/bin/python -m pytest tests/ -q

# CUDA 算子重编译（代码修改后）—— PTX-JIT
cd /root/PRISM-Pillars
export PATH=/root/miniconda3/envs/prism5090/bin:/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST="9.0+PTX"
find pcdet/ops -name "*.so" -delete
/root/miniconda3/envs/prism5090/bin/pip install --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

### D.2 RTX 4090D

```bash
# 登录
ssh -p 20801 root@connect.westb.seetacloud.com

# 激活环境
export PATH=/root/miniconda3/bin:$PATH

# 训练（示例：P0 基线）
cd /root/PRISM-Pillars
/root/miniconda3/bin/python tools/train.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16 --epochs 60 --fix_random_seed --extra_tag p0_5f_seed666

# 评估
/root/miniconda3/bin/python tools/test.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml --batch_size 8 \
    --ckpt output/cfgs/vod_models/vod_radarpillar/<tag>/ckpt/checkpoint_epoch_60.pth

# 单元测试
/root/miniconda3/bin/python -m pytest tests/ -q

# CUDA 算子重编译（代码修改后）—— 原生 sm_89
cd /root/PRISM-Pillars
export PATH=/root/miniconda3/bin:/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST="8.9"
find pcdet/ops -name "*.so" -delete
/root/miniconda3/bin/pip install --no-build-isolation \
    -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
```

### D.3 通用

```bash
# 监控
nvidia-smi; df -h /root/autodl-tmp; du -sh /root/autodl-tmp/outputs/*

# 判断平台
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
```

完整实验计划见 `docs/paper_plans/assessment_and_experiment_plan.md`。
