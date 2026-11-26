# DPCAM：SolDef_AI 两任务语义分割实验

本目录实现 **ConvNeXt-Base + U-Net 解码器 + DPCAM**，并提供 SolDef_AI 两个任务的原图分组划分、匹配初始化消融、现代分割模型对比与独立测试流程。当前实验范围为 SolDef_AI；VisA 不在本轮计划内。

本目录是独立发布副本；本次更新整理代码与复现说明。原工作目录的训练已由 Trae 恢复，本次目录调整不影响运行中的实验，完整最终测试尚未完成。已有阶段性验证结果不足以支持 DPCAM 优于基线的结论，详见 [实验状态](docs/EXPERIMENT_STATUS.md)。本目录不包含原始数据、处理后的图像、预训练权重、实验检查点或本机环境。

旧版 VisA / SolDef 入口及原论文 README 保留在仓库根目录，见 [仓库导航](../REPOSITORY_GUIDE.md)。

## 文档与入口

- [实验协议](docs/PROTOCOL.md)：数据划分、模型、训练、指标及测试门控。
- [依赖与安装](docs/DEPENDENCIES.md)：现代模型所需 CUDA 扩展与环境约束。
- [迁移说明](docs/MIGRATION.md)：与原 GitHub 代码的对应关系及已有实验的保留方式。
- [发布检查](docs/VALIDATION.md)：CPU 回归、来源核验与尚未验证的范围。
- [实验状态](docs/EXPERIMENT_STATUS.md)：已完成与未完成的工作、已知限制。
- [第三方来源](docs/THIRD_PARTY.md)：作者代码、固定版本、权重与许可。

| 目录 / 文件 | 用途 |
|---|---|
| `prepare_data.py` | 从原始标注生成裁剪、掩码和原图分组清单 |
| `models/`、`paper_protocol/model.py` | ConvNeXt 解码网络和注意力算子 |
| `soldef_study/`、`innovation_study/` | 匹配初始化、稳定 Dice/Focal、消融与残差候选 |
| `two_task_study/` | 两任务 × 九变体 × 五种子的 90 项计划与调度 |
| `modern_study/` | 作者网络的适配层与预训练权重接口 |
| `comparison_study/` | 60 项新增对比、最终测试、效率测量与表格导出 |
| `configs/` | 可核对的协议配置；启动新实验前按本机数据重新准备 |
| `data/splits/` | 原图级划分参考；不包含图像与标注 |
| `scripts/` | 作者源码、权重和新实验配置的准备工具 |
| `tests/` | 不依赖完整实验训练的工作流检查 |

## 从干净目录准备

以下命令用于建立独立的新复现实验。所有命令在 `soldef_experiments/` 目录执行（从仓库根目录先执行 `cd soldef_experiments`）；本次发布不会自动执行这些训练命令。

### 1. 安装环境与作者依赖

基础环境使用 Python 3.11、PyTorch 2.5.1、torchvision 0.20.1。完整六模型对比还需要与 PyTorch/CUDA 匹配的 NATTEN、selective scan 等扩展；先按 [依赖说明](docs/DEPENDENCIES.md) 安装并检查它们，仅安装基础 requirements 不足以运行全部现代网络。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -r requirements-modern.txt
python scripts/fetch_sources.py
python scripts/fetch_sources.py --check
python scripts/fetch_weights.py --only pvt_v2_b2 mambavision_t mit_b2_source mit_b2 resnet101
```

作者源码按固定提交下载到 `modern_study/vendor/`；对比权重放到 `modern_study/assets/`，ResNet-101 权重放到 `weights/`。SegMAN 的分类预训练权重需要从作者的 [指定 Google Drive 文件](https://drive.google.com/file/d/152hZo70iJKO2mg9DWxBgG0na-wnqq4WF/view) 手动下载，保留文件名 `SegMAN_Encoder_s.pth.tar`，然后导入并检查：

```bash
python scripts/fetch_weights.py --only segman_s --cache-dir /path/to/downloads --offline
python scripts/fetch_weights.py --check
```

继续按 [依赖说明](docs/DEPENDENCIES.md) 安装与当前平台匹配的 NATTEN、mamba-ssm 及 SegMAN selective-scan 扩展。上述命令只获取源码和权重，不安装这些原生算子。ConvNeXt 的编码器权重由 timm 管理，还需要相应缓存或网络；当前代码记录初始化来源，但未对该下载文件作独立 SHA256 固定。下载内容均不提交到本仓库。第三方来源与许可见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。

### 2. 准备两个任务的数据

从 [SolDef_AI 数据来源论文](https://www.mdpi.com/2504-4494/8/3/117) 或 [作者发布的数据集](https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection) 获取原始数据。下列路径应指向直接包含配对 `.json` 标注和 `.jpg` 图像的目录，两个任务分开提供：

```bash
python prepare_data.py --dataset soldef_task1 --soldef-task1-source /path/to/dataset1
python prepare_data.py --dataset soldef --soldef-source /path/to/dataset2
python prepare_data.py --dataset soldef_task1 --audit-only
python prepare_data.py --dataset soldef --audit-only
```

预期 Task 1 有 228 张带标注俯视原图，Task 2 有 200 张带标注 45° 原图。脚本按原图决定划分，将同一原图的两个裁剪放入同一集合。`--dataset all` 仅包括这两个 SolDef 任务。脚本拒绝覆盖已存在的处理数据目录；保留已有实验时请在独立仓库副本准备新数据。

### 3. 固定本次复现的配置

仓库的配置用于呈现完整实验设计；本机生成的数据清单和权重会影响校验哈希。首次复现使用准备脚本建立本机配置，再验证模型设置：

```bash
python scripts/prepare_release.py
python -m comparison_study.verify_setup
```

准备脚本核对公开划分参考中的来源图像 ID、哈希、划分、类别及裁剪坐标，备份参考配置后重建本机的 90 + 60 项配置；不启动训练或模型推理。JPEG 编码工具版本可能改变处理图像字节，因此配置冻结本机实际生成的数据清单。脚本拒绝含运行、冻结或设置检查产物的目录，只能在新实验开始前执行。`verify_setup` 随后逐一检查 12 份对比配置并生成报告。训练开始后不要手改哈希来绕过冻结检查；CPU 设置检查不代表 CUDA 算子、完整训练或最终测试已经通过。

### 4. 启动队列与查看表格

```bash
python -m two_task_study.suite
python -m comparison_study.suite
```

先完成 90 项消融及其验证，再执行 60 项对比队列。后者在训练前执行完整输入尺寸的 GPU 检查，所有 150 项均验证完成后冻结所选检查点，再进入最终测试和效率测量。默认每项训练 150 轮、输入 512 × 512、batch size 2、五个训练种子。完整计划需要较长时间与充足磁盘空间。

只汇总已有指标、无需启动模型训练时：

```bash
python -m two_task_study.suite --report-only
python -m comparison_study.suite --report-only
```

检查点分别保存在 `two_task_runs/` 和 `comparison_runs/`；汇总表格在 `reports/two_task/` 与 `reports/comparison_study/`。验证集与测试集分别输出，未完成项目保留空值。最终论文表必须注明评估集合、完成种子数、指标单位和效率测量条件。

## 研究范围

DPCAM 的乘法注意力版本与两个残差候选在计划中独立命名。残差版本改变了原方法的前向运算，不能作为原 DPCAM 的同名替换。模型比较使用统一数据与训练预算，但保留各作者模型必要的损失和辅助监督，因此不是各模型原始论文基准结果的复刻。

数据来源论文使用的实例分割 AP 与本仓库的语义分割 mIoU / mPA / F1 不同，不能直接比较其数值。实验协议及裁剪局限见 [PROTOCOL.md](docs/PROTOCOL.md)。
