# 仓库目录导航

根目录的 [README.md](README.md) 已恢复为原仓库版本，原文保持不变。旧版 SolDef_AI / VisA 共用代码与示例保留在原路径；新增的 SolDef_AI 两任务实验保留在独立目录，避免同名模块互相覆盖。

| 内容 | 位置 | 使用方式 |
|---|---|---|
| 原论文说明 | [README.md](README.md) | 原版环境、算法、数据来源和文件说明 |
| 旧版 VisA / SolDef 训练与评估 | `train.py`、`test.py`、`test2.py`、`inference.py`、`config.py`、`models/` | 从仓库根目录使用，配套根目录依赖与配置 |
| VisA 预处理 | [process_data2.py](process_data2.py) | 保留 PCB1–3 的裁剪、正常样本抽样及划分逻辑 |
| 原有示例与历史产物 | `data-example/visa/`、`loss/visa_*.json`、`confusion_marix/visa/` | 原仓库已有内容，不代表新协议结果 |
| 最新 SolDef 两任务实验 | [soldef_experiments/README.md](soldef_experiments/README.md) | 在 `soldef_experiments/` 目录运行其中的准备和队列命令 |
| 新实验协议与状态 | [协议](soldef_experiments/docs/PROTOCOL.md)、[状态](soldef_experiments/docs/EXPERIMENT_STATUS.md) | 两任务划分、消融、六模型对比、指标和完成情况 |

## 新版 SolDef 的工作目录

```bash
cd soldef_experiments
# 在此目录按本目录 README 安装依赖、准备数据和配置。
# 软件检查（不启动真实数据训练）：
python -m pytest -q
```

两个版本各有自己的 `models/`、`dataset.py`、`metrics.py`、依赖和配置。分别从对应目录启动，使用各自环境；不要用 `PYTHONPATH` 将旧版目录插入新版运行环境。旧检查点和旧结果不能直接混入新版原图分组实验。

本次恢复基于原提交 `07db8f138a58d3ef519931237ea34a093550a429`，新版目录基于 `729ac48396d12fcde62b0518447d5631b938adfe`。保留历史文件的同时不恢复 `.pyc` 缓存。旧仓库遗漏的 PSPNet 源码从原本地 VisA 项目补回，模型工厂所需导出一并核对；这些兼容性修复不改变 DPCAM 算法或训练协议。

## 本次检查范围

根目录 README 已与原提交逐字节核对；旧版模型工厂导入与 `train.py --help` 检查通过。新版目录的 25 项 CPU 回归检查通过。此次没有重新执行旧版训练或验证旧检查点。

旧版 PSPNet 补回的是历史实现：其可选 `use_cbam=True` 分支存在通道不匹配（CBAM 接收类别输出，但按 512 通道构建）。模型工厂默认启用该选项，使用旧版 PSPNet 时需显式关闭 CBAM；该历史分支未在本次恢复中改写。

VisA 代码保留不表示重启其训练。原工作目录中的 SolDef 训练已由 Trae 恢复；本次只协调发布仓库目录与文档，不启动、停止或重启训练进程。
