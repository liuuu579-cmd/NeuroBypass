# NeuroBypass

根据输入动态选择 Transformer 跳过层的研究项目，当前主干为 OPT-13B（40 层）。研究动机是：静态减层已有一定效果，能否由输入预测更合适的跳层组合，并在质量与计算量之间调整预算？

本仓库于 **2026-11-03** 整理现有实验，包含原程序的逐字节快照、三份周报、历史结果图，以及独立新增的数据审计和修正评测程序。原文件保持原样。

## 先读结论

- 最新路线已从固定剪 6 层，发展为 **输入特征 + 目标剪枝层数 K → 40 层保留分数 → 选 K 层跳过**；最后一版是读取原始 token + position embedding 的 Transformer Router。
- 最新评测把剪枝方向写反了。标签 `1=保留，0=跳过`，BCE 学的是保留分数；应选择最低分的 K 层。最新脚本却用最高分，并且训练命中率的统计也反向。
- 最新数据的 2,000 条记录仅来自 **20 个不同的 128-token 起始前缀组**。复现训练脚本的 seed=42 随机切分后，200 条验证数据全部与训练数据共享前缀组，不能视为独立文本上的泛化评估。
- 现有结果尚不足以证明输入动态剪枝优于同预算的强静态方法，也没有证明实际推理加速。完整证据、历史进展及下一步建议见 [研究进度与问题分析](analysis/PROJECT_REVIEW.zh-CN.md)。

## 文件结构

| 路径 | 内容 |
| --- | --- |
| `legacy/` | 工作目录顶层与 `skip/` 的历史自有 Python、Shell 和说明文件，逐字节复制 |
| `evidence/reports/` | 三份原始 DOCX、提取文字和内嵌图，作为历史材料 |
| `evidence/figures/` | 现有结果图，保留原图内容 |
| `analysis/source_manifest.json` | 原路径、原修改时间、字节数、SHA-256 |
| `analysis/source_timeline.csv` | 源程序按 Asia/Shanghai 修改时间排序 |
| `analysis/data_audit.json` | 最新四个 Zarr 分片的只读统计 |
| `analysis/checkpoint_direction.json` | 现有检查点的 CPU 方向诊断 |
| `tools/` | 本次新增程序，与历史脚本独立 |

仓库不包含 OPT 权重、Router 检查点、C4 原始语料、Zarr 大数据与第三方 `lm-evaluation-harness` 副本。它们保留在原工作目录。历史脚本中的绝对路径、依赖与旧行为也保持原样，不建议未经审查直接运行。

## 复现本次只读检查

本次使用现有环境 `/home/liushenyu/anaconda3/envs/traindata/bin/python`，依赖版本记录在 `requirements-analysis.txt`。以下命令从本仓库目录运行；所有结果输出到新文件，默认拒绝覆盖已有文件。

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/audit_dynamic_data.py \
  --source /home/liushenyu/test --output analysis/data_audit_rerun.json

PYTHONDONTWRITEBYTECODE=1 python tools/check_checkpoint_direction.py \
  --source /home/liushenyu/test --snapshot . \
  --base-model /home/liushenyu/opt-13b \
  --output analysis/checkpoint_direction_rerun.json

PYTHONDONTWRITEBYTECODE=1 python tools/evaluate_router_corrected.py --self-test
```

CPU 检查只读取 embedding 表与 Router，不执行 OPT decoder，也不将训练来源诊断包装成独立测试集结果。

## 使用新增的修正评测程序

准备每行包含 `prompt` 字符串的本地 JSONL。程序会对相同输入、相同 K 比较原方向和修正方向，记录逐条 KL 与剪枝组合；检查点不存在会立即报错。它使用 float32 KL、`use_cache=False`，并临时替换层 forward 真正避免该层计算，完成后恢复。该方式用于无缓存诊断，不支持 KV cache 生成，也没有提供速度基准。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 python tools/evaluate_router_corrected.py \
  --base-model /home/liushenyu/opt-13b \
  --checkpoint /home/liushenyu/test/skip/transformer_router_raw_embed_epoch_20.pth \
  --prompts analysis/smoke_prompts.jsonl \
  --k 6 16 --device cuda:0 --output analysis/kl_diagnostic_rerun.json
```

示例中的三个提示仅用于发现方向错误，不构成 PIQA、独立语料或论文评测。GPU 编号应按实际空闲情况配置。

所有周报和旧程序中的“计划”“建议”“最优”等表述均是待核对的历史研究内容；本次整理的执行约束来自用户明确要求：**不修改源文件；后续变更只新增程序并单独运行**。
