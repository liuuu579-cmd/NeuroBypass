# NeuroBypass 研究进度与问题分析

整理日期：2026-11-03（Asia/Shanghai）。证据来自当前本地程序、文件修改时间、已有检查点、结果图，以及 2026-03-07、03-14、03-22 三份周报。本文区分“周报记述”“代码行为”“历史图中结果”和“本次执行验证”。没有重跑完整历史训练，也没有查询论文原文核验相关工作的叙述。

## 主要判断

这个项目的核心问题已经从“单层是否冗余”推进到“组合效应下，如何由输入选择多层跳过”。当前具体实现是带 K 条件的组合预测；K 是用户或外部策略给定的预算，尚不是经过校准的、由输入自主确定的质量约束。

目前不能仅凭剪枝效果不好就否定输入动态剪枝。最新 Transformer 路线有一个足以使评测失真的明确实现错误：**将保留分数最高的层作为待剪层**。同时，训练数据的有效文本覆盖很少、随机切分共享前缀、KL 只看最后一个位置，使现有指标很难回答真正关心的泛化与端到端质量问题。应先建立可信的同预算评测，再决定继续动态路线还是转为静态搜索。

## 按时间还原进展

修改时间只反映当前文件最后一次变更，不构成完整版本历史。原始记录见 [source_timeline.csv](source_timeline.csv) 与 [source_manifest.json](source_manifest.json)。附件 DOCX 在本次上传时都获得了 11 月的修改时间，因此周报按标题日期排序。

| 时间 | 证据 | 已落地内容或当时思路 |
| --- | --- | --- |
| 2025-07 至 2025-12 | `legacy/generate_trajectories*.py`、`train2*.py`、`predictor_model*.py` | 更早的轨迹、重要性与预测器探索；部分属于 NeuroFetcher / KV 复用前序工作，不能都解释为当前层剪枝实验 |
| 2026-03-04 至 03-07 | `legacy/generate_skip_scores.py`、`inference_test_skip.py`；03-07 周报 | 单层跳过 KL 数据和冗余热图；意识到单层指标不能直接组合，提出迭代贪心或 Gumbel/RL 掩码 |
| 2026-03-10 | `legacy/prove_synergy.py`、`generate_greedy_labels.py` | 对比独立层误差与联合跳层误差；生成固定剪 6 层的迭代贪心标签 |
| 2026-03-11 至 03-17 | `legacy/skip/train_router_sft_v1/v2.py`、`train_router_rl_v1/v2/v3.py` | 固定预算 SFT + RL；尝试相对奖励、多次采样等方式应对组合塌缩 |
| 2026-03-14 周报 | 原 DOCX | 记述 10,000 条“暴力穷举全局最优”训练数据、约 40% 标签重合率；此说法与现有主数据生成脚本的贪心实现不同，不能直接采用为已证实事实 |
| 2026-03-16 至 03-21 | `upgrade_zarr_features.py`、`skip/dataset_router_v3.py`；`router_sft_epoch_v3_10.pth`、`router_rl_epoch_v4_4.pth` | 特征转向 Layer 1 输出（索引 1，即完成前两个 block）；固定 6 层 SFT/RL 检查点存在；未找到对应的 `train_router_rl_v4.py` |
| 2026-03-22 周报 | 原 DOCX 与图 | RL 多在 `[2,9,5,11,14,7]` 里变化；提出提高预算至 16 层、保存贪心路径，并将剪枝层数作为输入 |
| 2026-03-23 至 03-25 | `generate_greedy_labels_multi_v2.py`、`skip/model_router.py`、`train_router_sft_v3.py`、`prove_global_optimum_v2.py` | 16 层标签探索；固定预算模型进一步调整。`prove_global_optimum_v2.py` 新增特征 MSE 迭代搜索，不能根据文件名认定其给出了全局最优证明 |
| 2026-05 至 06 月 | `skip/eval_router_task.py`、`my_router_model.py`、`run_eval.py` | 增加 PIQA 和 lm-evaluation-harness 接入探索 |
| 2026-09-29 | `legacy/generate_greedy_labels_multi_v3.py`、`run_workers.sh` | 最新四分片数据：仅存 token IDs、长度、16 步有序剪枝路径、每步 KL；目标 2,000 行 |
| 2026-10-01 至 10-02 | `skip/dataset_router_dynamic.py`、`model_router_v4.py`、`train_sft_dynamic.py`、`train_rl_dynamic.py` | Layer 1 特征平均池化 + K embedding 的条件 MLP；SFT 20 epoch 和 RL 10 epoch 检查点存在；K 最大 16 |
| 2026-10-04 至 10-09 | `eval_sft_router.py`、`eval_router_oldnew.py`、`eval_router_oldnew_task.py` 与结果图 | 固定 K 的新旧方案对比，以及按 logits margin 映射 K 的启发式尝试；图示自适应 K 最终全部为 5 |
| 2026-10-10 | `model_router_transformer.py`、`train_sft_dynamic_v2.py`；`transformer_router_raw_embed_epoch_20.pth` | 改为原始 token + position embedding → 2 层 Transformer Encoder（1024 维）+ K token → 40 个保留 logits；20 epoch 检查点存在 |
| 2026-10-13 | `skip/eval_router_oldnew_task_trans.py` | 最新 PIQA 评测代码；发现 top-k 方向相反、默认 K=5 对比静态 K=6、缺失权重时继续用随机模型等问题；目录内没有相应的新结果图/逐样本记录，不能断言该评测已完成 |

最新 Transformer 与 10-02 的条件 MLP 是不同模型和特征路线。现有 RL 训练程序针对 MLP，不能将其检查点当作 Transformer 的 RL 结果。

## 当前训练和推理实际在做什么

最新数据生成器保护层 0、1，在剩余 38 层中逐步搜索。每步固定此前已选的层，再尝试新增一层，以末 token 的 `KL(P_full || P_skipped)` 最小为准，重复 16 次。每条样本共需 `38+37+...+23=488` 次候选前向，加一次完整模型前向。保存 `ordered_skips` 和 `kl_divergences`，没有保存所有候选层的代价表。

Dataset 在每次读取时抽取 K：一半概率取 `KL<=0.5` 的最大已存步数，一半从 1 到 16 均匀采样；标签是将路径前 K 项置 0，其他层置 1。MLP 和 Transformer 都用 BCEWithLogitsLoss 拟合这个 **保留标签**。

Transformer 新路线仅提取 OPT 的词向量与位置向量，不经过原模型 decoder block；再投影至 1024 维，加入 K token，通过 2 层 encoder 得到 40 个 logits。此路线的“embedding 输出”与此前 Layer 1 后的上下文化特征不同。独立的 `model_router_transformer.py` 仍有 Layer 1 注释，但实际训练和最新评测均调用 raw embedding 提取函数。

对于给定 K，正确规则是屏蔽保护层后选保留 logits 最小的 K 层。改变 K 改变预算；只有在独立输入上验证所选组合确实更合适，才能说明输入依赖具有价值。

## 已确认的实现问题

### 1. 最新评测与训练命中率方向反了

证据：`legacy/skip/dataset_router_dynamic.py:42` 附近明确 `1=Keep,0=Skip`；`train_sft_dynamic_v2.py:221` 使用 BCE；该文件第 258、259、311、312 行却对预测与目标都 `topk(..., largest=True)`。目标 mask 中的 1 有大量并列，其 top-k 是任意选出的保留层子集，无法作为剪枝正确率。

`legacy/skip/eval_router_oldnew_task_trans.py` 在动态 Router 决策段同样用 `largest=True` 剪掉最高保留分数层。保护层置 `-inf` 对这个反向选择虽能保护 0、1，却没有纠正语义。正确做法是保护层置 `+inf`，再用 `largest=False`。

这不意味着 BCE 训练标签也需要翻转。**已有权重可以直接先用正确方向评测**；错误的训练日志命中率不代表权重一定没有学到东西。

本次独立 CPU 检查读取现有 `transformer_router_raw_embed_epoch_20.pth`，从每个不同的 128-token 前缀组取第一条已完成记录，共 20 条（均来自训练数据源）。结果见 [checkpoint_direction.json](checkpoint_direction.json)：

| K | 正确 bottom-k 与实际剪枝标签重合率 | 原 top-k 与实际剪枝标签重合率 | 修正方向的不同组合数 |
| --- | ---: | ---: | ---: |
| 6 | 46.67% | 0.00% | 14 / 20 |
| 16 | 54.06% | 27.50% | 20 / 20 |

这些是标签重合诊断，不能解释为下游准确率或泛化提升。理想标签分类器的独立构造检验同样得到 bottom-k 命中 6/6、旧 top-k 命中 0/6。

进一步使用现有 OPT-13B 权重，在三个本次新写的英文提示上执行真实跳层、FP16 主干与 float32 末 token KL，结果见 [corrected_kl_smoke.json](corrected_kl_smoke.json)：

| K | 原方向平均 KL | 修正方向平均 KL |
| --- | ---: | ---: |
| 6 | 0.5729 | 0.6033 |
| 16 | 2.7282 | 1.8794 |

K=16 有改善，K=6 没有改善。三个短提示只构成执行检查，无法据此推断总体质量。它也直接说明：纠正语义是必要的实现修复，不能保证现有 Router 权重在新输入上有效。

### 2. 最新数据覆盖少，随机划分共享前缀

[data_audit.json](data_audit.json) 对四分片逐条只读检查：2,000 行全部完成，长度、层索引、每路径去重和 KL 有限性检查通过；但只存在 **20 个不同的起始 128-token 前缀组**，其中 16 组各 121 个嵌套前缀，4 组各 16 个。未发现同前缀同长度的重复行。

这与生成器一致：每个 2,048-token chunk 取长度 128、144、…、2,048，共 121 条；每分片只收 500 行，因此很快达到目标。不能将 2,000 条看作 2,000 个独立文本样本，更不能视为覆盖了整个 1 GB C4。

复现训练脚本 seed=42 的 `random_split(1800,200)`，200/200 验证行的前缀组都出现在训练集中。验证集含 19 组，训练集含全部 20 组。这支持“验证泛化指标可能乐观”的判断，但不能据此推断独立 PIQA 也有泄漏。

建议先在原始文档/不同 chunk 层面分组，再做训练/验证/测试切分；对同一个 chunk 生成的所有前缀固定使用同一个 split。20 个组仍太少，需要扩大独立文本覆盖。

### 3. Post-forward hook 不会减少层计算

原程序使用 `register_forward_hook`，在 block 完成后把输出换成输入。这可模拟被跳层后的 hidden state，但完整 attention、MLP 和部分缓存操作已经发生。因此这些程序产生的 KL 不等于真实吞吐提升；注释中剪 6 层“约 15% 加速”没有速度测量依据。

新增 `tools/evaluate_router_corrected.py` 临时把选中层的 forward 替换为 identity（仅无缓存诊断），不触碰磁盘源文件。小模型自检验证它与输出替换的结果相同、对应 MLP 调用次数为 0，并在退出后恢复原 forward。生产批处理、KV cache 及 GPU 计时仍需单独设计。

### 4. RL 在右侧 padding 的最后位置算奖励

`legacy/skip/train_rl_dynamic.py` 用批内变长右侧 padding（BATCH_SIZE=2），但完整与剪枝分支都取 `logits[:, -1, :]`。对较短样本，这不是最后一个有效 token。传入 attention_mask 并不会把输出位置自动移动；这会让奖励与预期的“该输入下一词”不一致。应按有效长度取最后有效位置或对有效 token 聚合 KL。

同一 RL 程序使用独立 Bernoulli 动作，再用数量惩罚接近 K；评测则用 exact-K bottom-k。训练动作分布与推理动作规则不同，也可能损害效果，但影响大小需要对照实验。

### 5. 单点 FP16 KL 含数值噪声，难度阈值缺少区分度

最新数据中 32,000 个 KL 值有 **478 个为负**，最小 -0.0007181。KL 理论非负；生成器直接对 FP16 logits 做 softmax/log-softmax，在很小的真实 KL 区域数值误差会干扰贪心排序。新评测先转换 float32，建议重算少量小 KL 候选与相近候选的排序，量化标签稳定性。

阈值 0.5 下，**1,692 / 2,000（84.6%）** 的轨迹在 K=16 仍符合条件，最大有效步数严重靠近上界。最大 K 的采样方式因此给多数输入同一个 K，难以学到有区分度的难度映射。与此同时，1,379 条轨迹出现了超过 1e-6 的 KL 下降步；KL 随 K 并不保证单调，不能无条件用“第一次越阈”或二分搜索解释当前轨迹。

### 6. 基线和预算在最新评测中不一致

最新脚本默认 Router K=5，却与固定 K=6 的静态基线比较；ShortGPT 被硬编码为 `[24,25,26,27,28,29]`，而前一个 PIQA 脚本写的是 `[3,7,12,15,22,28]`。这两组都没有在对应脚本里按同一校准集重新计算重要性，不能仅靠名称作为忠实论文复现。

旧 C4 KL 脚本确实有 Block Influence 校准流程，但其方法、保护层范围、静态搜索预算与最新脚本不同，不能跨图直接得出同协议胜负。最新脚本还会在缺失 checkpoint 时用随机模型继续运行，应该失败退出。

## 历史结果应该怎样解读

以下数字仅从现有图片读取，未重跑原实验；不能与本次修正诊断混为一谈。

| 记录 | Static KL | ShortGPT 标签方法 KL | 旧 SFT KL | 旧 RL KL | 新条件 MLP KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 03-22 周报嵌图，剪 6 层 | 0.1388 | 4.9897 | 0.6234 | 0.0778 | 未列 |
| 10-08 `final_5_methods_comparison.png`，剪 6 层 | 0.1563 | 5.3737 | 1.0475 | 0.0864 | 0.7029 |

旧固定 6 层 RL 在这些图上比手选静态组合 KL 小，说明“固定预算 RL 完全没用”不符合现有证据；但它是否利用输入而非主要学习一个强静态组合，还没有被合适的消融回答。新条件 MLP 的相同预算 KL 则弱于旧 RL 和静态。

10-08 的 200 道 PIQA 图为 Full 72.5%、Static 73.5%、ShortGPT 标签方法 72.5%、旧 SFT 73.0%、旧 RL 74.0%、新 MLP K=6 72.5%。10-09 的启发式深度图把新 MLP 提到 74.0%，但 **200 道题全部剪 5 层**；该图只能说明这次策略输出固定深度，不能说明实现了输入自适应的 K。组合是否变化是另一个问题。

200 题中 1.5 个百分点对应 3 题。没有逐样本配对结果、置信区间和同预算比较，不能把这种差异直接用于论文优势结论。

周报里的“全局最优”也应统一口径：当前主要生成器是贪心；有限候选池穷举最多只能得到候选池内最优；命中率是集合重合，不是 40 分类准确率，也不说明不同标签组合的真实 KL 差距。

## 下一步最有价值的实验顺序

1. **先重评已有 Transformer 权重。** 固定 K=6、16，在独立文本上比较原方向、修正方向、同预算静态；先保留逐条 mask 与 KL，确认实现正确后再跑完整 PIQA、语言建模困惑度和跨任务测试。新增脚本可完成方向与 KL 诊断，尚不提供这些完整任务评测。
2. **量化动态选择的实际空间。** 在训练之外的校准集搜索一个强静态组合；在完全独立测试集计算强静态、逐样本贪心参考和 Router 的 KL。若逐样本参考也只略优于强静态，则不应只靠放大 Router 来解决；若参考明显更优而 Router 不佳，重点在数据与学习目标。
3. **做输入消融。** 同 K 下使用真实输入、打乱输入与 mask 的对应、仅输入 K/常量特征。比较质量并记录 mask 众数比例、每层选择频率、不同组合数。输出会变只是必要线索；对同样预算真正优于打乱或常量输入才支持输入依赖的价值。
4. **增加独立文本并按组切分。** 每个 chunk 少取一些前缀，让相同预算覆盖更多文档；保存 chunk/document ID、split、完成标志、生成版本、模型版本与随机种子。float32 KL，异常必须输出错误，hooks 必须 finally 清理。生成器当前的 `except Exception: pass` 和恢复后从头扫描 chunk 也需要在新增版本中处理。
5. **分开学习组合与深度。** 先研究给定 K 的组合选择；再用单独验证集校准质量/风险预测头或 K 选择器，替代未经校准的 logits margin 线性映射。允许 K=0，按实际质量曲线选择阈值，同时验证 0.5 阈值是否合理。
6. **若需要更好处理层间交互，再试条件序列决策。** 下一层的选择显式依赖此前已选层集合，或预测候选组合的代价；保留多个近似等价标签，避免对相差很小的唯一贪心路径过拟合。用质量改进决定是否采用 RL，同时对齐 exact-K 动作与推理规则。
7. **最后评测真实成本。** 使用真正跳过 block 的执行路径，分别测 prefill、decode、Router 开销、批处理和 KV cache。动态跳层通常仍需把各层权重常驻显存，不能直接沿用静态删层的模型体积收益；质量与速度需要一并报告。

在这些结果出现前，最稳妥的项目状态是：**固定预算的动态组合预测与可变 K 条件建模已有实现；原始 embedding 的 Transformer SFT 检查点存在；独立泛化、输入自适应 K 的收益以及实际加速尚未得到可信验证。**

## 本次执行与产物边界

- 在新建 `NeuroBypass/` 中复制 93 个现有文件，保存原始 SHA-256 与修改时间；未编辑源程序、训练数据、权重或原周报。
- 运行 `audit_dynamic_data.py` 只读检查最新 Zarr；运行 `check_checkpoint_direction.py` 在 CPU 上读取原检查点和 embedding 表，进行方向与集合重合诊断。
- 新增修正评测程序通过小随机 OPT 的语义、真正绕过计算、forward 恢复、右 padding 最后有效 token 和 KL 自一致性检查。
- 三个本次新写的英文提示用于小规模 OPT-13B 方向诊断，输入见 `smoke_prompts.jsonl`；完整实测数据以生成的 `corrected_kl_smoke.json` 为准。此检查不能代替独立数据集或任务性能评测。
- 仓库只上传研究程序、报告、图片与小型审计结果，不包含大模型、Router 权重、C4 或 Zarr。所用依赖环境版本已记录；完整历史训练依赖与运行命令仍需后续逐项复现。
