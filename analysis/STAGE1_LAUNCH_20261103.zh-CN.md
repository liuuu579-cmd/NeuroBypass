# 第一阶段已启动：既有权重的同预算 C4 评测

启动时间：2026-11-03 16:37（Asia/Shanghai）。本文件是启动记录，不表示实验已经完成。

- 运行目录：`/home/liushenyu/test/NeuroBypass/runs/stage1_c4_20261103_02/`。
- 新增执行程序：`tools/stage1_benchmark_v2.py`，经 `tools/run_with_resources_v2.py` 启动。
- 物理 GPU 1、3；两个独立 worker 每卡一份 OPT-13B；进程树总计限制 CPU 0–20，共 21 核；每 worker PyTorch 默认 10 线程。
- 8 个校准块，256 个测试块，每块 256 token；两个 split 来自本地 C4 文件中相互分离的 byte 区段。保留 source 区段哈希、token offset 和每条输入哈希。
- 校准集上迭代搜索固定静态组合，共 16 步；选择目标为平均末 token KL。该组合只根据校准集决定，不使用测试结果挑选。
- 测试比较 K=6、16 的修正 Transformer Router、原反向策略（诊断）、校准静态组合、3 次随机组合，以及 K=6 的旧 RL。旧 RL 同时记录历史 raw token 特征与 Layer-1 特征，避免掩盖训练/评测特征口径的不确定性。
- 指标包含末 token KL、所有位置平均 KL、有效预测位置平均 KL、next-token NLL 和 PPL、逐条剪枝 mask 和组合分布。
- 真正绕过 decoder block；无 KV cache，仅评价质量，不报告加速倍数。

首次版本在采样检查阶段遇到重复前缀，已自动报错退出，未加载权重。保留原程序和失败运行目录；新增 v2 后，按固定 seed 在不重叠 token 网格上采样，跳过重复块并补足样本。额外审计对包含 BOS 与去掉 BOS 的内容前缀均进行了检查，和最新 v3 训练前缀的匹配数为 0，校准/测试内部没有重复内容前缀。证据见 `stage1_data_verification_20261103.json`。

数据限制：这里保证的是不同 token 块，不保证不同原始文档；旧 feature-only 训练数据缺少完整 token provenance，无法穷尽排除历史数据重合。这不是外部全新语料。

小随机 OPT 上的预检已通过：KL 自一致性、next-token NLL 与 Transformers labels loss 一致、真正绕过选中 block、恢复原 forward，以及分片和统计聚合行为。预检结果保存在 `runs/stage1_preflight_20261103_01/`。

后台主进程启动后不依赖当前聊天持续打开。进度读取：

```bash
tail -n 20 /home/liushenyu/test/NeuroBypass/runs/stage1_c4_20261103_02/stdout.log
```

预期产物：`static_calibration_steps.jsonl`、`static_policy.json`、`sample_metrics.jsonl`、`summary.json`，最终退出状态见 `status.json`。语料 token 数组 `private_eval_token_ids.npz` 仅保留本地，不上传 GitHub；模型和 Router 权重也不上传。

该任务只启动评测。新的数据生成、SFT/RL 训练、逐输入贪心对照、输入打乱消融及完整 PIQA 将根据第一轮证据继续推进。
