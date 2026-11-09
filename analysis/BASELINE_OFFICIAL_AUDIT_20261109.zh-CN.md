# ShortGPT 与 Shortened LLaMA 官方实现核查

核查日期：2026-11-09。已找到官方仓库，不需要用户另行提供地址。本文核查官方README与剪枝关键代码，不将本项目当前OPT适配结果当作原论文完整复现。没有修改既有源程序、路径、检查点或结果。

## 官方来源和固定版本

| 方法 | 仓库 | 本次核查提交 |
| --- | --- | --- |
| ShortGPT | https://github.com/icip-cas/ShortGPT | `9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c` |
| Shortened LLaMA | https://github.com/Nota-NetsPresso/shortened-llm | `4031a9212aa0dbeec8f2ad504af9346c758b0147` |

关键证据均按以上提交定位，避免以后默认分支变化。

## ShortGPT

[`shortgpt/metrics.py`](https://github.com/icip-cas/ShortGPT/blob/9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c/shortgpt/metrics.py#L4)定义BI为层输入/输出的逐位置`1-cosine similarity`。[`shortgpt/shortgpt.py`](https://github.com/icip-cas/ShortGPT/blob/9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c/shortgpt/shortgpt.py#L26)积累相邻hidden states的BI，再按升序一次选出待删除层。[`run_shortgpt.py`](https://github.com/icip-cas/ShortGPT/blob/9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c/shortgpt/run_shortgpt.py#L33)展示排序、物理删层和重新设置保留层的attention layer_idx。

当前本项目对每个OPT decoder block直接采集输入/输出，按160个位置上的BI平均值排序，取前K层。统一使用128条C4训练校准文档，保护0/1层；未恢复训练。一次BI排名属于该方法的核心规则，不能因为深剪退化就改成条件联合贪心后仍称ShortGPT。

官方README示例包括Llama2-7B的21–29层和Llama2-13B的26–35层。这是特定模型的结果，不能直接照搬到OPT-13B，也不能由此认定本项目选中较浅层一定是索引错误。后续若扩大论文比较，应在官方支持的共同骨干上另做对照，检查padding统计、最后一层归一化的观测约定、校准数据和保护层策略。

## Shortened LLaMA

[`src/anal_block_sensitivity_ppl.py`](https://github.com/Nota-NetsPresso/shortened-llm/blob/4031a9212aa0dbeec8f2ad504af9346c758b0147/src/anal_block_sensitivity_ppl.py#L73)在完整模型上逐个单独删除block，使用BookCorpus校准样本的token NLL计算PPL，再对各block分数一次排序。它没有在每删一层后重新联合排名；当前本项目的PPL一次排序与这一核心规则对应。

[`src/block_prune.py`](https://github.com/Nota-NetsPresso/shortened-llm/blob/4031a9212aa0dbeec8f2ad504af9346c758b0147/src/block_prune.py#L67)默认启用plus heuristic，保护前4层与最后2层；只有`no_plus_heuristic`才关闭。对应40层OPT时是0/1/2/3/38/39，本项目`Shortened-PPL+`的保护集合与这个规则一致。因此在介绍当前适配结果时，应把PPL+作为更接近官方默认保护规则的对照，同时保留普通PPL以披露保护策略影响。既有阶段7表格不覆盖修改，解释通过本文追加。

[`script/prune_llama-7b_crit-ppl.sh`](https://github.com/Nota-NetsPresso/shortened-llm/blob/4031a9212aa0dbeec8f2ad504af9346c758b0147/script/prune_llama-7b_crit-ppl.sh)明确串联敏感度分析、剪枝、LoRA恢复训练和评测。示例为10条校准、剪6层、LoRA rank8、2epochs、lr1e-4、batch64。官方README还说明严重剪枝时continued pretraining优于LoRA，并单独发布相应恢复模型。

当前本项目使用相同C4校准数据进行OPT适配，序列长160，按全部159个next-token NLL评分，没有执行LoRA/CPT恢复。因此已有数值是“统一数据下的OPT-13B PPL/PPL+剪枝准则，无恢复训练”的结果。与官方恢复后模型相比，基础模型、数据和训练预算均不相同。

## 目前可以说什么

“ShortGPT、Shortened当前适配版本退化较大”指记录在阶段6/7中的这些具体OPT适配配置，在较深剪层预算下出现质量下降。它既不是认定原论文方法普遍无效，也不是宣称已击败原论文公开的完整恢复模型。

现有比较可以讨论统一骨干、统一校准数据、冻结OPT权重下的层选择规则优劣。路由器另有800条训练文档与离线标签搜索成本，论文须披露，不能把“OPT权重都冻结”说成各方法总训练预算完全相等。静态联合KL贪心和SLEB适配继续作为强对照保留。

后续更严格对照需要明确两种问题：保持基础模型冻结时的剪枝准则比较；以及给定相近恢复数据和训练预算后的最终模型质量比较。后者需要新程序、新运行目录，给相关方案明确且可比的恢复预算，单独测质量、吞吐和显存。不能只替某一弱基线改校准设置、从已审阅测试分数择优，再与原表混写。

## 关键官方文件的SHA256

| 官方文件 | SHA256 |
| --- | --- |
| ShortGPT `shortgpt/shortgpt.py` | `60e4a3d4a3226b53291b063d4570351172ace652167b1f1ab119d01b8f109320` |
| ShortGPT `shortgpt/run_shortgpt.py` | `8f06e14f5287727446ca01bd17dc2b57a0816b0ffdee37a6aecdf92389b8e10a` |
| Shortened `script/prune_llama-7b_crit-ppl.sh` | `9351c28c52cf97db47d970db93a2123422a6e609ff5cd2aad166126e6ce220dd` |
| Shortened `src/anal_block_sensitivity_ppl.py` | `b61fd8a03766008b33c353cdc2534d4ccc6c87a833b2dfeb2486fb3569c1f3ca` |
| Shortened `src/block_prune.py` | `e90b3e3acd8d2bc04426c896687059b694dca1ddcf40ade1d769f5f6ba779718` |

只记录来源、方法核查及文件摘要；不向本项目上传第三方仓库副本。
