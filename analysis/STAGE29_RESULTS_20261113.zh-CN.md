# 阶段29：EvoPress 与冻结动态路由的完整横评

比较 OPT-13B 在剪去12/16/20/24个完整解码层（30%/40%/50%/60%）后的质量。
所有基座权重冻结，无恢复微调；保护原始层0/1，每道题所有选项共用一次完整提示词路由。

EvoPress 使用整层模式、原生全序列 next-token 教师KL、多阶段候选筛选和精英保留。
校准来自原TRAIN文档，未用任务分数选择静态组合或动态检查点。主实验为64初始候选、32子代、64代、三搜索种子；
筛选预算为2048/32768输入token（末档不超过可用TRAIN token）。这一预算低于论文脚本，结果只代表本次适配与搜索预算。
既有静态联合KL以32个续写位置为目标，EvoPress为全序列目标；强静态bank还使用额外TRAIN搜索，不能据此声称等搜索成本。
SLEB、ShortGPT、Shortened也为冻结OPT适配版本，没有复现各论文全部训练配方。

动态主参照固定为Stage13 m=6，各K和种子均使用先前选定检查点；K20另列Stage19 bank8路由。没有新训练或按横评分数挑种子。
结果按独立种子平均，不是集成投票。保留静态联合KL与K20强静态，以区分文献比较和输入适配收益。

前三项选择题指标为准确率（PIQA/HellaSwag使用acc_norm），Wiki为token加权条件PPL，越低越好。
这些是已使用过的任务人口，并非新保留终测；Wiki不是标准rolling PPL。小试的8题/任务仅检查工程流程。

| 任务 | K | 方法 | 种子数 | 指标 |
|---|---:|---|---:|---:|
| hellaswag | 0 | full | 1 | 69.906% |
| hellaswag | 12 | evopress_whole_opt | 3 | 61.800% |
| hellaswag | 12 | m06 | 3 | 61.200% |
| hellaswag | 12 | random_static_seed579 | 1 | 57.757% |
| hellaswag | 12 | shortened_ppl_opt | 1 | 26.867% |
| hellaswag | 12 | shortened_ppl_plus | 1 | 27.216% |
| hellaswag | 12 | shortgpt_bi_opt | 1 | 26.867% |
| hellaswag | 12 | sleb_nll_opt | 1 | 59.898% |
| hellaswag | 12 | static_joint_kl | 1 | 61.900% |
| hellaswag | 16 | evopress_whole_opt | 3 | 52.506% |
| hellaswag | 16 | m06 | 3 | 51.736% |
| hellaswag | 16 | random_static_seed579 | 1 | 47.112% |
| hellaswag | 16 | shortened_ppl_opt | 1 | 26.718% |
| hellaswag | 16 | shortened_ppl_plus | 1 | 27.156% |
| hellaswag | 16 | shortgpt_bi_opt | 1 | 26.628% |
| hellaswag | 16 | sleb_nll_opt | 1 | 48.526% |
| hellaswag | 16 | static_joint_kl | 1 | 50.856% |
| hellaswag | 20 | codebook8_train256_prompt | 3 | 42.976% |
| hellaswag | 20 | evopress_whole_opt | 3 | 41.950% |
| hellaswag | 20 | m06 | 3 | 40.931% |
| hellaswag | 20 | random_static_seed579 | 1 | 30.412% |
| hellaswag | 20 | shortened_ppl_opt | 1 | 25.891% |
| hellaswag | 20 | shortened_ppl_plus | 1 | 26.240% |
| hellaswag | 20 | shortgpt_bi_opt | 1 | 25.802% |
| hellaswag | 20 | sleb_nll_opt | 1 | 40.012% |
| hellaswag | 20 | static_joint_kl | 1 | 41.575% |
| hellaswag | 20 | static_refined_bank_train800 | 1 | 44.244% |
| hellaswag | 24 | evopress_whole_opt | 3 | 30.751% |
| hellaswag | 24 | m06 | 3 | 32.128% |
| hellaswag | 24 | random_static_seed579 | 1 | 28.421% |
| hellaswag | 24 | shortened_ppl_opt | 1 | 26.031% |
| hellaswag | 24 | shortened_ppl_plus | 1 | 25.712% |
| hellaswag | 24 | shortgpt_bi_opt | 1 | 26.250% |
| hellaswag | 24 | sleb_nll_opt | 1 | 31.468% |
| hellaswag | 24 | static_joint_kl | 1 | 30.562% |
| piqa | 0 | full | 1 | 76.605% |
| piqa | 12 | evopress_whole_opt | 3 | 73.105% |
| piqa | 12 | m06 | 3 | 72.543% |
| piqa | 12 | random_static_seed579 | 1 | 70.022% |
| piqa | 12 | shortened_ppl_opt | 1 | 50.816% |
| piqa | 12 | shortened_ppl_plus | 1 | 52.067% |
| piqa | 12 | shortgpt_bi_opt | 1 | 50.816% |
| piqa | 12 | sleb_nll_opt | 1 | 72.579% |
| piqa | 12 | static_joint_kl | 1 | 73.667% |
| piqa | 16 | evopress_whole_opt | 3 | 68.879% |
| piqa | 16 | m06 | 3 | 68.517% |
| piqa | 16 | random_static_seed579 | 1 | 67.029% |
| piqa | 16 | shortened_ppl_opt | 1 | 50.707% |
| piqa | 16 | shortened_ppl_plus | 1 | 51.360% |
| piqa | 16 | shortgpt_bi_opt | 1 | 50.707% |
| piqa | 16 | sleb_nll_opt | 1 | 67.573% |
| piqa | 16 | static_joint_kl | 1 | 69.478% |
| piqa | 20 | codebook8_train256_prompt | 3 | 64.744% |
| piqa | 20 | evopress_whole_opt | 3 | 63.584% |
| piqa | 20 | m06 | 3 | 62.532% |
| piqa | 20 | random_static_seed579 | 1 | 54.026% |
| piqa | 20 | shortened_ppl_opt | 1 | 50.544% |
| piqa | 20 | shortened_ppl_plus | 1 | 52.231% |
| piqa | 20 | shortgpt_bi_opt | 1 | 50.544% |
| piqa | 20 | sleb_nll_opt | 1 | 61.099% |
| piqa | 20 | static_joint_kl | 1 | 63.112% |
| piqa | 20 | static_refined_bank_train800 | 1 | 64.472% |
| piqa | 24 | evopress_whole_opt | 3 | 55.731% |
| piqa | 24 | m06 | 3 | 57.744% |
| piqa | 24 | random_static_seed579 | 1 | 52.666% |
| piqa | 24 | shortened_ppl_opt | 1 | 50.653% |
| piqa | 24 | shortened_ppl_plus | 1 | 50.925% |
| piqa | 24 | shortgpt_bi_opt | 1 | 50.218% |
| piqa | 24 | sleb_nll_opt | 1 | 55.550% |
| piqa | 24 | static_joint_kl | 1 | 55.550% |
| wikitext_conditional | 0 | full | 1 | 11.3768 |
| wikitext_conditional | 12 | evopress_whole_opt | 3 | 17.8950 |
| wikitext_conditional | 12 | m06 | 3 | 18.3059 |
| wikitext_conditional | 12 | random_static_seed579 | 1 | 24.7858 |
| wikitext_conditional | 12 | shortened_ppl_opt | 1 | 6857.6302 |
| wikitext_conditional | 12 | shortened_ppl_plus | 1 | 2255.1320 |
| wikitext_conditional | 12 | shortgpt_bi_opt | 1 | 6857.6302 |
| wikitext_conditional | 12 | sleb_nll_opt | 1 | 19.4433 |
| wikitext_conditional | 12 | static_joint_kl | 1 | 17.2691 |
| wikitext_conditional | 16 | evopress_whole_opt | 3 | 28.1839 |
| wikitext_conditional | 16 | m06 | 3 | 29.8982 |
| wikitext_conditional | 16 | random_static_seed579 | 1 | 44.8982 |
| wikitext_conditional | 16 | shortened_ppl_opt | 1 | 5604.5491 |
| wikitext_conditional | 16 | shortened_ppl_plus | 1 | 4526.6762 |
| wikitext_conditional | 16 | shortgpt_bi_opt | 1 | 5552.5822 |
| wikitext_conditional | 16 | sleb_nll_opt | 1 | 32.2621 |
| wikitext_conditional | 16 | static_joint_kl | 1 | 27.4629 |
| wikitext_conditional | 20 | codebook8_train256_prompt | 3 | 51.8861 |
| wikitext_conditional | 20 | evopress_whole_opt | 3 | 51.6168 |
| wikitext_conditional | 20 | m06 | 3 | 59.2855 |
| wikitext_conditional | 20 | random_static_seed579 | 1 | 918.1497 |
| wikitext_conditional | 20 | shortened_ppl_opt | 1 | 11523.3124 |
| wikitext_conditional | 20 | shortened_ppl_plus | 1 | 3756.0933 |
| wikitext_conditional | 20 | shortgpt_bi_opt | 1 | 8638.2360 |
| wikitext_conditional | 20 | sleb_nll_opt | 1 | 65.3370 |
| wikitext_conditional | 20 | static_joint_kl | 1 | 53.5313 |
| wikitext_conditional | 20 | static_refined_bank_train800 | 1 | 50.6287 |
| wikitext_conditional | 24 | evopress_whole_opt | 3 | 260.1314 |
| wikitext_conditional | 24 | m06 | 3 | 419.5820 |
| wikitext_conditional | 24 | random_static_seed579 | 1 | 2460.2109 |
| wikitext_conditional | 24 | shortened_ppl_opt | 1 | 5735.0347 |
| wikitext_conditional | 24 | shortened_ppl_plus | 1 | 5402.7847 |
| wikitext_conditional | 24 | shortgpt_bi_opt | 1 | 10009.1515 |
| wikitext_conditional | 24 | sleb_nll_opt | 1 | 394.3958 |
| wikitext_conditional | 24 | static_joint_kl | 1 | 413.2684 |
| winogrande | 0 | full | 1 | 66.851% |
| winogrande | 12 | evopress_whole_opt | 3 | 60.116% |
| winogrande | 12 | m06 | 3 | 59.353% |
| winogrande | 12 | random_static_seed579 | 1 | 59.669% |
| winogrande | 12 | shortened_ppl_opt | 1 | 50.908% |
| winogrande | 12 | shortened_ppl_plus | 1 | 49.329% |
| winogrande | 12 | shortgpt_bi_opt | 1 | 50.908% |
| winogrande | 12 | sleb_nll_opt | 1 | 58.169% |
| winogrande | 12 | static_joint_kl | 1 | 60.142% |
| winogrande | 16 | evopress_whole_opt | 3 | 54.380% |
| winogrande | 16 | m06 | 3 | 55.591% |
| winogrande | 16 | random_static_seed579 | 1 | 53.828% |
| winogrande | 16 | shortened_ppl_opt | 1 | 49.645% |
| winogrande | 16 | shortened_ppl_plus | 1 | 51.066% |
| winogrande | 16 | shortgpt_bi_opt | 1 | 50.592% |
| winogrande | 16 | sleb_nll_opt | 1 | 52.960% |
| winogrande | 16 | static_joint_kl | 1 | 55.012% |
| winogrande | 20 | codebook8_train256_prompt | 3 | 52.644% |
| winogrande | 20 | evopress_whole_opt | 3 | 51.355% |
| winogrande | 20 | m06 | 3 | 53.433% |
| winogrande | 20 | random_static_seed579 | 1 | 50.197% |
| winogrande | 20 | shortened_ppl_opt | 1 | 50.908% |
| winogrande | 20 | shortened_ppl_plus | 1 | 49.803% |
| winogrande | 20 | shortgpt_bi_opt | 1 | 50.592% |
| winogrande | 20 | sleb_nll_opt | 1 | 51.223% |
| winogrande | 20 | static_joint_kl | 1 | 52.328% |
| winogrande | 20 | static_refined_bank_train800 | 1 | 53.591% |
| winogrande | 24 | evopress_whole_opt | 3 | 51.539% |
| winogrande | 24 | m06 | 3 | 50.566% |
| winogrande | 24 | random_static_seed579 | 1 | 50.118% |
| winogrande | 24 | shortened_ppl_opt | 1 | 49.645% |
| winogrande | 24 | shortened_ppl_plus | 1 | 50.908% |
| winogrande | 24 | shortgpt_bi_opt | 1 | 49.092% |
| winogrande | 24 | sleb_nll_opt | 1 | 49.882% |
| winogrande | 24 | static_joint_kl | 1 | 51.381% |

配对差异、未校正95%区间、逐种子指标和质量审计见同运行目录的summary.json/verification.json。
先在同题上平均独立种子，再按题重采样；Wiki按文章聚类。Wiki配对项使用窗口平均NLL，与token加权PPL另列。
不能因若干均值领先就宣称全面优于所有静态方法；搜索收敛、等预算恢复与其他骨干仍需分别验证。

参考：[EvoPress论文](https://arxiv.org/abs/2410.14649)，[官方实现](https://github.com/IST-DASLab/EvoPress)。
官方代码审阅固定到 a66a6974e91d27ede9043f07f885709732a9a9b1；本仓库仅发布独立适配程序，不上传第三方副本。

运行目录：`runs/stage29_literature_20261113_01_main`。
