# 完整提示词路由修复与重训重测

旧推理入口仅观察前128 token。新版本读取基座实际可见的完整提示词，超长时显式保留最近上下文，路由与prefill使用完全相同的token。答案和未来续写不进入预测器。

旧训练本来只有128-token提示词；旧短题与Wiki128-token条件上下文结果保留其原协议含义。长提示词成本测试存在截断，须以本轮新部署结果为准。新旧训练分布和静态校准均有变化，因此差异不能仅归因于修复一行截断。

新训练保留原800/100文档划分；预留100测试文档未使用。m=0/4/6/8，各三个种子，逐步条件联合KL贪心与替代分支，冻结基座，无恢复微调。全部运行经4卡/42核资源与旧文件哈希审计。

## 开发集输入匹配诊断

| 任务 | m | K | matched KL | shuffled KL | 改善及95%CI |
|---|---:|---:|---:|---:|---:|
| c4_development | 6 | 10 | 0.250214 | 0.257929 | 0.007716 [0.001618, 0.013785] |
| c4_development | 6 | 12 | 0.399794 | 0.407412 | 0.007619 [-0.001658, 0.016725] |
| hellaswag | 6 | 10 | 0.224618 | 0.226697 | 0.002079 [-0.003420, 0.007457] |
| hellaswag | 6 | 12 | 0.347035 | 0.351837 | 0.004802 [-0.001612, 0.011502] |
| piqa | 6 | 10 | 0.238909 | 0.240457 | 0.001548 [-0.005843, 0.009007] |
| piqa | 6 | 12 | 0.359159 | 0.359190 | 0.000031 [-0.012325, 0.011887] |
| wikitext_conditional | 6 | 10 | 0.310907 | 0.310722 | -0.000184 [-0.003770, 0.003430] |
| wikitext_conditional | 6 | 12 | 0.482631 | 0.482089 | -0.000542 [-0.004839, 0.004028] |
| winogrande | 6 | 10 | 0.195962 | 0.204965 | 0.009004 [-0.000971, 0.019753] |
| winogrande | 6 | 12 | 0.301964 | 0.308934 | 0.006970 [-0.007403, 0.021931] |

先在输入内平均模型种子和三次置换，再按问题/文档bootstrap；Wiki按文章聚类。完整任务准确率、长度分组及全部m/K见paired_summary。仍属开发诊断；跨长度供体不能证明仅由内容匹配产生收益。

## 完整横向评测

| 任务 | 方法 | seed | K | 质量指标 |
|---|---|---:|---:|---:|
| hellaswag | full | None | 0 | 0.699064 |
| hellaswag | m00 | 579 | 10 | 0.637622 |
| hellaswag | m00 | 579 | 12 | 0.595598 |
| hellaswag | m00 | 579 | 16 | 0.491237 |
| hellaswag | m00 | 580 | 10 | 0.629357 |
| hellaswag | m00 | 580 | 12 | 0.587931 |
| hellaswag | m00 | 580 | 16 | 0.483768 |
| hellaswag | m00 | 581 | 10 | 0.628162 |
| hellaswag | m00 | 581 | 12 | 0.592014 |
| hellaswag | m00 | 581 | 16 | 0.490042 |
| hellaswag | m04 | 579 | 10 | 0.650070 |
| hellaswag | m04 | 579 | 12 | 0.613125 |
| hellaswag | m04 | 579 | 16 | 0.510157 |
| hellaswag | m04 | 580 | 10 | 0.648277 |
| hellaswag | m04 | 580 | 12 | 0.616909 |
| hellaswag | m04 | 580 | 16 | 0.518323 |
| hellaswag | m04 | 581 | 10 | 0.645589 |
| hellaswag | m04 | 581 | 12 | 0.613424 |
| hellaswag | m04 | 581 | 16 | 0.509958 |
| hellaswag | m06 | 579 | 10 | 0.651165 |
| hellaswag | m06 | 579 | 12 | 0.615216 |
| hellaswag | m06 | 579 | 16 | 0.518323 |
| hellaswag | m06 | 580 | 10 | 0.653157 |
| hellaswag | m06 | 580 | 12 | 0.611034 |
| hellaswag | m06 | 580 | 16 | 0.517825 |
| hellaswag | m06 | 581 | 10 | 0.650866 |
| hellaswag | m06 | 581 | 12 | 0.609739 |
| hellaswag | m06 | 581 | 16 | 0.515933 |
| hellaswag | m08 | 579 | 10 | 0.656244 |
| hellaswag | m08 | 579 | 12 | 0.607250 |
| hellaswag | m08 | 579 | 16 | 0.512647 |
| hellaswag | m08 | 580 | 10 | 0.652659 |
| hellaswag | m08 | 580 | 12 | 0.613523 |
| hellaswag | m08 | 580 | 16 | 0.510954 |
| hellaswag | m08 | 581 | 10 | 0.647879 |
| hellaswag | m08 | 581 | 12 | 0.611830 |
| hellaswag | m08 | 581 | 16 | 0.510755 |
| hellaswag | random_static_seed579 | None | 10 | 0.592412 |
| hellaswag | random_static_seed579 | None | 12 | 0.577574 |
| hellaswag | random_static_seed579 | None | 16 | 0.471121 |
| hellaswag | shortened_ppl_opt | None | 10 | 0.264190 |
| hellaswag | shortened_ppl_opt | None | 12 | 0.268672 |
| hellaswag | shortened_ppl_opt | None | 16 | 0.267178 |
| hellaswag | shortened_ppl_plus | None | 10 | 0.284903 |
| hellaswag | shortened_ppl_plus | None | 12 | 0.272157 |
| hellaswag | shortened_ppl_plus | None | 16 | 0.271559 |
| hellaswag | shortgpt_bi_opt | None | 10 | 0.268572 |
| hellaswag | shortgpt_bi_opt | None | 12 | 0.268672 |
| hellaswag | shortgpt_bi_opt | None | 16 | 0.266282 |
| hellaswag | sleb_nll_opt | None | 10 | 0.633240 |
| hellaswag | sleb_nll_opt | None | 12 | 0.598984 |
| hellaswag | sleb_nll_opt | None | 16 | 0.485262 |
| hellaswag | static_joint_kl | None | 10 | 0.656443 |
| hellaswag | static_joint_kl | None | 12 | 0.619000 |
| hellaswag | static_joint_kl | None | 16 | 0.508564 |
| piqa | full | None | 0 | 0.766050 |
| piqa | m00 | 579 | 10 | 0.740479 |
| piqa | m00 | 579 | 12 | 0.724157 |
| piqa | m00 | 579 | 16 | 0.678999 |
| piqa | m00 | 580 | 10 | 0.735582 |
| piqa | m00 | 580 | 12 | 0.710555 |
| piqa | m00 | 580 | 16 | 0.659412 |
| piqa | m00 | 581 | 10 | 0.737758 |
| piqa | m00 | 581 | 12 | 0.708923 |
| piqa | m00 | 581 | 16 | 0.656692 |
| piqa | m04 | 579 | 10 | 0.743199 |
| piqa | m04 | 579 | 12 | 0.728509 |
| piqa | m04 | 579 | 16 | 0.681175 |
| piqa | m04 | 580 | 10 | 0.744287 |
| piqa | m04 | 580 | 12 | 0.724157 |
| piqa | m04 | 580 | 16 | 0.676823 |
| piqa | m04 | 581 | 10 | 0.744287 |
| piqa | m04 | 581 | 12 | 0.731230 |
| piqa | m04 | 581 | 16 | 0.685528 |
| piqa | m06 | 579 | 10 | 0.740479 |
| piqa | m06 | 579 | 12 | 0.723613 |
| piqa | m06 | 579 | 16 | 0.687160 |
| piqa | m06 | 580 | 10 | 0.737214 |
| piqa | m06 | 580 | 12 | 0.727421 |
| piqa | m06 | 580 | 16 | 0.677367 |
| piqa | m06 | 581 | 10 | 0.747552 |
| piqa | m06 | 581 | 12 | 0.725245 |
| piqa | m06 | 581 | 16 | 0.690968 |
| piqa | m08 | 579 | 10 | 0.750816 |
| piqa | m08 | 579 | 12 | 0.730141 |
| piqa | m08 | 579 | 16 | 0.676823 |
| piqa | m08 | 580 | 10 | 0.741023 |
| piqa | m08 | 580 | 12 | 0.722524 |
| piqa | m08 | 580 | 16 | 0.684984 |
| piqa | m08 | 581 | 10 | 0.742655 |
| piqa | m08 | 581 | 12 | 0.724157 |
| piqa | m08 | 581 | 16 | 0.664853 |
| piqa | random_static_seed579 | None | 10 | 0.715452 |
| piqa | random_static_seed579 | None | 12 | 0.700218 |
| piqa | random_static_seed579 | None | 16 | 0.670294 |
| piqa | shortened_ppl_opt | None | 10 | 0.523939 |
| piqa | shortened_ppl_opt | None | 12 | 0.508161 |
| piqa | shortened_ppl_opt | None | 16 | 0.507073 |
| piqa | shortened_ppl_plus | None | 10 | 0.521763 |
| piqa | shortened_ppl_plus | None | 12 | 0.520675 |
| piqa | shortened_ppl_plus | None | 16 | 0.513602 |
| piqa | shortgpt_bi_opt | None | 10 | 0.515234 |
| piqa | shortgpt_bi_opt | None | 12 | 0.508161 |
| piqa | shortgpt_bi_opt | None | 16 | 0.507073 |
| piqa | sleb_nll_opt | None | 10 | 0.748640 |
| piqa | sleb_nll_opt | None | 12 | 0.725789 |
| piqa | sleb_nll_opt | None | 16 | 0.675734 |
| piqa | static_joint_kl | None | 10 | 0.748640 |
| piqa | static_joint_kl | None | 12 | 0.736670 |
| piqa | static_joint_kl | None | 16 | 0.694777 |
| wikitext_conditional | full | None | 0 | 11.376821 |
| wikitext_conditional | m00 | 579 | 10 | 15.753423 |
| wikitext_conditional | m00 | 579 | 12 | 19.609132 |
| wikitext_conditional | m00 | 579 | 16 | 33.973572 |
| wikitext_conditional | m00 | 580 | 10 | 15.839242 |
| wikitext_conditional | m00 | 580 | 12 | 19.407954 |
| wikitext_conditional | m00 | 580 | 16 | 32.804344 |
| wikitext_conditional | m00 | 581 | 10 | 15.606555 |
| wikitext_conditional | m00 | 581 | 12 | 18.972597 |
| wikitext_conditional | m00 | 581 | 16 | 32.186113 |
| wikitext_conditional | m04 | 579 | 10 | 15.397252 |
| wikitext_conditional | m04 | 579 | 12 | 18.422581 |
| wikitext_conditional | m04 | 579 | 16 | 30.206839 |
| wikitext_conditional | m04 | 580 | 10 | 15.522583 |
| wikitext_conditional | m04 | 580 | 12 | 18.579005 |
| wikitext_conditional | m04 | 580 | 16 | 29.457289 |
| wikitext_conditional | m04 | 581 | 10 | 15.415199 |
| wikitext_conditional | m04 | 581 | 12 | 18.722150 |
| wikitext_conditional | m04 | 581 | 16 | 30.168627 |
| wikitext_conditional | m06 | 579 | 10 | 15.520077 |
| wikitext_conditional | m06 | 579 | 12 | 18.406882 |
| wikitext_conditional | m06 | 579 | 16 | 29.628096 |
| wikitext_conditional | m06 | 580 | 10 | 15.233064 |
| wikitext_conditional | m06 | 580 | 12 | 18.014756 |
| wikitext_conditional | m06 | 580 | 16 | 30.427557 |
| wikitext_conditional | m06 | 581 | 10 | 15.690240 |
| wikitext_conditional | m06 | 581 | 12 | 18.496031 |
| wikitext_conditional | m06 | 581 | 16 | 29.639054 |
| wikitext_conditional | m08 | 579 | 10 | 14.911861 |
| wikitext_conditional | m08 | 579 | 12 | 18.212536 |
| wikitext_conditional | m08 | 579 | 16 | 29.191592 |
| wikitext_conditional | m08 | 580 | 10 | 15.073602 |
| wikitext_conditional | m08 | 580 | 12 | 17.957913 |
| wikitext_conditional | m08 | 580 | 16 | 29.470566 |
| wikitext_conditional | m08 | 581 | 10 | 15.056207 |
| wikitext_conditional | m08 | 581 | 12 | 18.047164 |
| wikitext_conditional | m08 | 581 | 16 | 28.629905 |
| wikitext_conditional | random_static_seed579 | None | 10 | 21.615195 |
| wikitext_conditional | random_static_seed579 | None | 12 | 24.785802 |
| wikitext_conditional | random_static_seed579 | None | 16 | 44.898243 |
| wikitext_conditional | shortened_ppl_opt | None | 10 | 18301.354656 |
| wikitext_conditional | shortened_ppl_opt | None | 12 | 6857.630193 |
| wikitext_conditional | shortened_ppl_opt | None | 16 | 5604.549118 |
| wikitext_conditional | shortened_ppl_plus | None | 10 | 1949.961351 |
| wikitext_conditional | shortened_ppl_plus | None | 12 | 2255.131957 |
| wikitext_conditional | shortened_ppl_plus | None | 16 | 4526.676199 |
| wikitext_conditional | shortgpt_bi_opt | None | 10 | 8474.481881 |
| wikitext_conditional | shortgpt_bi_opt | None | 12 | 6857.630193 |
| wikitext_conditional | shortgpt_bi_opt | None | 16 | 5552.582231 |
| wikitext_conditional | sleb_nll_opt | None | 10 | 15.909004 |
| wikitext_conditional | sleb_nll_opt | None | 12 | 19.443280 |
| wikitext_conditional | sleb_nll_opt | None | 16 | 32.262056 |
| wikitext_conditional | static_joint_kl | None | 10 | 14.895376 |
| wikitext_conditional | static_joint_kl | None | 12 | 17.269108 |
| wikitext_conditional | static_joint_kl | None | 16 | 27.462915 |
| winogrande | full | None | 0 | 0.668508 |
| winogrande | m00 | 579 | 10 | 0.637727 |
| winogrande | m00 | 579 | 12 | 0.610892 |
| winogrande | m00 | 579 | 16 | 0.569850 |
| winogrande | m00 | 580 | 10 | 0.610103 |
| winogrande | m00 | 580 | 12 | 0.580900 |
| winogrande | m00 | 580 | 16 | 0.554065 |
| winogrande | m00 | 581 | 10 | 0.623520 |
| winogrande | m00 | 581 | 12 | 0.583268 |
| winogrande | m00 | 581 | 16 | 0.565114 |
| winogrande | m04 | 579 | 10 | 0.624309 |
| winogrande | m04 | 579 | 12 | 0.607735 |
| winogrande | m04 | 579 | 16 | 0.564325 |
| winogrande | m04 | 580 | 10 | 0.618785 |
| winogrande | m04 | 580 | 12 | 0.612470 |
| winogrande | m04 | 580 | 16 | 0.565114 |
| winogrande | m04 | 581 | 10 | 0.629834 |
| winogrande | m04 | 581 | 12 | 0.589582 |
| winogrande | m04 | 581 | 16 | 0.558800 |
| winogrande | m06 | 579 | 10 | 0.632202 |
| winogrande | m06 | 579 | 12 | 0.598264 |
| winogrande | m06 | 579 | 16 | 0.545383 |
| winogrande | m06 | 580 | 10 | 0.632202 |
| winogrande | m06 | 580 | 12 | 0.581689 |
| winogrande | m06 | 580 | 16 | 0.560379 |
| winogrande | m06 | 581 | 10 | 0.619574 |
| winogrande | m06 | 581 | 12 | 0.600631 |
| winogrande | m06 | 581 | 16 | 0.561957 |
| winogrande | m08 | 579 | 10 | 0.610892 |
| winogrande | m08 | 579 | 12 | 0.599053 |
| winogrande | m08 | 579 | 16 | 0.547751 |
| winogrande | m08 | 580 | 10 | 0.609313 |
| winogrande | m08 | 580 | 12 | 0.591160 |
| winogrande | m08 | 580 | 16 | 0.538279 |
| winogrande | m08 | 581 | 10 | 0.604578 |
| winogrande | m08 | 581 | 12 | 0.585635 |
| winogrande | m08 | 581 | 16 | 0.554065 |
| winogrande | random_static_seed579 | None | 10 | 0.587214 |
| winogrande | random_static_seed579 | None | 12 | 0.596685 |
| winogrande | random_static_seed579 | None | 16 | 0.538279 |
| winogrande | shortened_ppl_opt | None | 10 | 0.505919 |
| winogrande | shortened_ppl_opt | None | 12 | 0.509077 |
| winogrande | shortened_ppl_opt | None | 16 | 0.496448 |
| winogrande | shortened_ppl_plus | None | 10 | 0.512234 |
| winogrande | shortened_ppl_plus | None | 12 | 0.493291 |
| winogrande | shortened_ppl_plus | None | 16 | 0.510655 |
| winogrande | shortgpt_bi_opt | None | 10 | 0.501184 |
| winogrande | shortgpt_bi_opt | None | 12 | 0.509077 |
| winogrande | shortgpt_bi_opt | None | 16 | 0.505919 |
| winogrande | sleb_nll_opt | None | 10 | 0.604578 |
| winogrande | sleb_nll_opt | None | 12 | 0.581689 |
| winogrande | sleb_nll_opt | None | 16 | 0.529597 |
| winogrande | static_joint_kl | None | 10 | 0.610103 |
| winogrande | static_joint_kl | None | 12 | 0.601421 |
| winogrande | static_joint_kl | None | 16 | 0.550118 |

PIQA/HellaSwag报告acc_norm，WinoGrande为acc，Wiki为128-token条件窗口PPL。论文方法是无恢复的OPT适配版，不代表完整官方配方。

## 部署成本

| GPU任务 | 部署 | 实际输入长度 | 热TTFT秒 | 热端到端token/s | allocated峰值GiB |
|---:|---|---:|---:|---:|---:|
| 0 | selective_cpu_bank | 128–128 | 0.4896 | 36.439 | 18.249 |
| 0 | resident_dynamic | 128–128 | 0.0341 | 41.817 | 24.109 |
| 0 | full | 128–128 | 0.0401 | 31.659 | 24.158 |
| 0 | static_physical | 128–128 | 0.0308 | 41.843 | 18.239 |
| 1 | selective_cpu_bank | 512–512 | 0.7444 | 34.975 | 17.274 |
| 1 | resident_dynamic | 512–512 | 0.0804 | 42.631 | 24.307 |
| 1 | full | 512–512 | 0.1199 | 30.108 | 24.453 |
| 1 | static_physical | 512–512 | 0.0784 | 42.665 | 17.264 |
| 2 | selective_cpu_bank | 1024–1024 | 0.7484 | 31.984 | 18.802 |
| 2 | resident_dynamic | 1024–1024 | 0.1942 | 37.063 | 24.662 |
| 2 | full | 1024–1024 | 0.2610 | 28.036 | 24.858 |
| 2 | static_physical | 1024–1024 | 0.1849 | 37.241 | 18.792 |
| 3 | selective_cpu_bank | 1920–1920 | 1.0190 | 30.175 | 18.184 |
| 3 | resident_dynamic | 1920–1920 | 0.3320 | 35.841 | 25.216 |
| 3 | full | 1920–1920 | 0.4891 | 25.254 | 25.661 |
| 3 | static_physical | 1920–1920 | 0.3443 | 35.771 | 18.175 |

按需加载包含换层搬运与完整prefill；静态物理删层的布局准备不计入稳态请求时间。冷启动、100ms采样进程显存峰值与CPU权重银行内存见worker证据。采样峰值是下界；这里是独立请求，未测真实多轮对话或并发服务。

恢复微调暂未进行，先用本轮冻结权重结果重新判断需要；动态与静态方案应使用相同恢复预算。

## 证据

- training: `/home/liushenyu/test/NeuroBypass/runs/stage13_full_context_20261110_02_training`
- development: `/home/liushenyu/test/NeuroBypass/runs/stage13_full_context_20261110_02_development`
- horizontal: `/home/liushenyu/test/NeuroBypass/runs/stage13_full_context_20261110_02_horizontal`
- cost: `/home/liushenyu/test/NeuroBypass/runs/stage13_full_context_20261110_02_cost`
