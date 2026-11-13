# 阶段28：首窗与多窗口联合KL监督配对小试

原训练集800条中，351条在保持全文提示词不变时还有128个同文档续写token且总长不超过2048；全部纳入，原100条开发文档不参与。固定K20/40及原8个完整剪层组合。每个组合的标签来自实际联合前向，不相加单层代价；回答内固定选层。

四折按正文及开头哈希分组，每折约228条拟合、35条内层选模、88条外折复核，三个种子579/580/581。四组为first32/mean128监督各配全文embedding及无输入constant。每组24轮、相同初始化参数、数据顺序、batch与优化器。first32取同一长序列首窗，mean128取四个32-token窗等权均值，消除短/长序列数值形状混杂。监督范围与提示词观察范围是两个不同概念：两组路由器都只读取完整提示词，不读取未来token。

训练拟合标签决定各组先验与缩放；所有检查点均用共同内层128-token KL选择，epoch0/1/4/...24，严格改善才保存。共同静态参考只由拟合子集的128-token KL均值选择；门槛也统一在内层128-token KL上选。训练接口的fit/selection标签按互斥ID构造，不混用。无输入组参数量与更新预算相同，不含内容或长度。mean3表示三个独立种子的质量均值，不是集成推理。

| 外折策略 | 全128 KL，越低越好 | 相对折内静态改善 | 探索性95%区间 |
|---|---:|---:|---|
| fold_static | 1.283067 | +0.000000 | [+0.000000, +0.000000] |
| historical_static1 | 1.283067 | +0.000000 | [+0.000000, +0.000000] |
| future_known_best128 | 1.222839 | +0.060229 | [+0.051601, +0.069697] |
| first32_embedding_raw_mean3 | 1.284436 | -0.001369 | [-0.003055, +0.000471] |
| first32_embedding_gate_mean3 | 1.284436 | -0.001369 | [-0.003055, +0.000471] |
| mean128_embedding_raw_mean3 | 1.283391 | -0.000324 | [-0.004415, +0.003645] |
| mean128_embedding_gate_mean3 | 1.283731 | -0.000664 | [-0.004310, +0.003196] |
| first32_constant_raw_mean3 | 1.283067 | +0.000000 | [+0.000000, +0.000000] |
| first32_constant_gate_mean3 | 1.283067 | +0.000000 | [+0.000000, +0.000000] |
| mean128_constant_raw_mean3 | 1.283067 | +0.000000 | [+0.000000, +0.000000] |
| mean128_constant_gate_mean3 | 1.283067 | +0.000000 | [+0.000000, +0.000000] |

| 多窗口embedding比较 | 参考 | 全128 KL改善 | 探索性95%区间 |
|---|---|---:|---|
| mean128_embedding_raw_mean3 | first32_embedding_raw_mean3 | +0.001045 | [-0.002177, +0.004202] |
| mean128_embedding_raw_mean3 | mean128_constant_raw_mean3 | -0.000324 | [-0.004415, +0.003645] |
| mean128_embedding_raw_mean3 | fold_static | -0.000324 | [-0.004415, +0.003645] |
| mean128_embedding_raw_mean3 | historical_static1 | -0.000324 | [-0.004415, +0.003645] |
| mean128_embedding_gate_mean3 | first32_embedding_gate_mean3 | +0.000705 | [-0.002086, +0.003380] |
| mean128_embedding_gate_mean3 | mean128_constant_gate_mean3 | -0.000664 | [-0.004310, +0.003196] |
| mean128_embedding_gate_mean3 | fold_static | -0.000664 | [-0.004310, +0.003196] |
| mean128_embedding_gate_mean3 | historical_static1 | -0.000664 | [-0.004310, +0.003196] |

正改善表示多窗口组更好。历史静态索引1和future_known_best128单列：前者已有候选筛选依赖，后者读取真实未来，仅为同库收益空间诊断。主要对比预先固定为mean128_embedding_gate与first32_embedding_gate，同时必须检查constant及折内静态。

这是反复查看过的旧训练文档上的开发诊断，候选库曾用其中部分数据筛选，不是独立测试。351条有续写长度选择偏倚；区间仅按外折文档重采样，忽略训练与折间依赖，未做多重比较校正。不能把更低训练标签损失直接解释为更好动态决策。没有新增恢复微调、任务准确率或加速测量。

GPU0—3 /42CPU，四worker各10核、控制器2核。所有旧源文件、报告和结果验哈希保全；新增参数、缓存、语料和逐文档记录仅本地保存，公开程序、协议、审计与聚合结果。

[协议](../runs/stage28_window_supervision_20261113_01_main/protocol.json) · [汇总](../runs/stage28_window_supervision_20261113_01_main/summary.json) · [独立审计](../runs/stage28_window_supervision_20261113_01_main/independent_audit.json)
