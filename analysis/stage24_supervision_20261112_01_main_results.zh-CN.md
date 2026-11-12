# 阶段24：全文提示词、续写与混合监督对照

K20/40，原8个完整组合库不变，全文提示词embedding输入，回答内固定路径，OPT冻结。原800训练文档、原32条选检查点、其余68条单列复核；100条仍为反复使用的开发文档。

未来32-token监督对照复用Stage20 Huber三种子及常量对照，已逐条重放路由与KL。新增全文提示词lambda1和混合lambda0.5两组，各三种子及常量对照。提示词标签覆盖P-1个已观察下一词位置，不读取续写；完整组合实际联合KL保留协同。

F和P分别相对index0；A=训练均值F+P-训练均值P，T=(1-lambda)F+lambdaA。所有组共享F的训练均值先验与训练标准差、初始化、smoothL1、AdamW2e-4、batch16、4096token微批、24轮1200更新、调度器与检查点评测频率。仅训练目标改变；强静态仍是800训练选定的index1。

| 群体 | 监督 | 观察 | KL | 强静态KL | 静态减动态 | 配对95%区间 |
|---|---|---|---:|---:|---:|---|
| other68 | future32 | prompt | 1.42656 | 1.39958 | -0.02698 | [-0.04814, -0.00487] |
| other68 | future32 | constant | 1.39958 | 1.39958 | +0.00000 | [+0.00000, +0.00000] |
| other68 | mixed | prompt | 1.43416 | 1.39958 | -0.03457 | [-0.05747, -0.01268] |
| other68 | mixed | constant | 1.39958 | 1.39958 | +0.00000 | [+0.00000, +0.00000] |
| other68 | prompt_full | prompt | 1.44219 | 1.39958 | -0.04261 | [-0.06579, -0.01938] |
| other68 | prompt_full | constant | 1.39958 | 1.39958 | +0.00000 | [+0.00000, +0.00000] |
| all_validation100 | future32 | prompt | 1.35923 | 1.34864 | -0.01059 | [-0.03186, +0.01388] |
| all_validation100 | future32 | constant | 1.34864 | 1.34864 | +0.00000 | [+0.00000, +0.00000] |
| all_validation100 | mixed | prompt | 1.36377 | 1.34864 | -0.01514 | [-0.03598, +0.00931] |
| all_validation100 | mixed | constant | 1.34864 | 1.34864 | +0.00000 | [+0.00000, +0.00000] |
| all_validation100 | prompt_full | prompt | 1.36875 | 1.34864 | -0.02011 | [-0.04052, +0.00501] |
| all_validation100 | prompt_full | constant | 1.34864 | 1.34864 | +0.00000 | [+0.00000, +0.00000] |

全部NLL、逐种子、选择集、原续写监督对照的配对比较见汇总。区间按同文档先平均3种子再bootstrap，未校正多重比较，不含完整种子不确定性。没有按68条选择lambda、种子或门槛，没有新增校准/集成。

4张GPU/42CPU；worker各10核、控制器2核。训练前验证新旧全文提示词标签重放、embedding长度及值一致；实际模型重放每条开发路径KL。旧文件与输入检查点验哈希保全。源码、协议、审计与汇总上传，权重、token、embedding和逐文档标签留本地。

[协议](../runs/stage24_supervision_20261112_01_main/protocol.json) · [汇总](../runs/stage24_supervision_20261112_01_main/summary.json) · [验收](../runs/stage24_supervision_20261112_01_main/verification.json)
