# 阶段22：相对强静态收益的门控诊断

仅用既有预测和完整组合联合KL离线计算，不更新权重。K20/40、原8组合、完整提示词路由不变；9个旧提示词路由器预测均值与分歧作为诊断。所有代价先减去同文档的强静态index1代价，以免将文档难度误当作换层收益。

逐组合校准斜率及残差只用800训练；收缩系数、残差/分歧惩罚和切换门槛的181个固定配置，仅原32条选择。68条只在选完之后复核。训练残差是样本内残差，分歧是启发式指标，均不提供置信保证；全部100条仍是已查看的开发文档。

| 群体 | 策略 | KL | 强静态KL | 静态减动态 | 配对95%区间 | 有益/有害切换 |
|---|---|---:|---:|---:|---|---|
| selection32 | relative_uncertainty_gate | 1.20144 | 1.24038 | +0.03894 | [-0.00535, +0.09148] | 7/3 |
| other68 | relative_uncertainty_gate | 1.40373 | 1.39958 | -0.00415 | [-0.02749, +0.02172] | 6/16 |
| all_validation100 | relative_uncertainty_gate | 1.33900 | 1.34864 | +0.00964 | [-0.01247, +0.03484] | 13/19 |
| selection32 | raw_ensemble | 1.21834 | 1.24038 | +0.02203 | [-0.03095, +0.08007] | 9/6 |
| other68 | raw_ensemble | 1.40814 | 1.39958 | -0.00856 | [-0.03469, +0.02009] | 10/21 |
| all_validation100 | raw_ensemble | 1.34741 | 1.34864 | +0.00123 | [-0.02253, +0.02822] | 19/27 |

原32条选定配置：`{'shrink': 0.5, 'penalty': 0.0, 'margin_units': 0.1, 'static': False, 'serial': 75, 'selection_kl': 1.201439879834652, 'selection_switches': 10}`。

复核68条仍未超过强静态；停止继续堆叠分数校准和门槛，下一步检查监督代价是否能由提示词预测。

常量观察对照始终回到强静态。旧Stage21原始集成KL完全复现。CPU限制2核、torch1线程、无GPU计算。旧文件验哈希保全；不上传逐文档路由、原始标签、embedding和权重。9模型集成有额外推理开销，本轮无速度或显存声明。

[协议](../runs/stage22_route1_uncertainty_diagnostic_20261112_02/protocol.json) · [汇总](../runs/stage22_route1_uncertainty_diagnostic_20261112_02/summary.json) · [配置](../runs/stage22_route1_uncertainty_diagnostic_20261112_02/policy_fits.json) · [校验](../runs/stage22_route1_uncertainty_diagnostic_20261112_02/verification.json)
