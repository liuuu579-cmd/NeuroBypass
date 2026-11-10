# 输入与路径匹配诊断（正式开发复核）

本轮冻结m=6路由器与checkpoint，比较输入自身路径、三次任务内置换路径和静态联合KL。路由只看输入前缀；候选答案、gold和未来目标token不进入路由。C4使用68条未参与checkpoint选择的开发文档，PIQA、HellaSwag、WinoGrande和WikiText各预先固定64条输入。这是开发诊断，不是新的独立测试。

C4与WikiText使用完整词表teacher KL/token；选择题报告acc_norm（WinoGrande为acc）。未来目标可见的最小KL仅作离线上界诊断，不能解释为可部署路由。

| 任务 | K | matched | shuffled均值 | static | 指标 |
| --- | ---: | ---: | ---: | ---: | --- |
| c4_development | 10 | 0.244606 | 0.252458 | 0.238661 | KL/token（越低越好） |
| c4_development | 12 | 0.388800 | 0.410295 | 0.382170 | KL/token（越低越好） |
| hellaswag | 10 | 0.713542 | 0.713542 | 0.687500 | 准确率（越高越好） |
| hellaswag | 12 | 0.682292 | 0.692708 | 0.671875 | 准确率（越高越好） |
| piqa | 10 | 0.729167 | 0.718750 | 0.734375 | 准确率（越高越好） |
| piqa | 12 | 0.718750 | 0.723958 | 0.703125 | 准确率（越高越好） |
| wikitext_conditional | 10 | 0.315519 | 0.309860 | 0.302979 | KL/token（越低越好） |
| wikitext_conditional | 12 | 0.501119 | 0.493323 | 0.468533 | KL/token（越低越好） |
| winogrande | 10 | 0.666667 | 0.687500 | 0.718750 | 准确率（越高越好） |
| winogrande | 12 | 0.656250 | 0.682292 | 0.656250 | 准确率（越高越好） |

matched相对shuffled若稳定更好，才支持输入与路径匹配；同时降低teacher KL比单看任务准确率更有说服力。当前结果只用于决定是否值得进入最终独立测试。路径多样性见`stage12_route_adaptivity_reanalysis_20261110.json`。

验收9720条记录、324个输入，1227个既有文件哈希不变。运行目录：`runs/stage12_route_fidelity_main_20261110_01`。
