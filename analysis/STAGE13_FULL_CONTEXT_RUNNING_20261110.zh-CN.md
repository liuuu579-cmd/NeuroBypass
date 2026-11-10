# 完整提示词修复已通过小试，正式重训运行中

这是启动快照，不是正式重训结果。当前有效主目录：

`runs/stage13_full_context_20261110_02_training`

CPU回归、真实OPT四种长度候选核验、四arm训练小试已完成。评分/置换小试审计900行；部署小试审计32次请求，按需和全常驻的路径、首步logits、生成哈希一致。当前正式重训4张RTX6000 Ada、42核，先在新变长数据上重算静态校准与条件联合贪心标签，随后训练m=0/4/6/8、各三个种子、24epoch、K最多24。

正式训练后，队列依次自动执行开发质量/置换、完整横向质量、吞吐/显存、独立审计与报告发布。主结果尚未产生，不能用工程小试的质量值作论文结论。恢复微调未开始。

## 有效程序入口

- 共同观察/变长编码/候选评分：`tools/full_context_core_v1.py`
- 文档恢复与新变长数据：`tools/full_context_data_v1.py`
- 正式训练：`tools/train_full_context_four_gpu_v2.py`
- 开发与横向质量：`tools/evaluate_full_context_four_gpu_v1.py`
- 成本/按需部署：`tools/measure_full_context_cost_four_gpu_v2.py`
- 当前持续执行队列：`tools/run_full_context_pipeline_v3.py`
- 最终审计/报告：`tools/finalize_full_context_v1.py`

有效队列日志：`analysis/stage13_full_context_pipeline_v3_20261110.log`。训练进度：主目录的`stdout.log`。完成或失败最终收据：`analysis/stage13_full_context_20261110_02_completion.json`，此快照时尚未创建。

v1/v2旧控制器和失败记录按要求保留。v3接管已运行的正式训练，没有重启或中断GPU任务；将持续追加的协调事件改记`.log`，避免把自身活跃进度日志误当作不可变研究输入参与后续哈希核验。数据、源文件与旧结果未改，接管时正式训练快照的1400个受保护文件全部哈希一致。

修复、数据分布、影响范围和评测限制详见`STAGE13_FULL_CONTEXT_FIX_PROTOCOL_20261110.zh-CN.md`。旧128-token训练和短题/Wiki条件协议保留原含义；长提示词推理截断已在新入口取消。路由与基座共享完整可见提示词，目标续写不会进入预测器。
