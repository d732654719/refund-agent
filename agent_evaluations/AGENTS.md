# 评测模块

## 职责与接口

本目录把版本管理的用例交给 Agent，计算指标并汇总报告数据；不负责 HTML 模板实现。

| 入口 | 责任与契约 |
| --- | --- |
| [单轮用例](single_turn_golden_cases.json)、[多轮用例](multi_turn_conversation_cases.json) | 输入、用户身份、预期状态和工具路径的源数据 |
| [test_llm_single_turn_cases.py](test_llm_single_turn_cases.py)、[test_llm_multi_turn_cases.py](test_llm_multi_turn_cases.py) | 真实 Agent 与 Judge 的执行、异常捕获和逐案例记录 |
| [test_deterministic_golden_cases.py](test_deterministic_golden_cases.py) | 确定性基线评测；`no_llm_xfail_reason` 只描述该基线的已知缺口 |
| [deepeval_adapters.py](deepeval_adapters.py) | AgentResult/Trace 到 DeepEval 对象的转换、Judge 构造、多轮契约和超时归一化 |
| [code_contract_metrics.py](code_contract_metrics.py) | 状态、工具路径、数据完整性及执行错误等确定性指标 |
| [refund_workflow_efficiency_metric.py](refund_workflow_efficiency_metric.py) | 退款流程效率指标；业务计算留在本项目 |
| [conftest.py](conftest.py) | 仅在实际选中 `llm_eval` 测试且非 collect-only 时启动报告，结束时汇总 |
| [evaluation_report_session.py](evaluation_report_session.py) | `record_case_metrics()` 收集案例；`finalize_evaluation_session(exit_status=...)` 汇总运行并调用 writer |
| [evaluation_report_writer.py](evaluation_report_writer.py) | `write_evaluation_report(payload)` 调用共享包，返回 `ReportPaths`；另保留兼容文本日志 |

## 决策约束

- 单轮用例单独创建 Agent/仓库；一个多轮用例连续执行并保留每轮结果。多轮预期结构由 `validate_multi_turn_case_contract()` 检查。
- 确定性指标与 Judge 评分分别保留；不能用语言质量高分掩盖状态、权限或工具路径错误。不要为了通过评测改掉业务预期或移除失败用例。
- Target/Judge 超时、协议错误和用例数据不完整应进入报告；不要在记录之前直接失败，也不要把缺失 usage 当作零消耗。
- `no_llm_xfail_reason` 不豁免真实 LLM 评测。`expected_response_contains` 的固定词句检查用于确定性基线，真实 LLM 回复不套用同一固定措辞要求。
- JSON payload 在本模块组装；Schema、历史迁移和渲染由外部 `deepeval_report_template.render_report()` 负责。报告字段变化需同时检查共享包契约，不能加回本地模板副本。

## Langfuse 数据集与实验

[langfuse_golden_experiment.py](langfuse_golden_experiment.py) 从上述本地 JSON 构造稳定 item ID 和内容指纹；`sync_golden_dataset()` upsert 云端条目，当前不会删除远端多余条目。因此本地 JSON 是用例源头，但同步成功不代表云端集合与本地完全相同。

`run_golden_experiment()` 读取云端 Dataset，支持筛选用例，运行真实 Target。当前 evaluator 检查执行错误、状态、工具路径及注入安全，不等同于 DeepEval 的完整 Judge 指标，也尚未实现 Prompt 版本的 control/candidate 切换。

项目根目录的 `sync_langfuse_golden_dataset.cmd` 写入云端；`run_langfuse_golden_experiment.cmd` 执行实验；`run_langfuse_validation.cmd` 运行真实模型观测场景、不调用 Judge。CLI 参数以 [实验入口](run_langfuse_golden_experiment.py)、[同步入口](sync_langfuse_golden_dataset.py) 为准。使用前按当前任务确定范围，结果需区分本地运行与云端可查询证据。

## 验证与详细说明

从项目根目录执行 `.venv\Scripts\python.exe -m pytest unit_tests -q` 验证适配器和报告生命周期；`.\run_deterministic_golden_cases.cmd` 验证离线基线；`.\run_llm_evaluations.cmd` 才是真实 Target/Judge 评测。后两者验证对象不同，按任务选择。

用例扩充见 [用例矩阵](../refund_evaluation_case_matrix.md)，评测操作见 [评测指南](../llm_evaluation_guide.md)，测试入口见 [unit_tests](../unit_tests/AGENTS.md)。
