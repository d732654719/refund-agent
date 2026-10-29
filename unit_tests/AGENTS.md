# 单元验证模块

## 职责与入口

本目录通过模型替身、隔离的内存订单和临时输出，验证程序契约。测试通过不表示真实模型或外部平台通过验收。

| 修改内容 | 优先检查 |
| --- | --- |
| 工具调度、回复与对话状态 | [test_llm_agent.py](test_llm_agent.py)、[test_deterministic_refund_agent.py](test_deterministic_refund_agent.py) |
| 状态目录、CLI、认证配置 | [test_agent_status_catalog.py](test_agent_status_catalog.py)、[test_command_line_interface.py](test_command_line_interface.py)、[test_model_runtime.py](test_model_runtime.py) |
| DeepEval 转换与多轮契约 | [test_evaluation_adapters.py](test_evaluation_adapters.py) |
| 报告数据、写入、pytest 生命周期 | [test_evaluation_report_session.py](test_evaluation_report_session.py)、[test_evaluation_report_writer.py](test_evaluation_report_writer.py)、[test_evaluation_report_hooks.py](test_evaluation_report_hooks.py) |

## 运行与约束

从项目根目录执行 `.venv\Scripts\python.exe -m pytest unit_tests -q`；定位问题时把 `unit_tests` 换成相关文件路径。不要从另一个 worktree 导入同名包。

- 回归测试应覆盖触发条件和可观察结果，例如已退款后遇到超时仍保留金额事实；不要只复述分支实现。
- 修改这里的测试时先查 [运行模块](../refund_agent/AGENTS.md) 或 [评测模块](../agent_evaluations/AGENTS.md) 的契约，不用放宽断言代替修复。
- 文档、格式等低影响改动只做相关静态检查；不得为测试方便调用真实模型、写入真实报告目录或污染共享样例状态。
