# 观测适配模块

## 职责与接口

本目录封装 Langfuse SDK，由应用或实验代码向 Agent 注入；不决定退款资格和最终业务状态。公开导出见 [__init__.py](__init__.py)，实现见 [langfuse_adapter.py](langfuse_adapter.py)。

| 接口 | 用途 |
| --- | --- |
| `create_langfuse_telemetry(client=None)` | 构造可选的运行期适配器 |
| `LangfuseTelemetry.start_agent/start_generation/start_tool` | 返回观测上下文，关联 Agent、模型调用和工具 |
| `update(observation, **values)`、`flush()` | 更新与提交观测数据 |
| `create_langfuse_client()` | 显式云端操作的客户端；缺少凭据时抛错 |

## 决策约束

- 运行期缺少凭据时适配器可禁用；显式数据集/实验命令应暴露配置错误，不能把未连接云端的运行报告成云端成功。
- 摘要函数保留长度、摘要和必要结构，不直接上报完整 Prompt、订单内容或工具参数。增加字段时检查摘要路径；哈希摘要不应被描述成绝对匿名保证。
- `session_id` 的传播在适配器内完成；Trace/Experiment 关联参数从运行入口传入。不要把 SDK 全局状态变成业务依赖。
- 适配器尝试隔离观测故障，但不应吞掉已经进入 Agent 上下文后抛出的业务异常。修改上下文管理时验证进入、执行、退出各阶段。
- 装饰器和 Trace 数量不能证明每个执行分支均被覆盖，需用具体场景检查关联和缺失数据。

## 配置与验证

环境变量名与客户端构造以 `create_langfuse_client()` 为准；模型认证仍由 Agent 的 `model_runtime.py` 管理，不在这里另建密钥来源。

从项目根目录运行 `.venv\Scripts\python.exe -m pytest unit_tests/test_langfuse_observability.py -q`。真实模型观测入口是 `run_langfuse_validation.cmd`，云端实验入口见 [评测模块](../agent_evaluations/AGENTS.md)；本地替身测试与云端验证分别报告。
