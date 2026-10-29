# Refund Agent Langfuse 项目入口

## 这是什么项目

本分支在退款 Agent 和 DeepEval 上集成 Langfuse 观测、数据集和实验；`main` 是 DeepEval 基础版。
报告模板由 [独立报告仓库](https://github.com/d732654719/deepeval-report-template) 提供，通过 [requirements-report.txt](requirements-report.txt) 安装固定版本；不要在本项目复制模板、Schema 或渲染器。

先运行 `git status --short --branch` 核对分支与已有改动。公共修复按任务范围同步，Langfuse 专属能力留在本分支；保留无关改动，只暂存本次确认的文件。

## 按任务找模块

进入目录前阅读对应说明；接口和特殊约束留在模块旁边。

| 要改什么 | 入口 |
| --- | --- |
| 退款行为、模型协议、工具与状态 | [refund_agent/AGENTS.md](refund_agent/AGENTS.md) |
| 用例、指标、报告数据、云端实验 | [agent_evaluations/AGENTS.md](agent_evaluations/AGENTS.md) |
| Langfuse 上报、降级与 Trace 关联 | [agent_observability/AGENTS.md](agent_observability/AGENTS.md) |
| 单元验证与回归范围 | [unit_tests/AGENTS.md](unit_tests/AGENTS.md) |
| 文档同步与新会话复核 | [docs/documentation.md](docs/documentation.md) |

## 怎么跑

命令在项目根目录执行，先创建 `.venv` 虚拟环境；版本与依赖以 [pyproject.toml](pyproject.toml) 和 [requirements-report.txt](requirements-report.txt) 为准。

```powershell
# 仅在需要安装或修复依赖时执行
.venv\Scripts\python.exe -m pip install -e . -r requirements-report.txt
# 确定性运行，不调用真实模型
.venv\Scripts\python.exe -m refund_agent.command_line_interface "帮我退款订单 A1001" --mode deterministic --show-trace
```

真实模型运行方式和配置见 [README](README.md)。不输出或提交实际密钥、`auth.json`、真实报告和缓存。

## 怎么验证

```powershell
# 单元验证
.venv\Scripts\python.exe -m pytest unit_tests -q
# 离线 Golden 基线
.\run_deterministic_golden_cases.cmd
# 真实 Target/Judge 评测，按当前任务授权执行
.\run_llm_evaluations.cmd
# 文档链接检查与变更提醒，无外部模型调用
.venv\Scripts\python.exe scripts/check_docs.py
```

数据集同步、观测验收和云端实验的入口与限制见评测模块说明；不要把离线替身测试当作云端验收。
修改代码时同步检查模块说明；不需要改文档时在交付中写明原因。计划或历史结果不能替代本次验证证据。
