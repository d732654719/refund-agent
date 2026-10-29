from __future__ import annotations

import argparse
import json
import sys

from .deterministic_refund_agent import RefundAgent
from .model_runtime import DEFAULT_TARGET_MODEL
from .llm_agent import LLMRefundAgent
from .sample_order_repository import create_sample_repository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="最小电商退款 Agent")
    parser.add_argument("message", nargs="?", help="单次用户消息；省略则进入交互模式")
    parser.add_argument("--user-id", default="user-001", help="当前登录用户")
    parser.add_argument(
        "--mode",
        choices=("llm", "deterministic"),
        default="llm",
        help="运行模式，默认使用真实 LLM",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TARGET_MODEL,
        help="LLM 模型名称",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="打印结构化工具执行轨迹",
    )
    return parser


def print_result(result, show_trace: bool) -> None:
    output = [
        "",
        "================ Agent 回答 ================",
        result.response,
    ]
    if show_trace:
        output.extend(["", "================ 执行 Trace ================"])
        trace_output = {
            "status": result.status,
            "conversation_turn": result.conversation_turn,
            "model_turns": result.model_turns,
        }
        if result.events:
            display_events = []
            for event in result.events:
                display_event = dict(event)
                if (
                    display_event.get("event") == "model_response"
                    and display_event.get("content") == result.response
                    and not display_event.get("tool_calls")
                ):
                    display_event["content"] = "[与上方 Agent 回答相同]"
                display_events.append(display_event)
            trace_output["events"] = display_events
        else:
            trace_output["steps"] = [step.to_dict() for step in result.trace]
        output.append(json.dumps(trace_output, ensure_ascii=False, indent=2))
    output.append("============================================")

    # Emit a complete response atomically so terminal capture cannot interleave
    # fragments from the answer, trace, and the following input prompt.
    sys.stdout.write("\n".join(output) + "\n")
    sys.stdout.flush()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    repository = create_sample_repository()
    agent = (
        LLMRefundAgent.from_environment(repository, model=args.model)
        if args.mode == "llm"
        else RefundAgent(repository)
    )

    if args.message:
        print_result(agent.run(args.message, args.user_id), args.show_trace)
        return

    print(
        "退款 Agent 已启动。输入 exit 退出，输入 /reset 清空对话上下文。"
        "示例：帮我退款订单 A1001"
    )
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"exit", "quit", "退出"}:
            break
        if message.lower() == "/reset":
            if hasattr(agent, "reset_conversation"):
                agent.reset_conversation()
                print("对话上下文已清空；订单和退款状态保持不变。")
            else:
                print("确定性模式没有 LLM 对话上下文。")
            continue
        print_result(agent.run(message, args.user_id), args.show_trace)


if __name__ == "__main__":
    main()
