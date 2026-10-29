"""CLI entry point for a real-model Langfuse Golden Experiment."""

from __future__ import annotations

import argparse
import json

from .langfuse_golden_experiment import (
    DEFAULT_DATASET_NAME,
    cloud_client_from_environment,
    run_golden_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行退款 Agent Langfuse Golden Experiment")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--limit", type=int, help="仅运行前 N 条，用于受控冒烟验证。"
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="仅运行指定 Golden case；可重复传入此参数。",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正整数。")

    result = run_golden_experiment(
        client=cloud_client_from_environment(),
        dataset_name=args.dataset_name,
        run_name=args.run_name,
        limit=args.limit,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    print(
        json.dumps(
            {
                "experiment_name": result.name,
                "run_name": result.run_name,
                "dataset_run_id": result.dataset_run_id,
                "dataset_run_url": result.dataset_run_url,
                "item_count": len(result.item_results),
                "items": [
                    {
                        "case_id": (
                            item.item.metadata.get("case_id")
                            if hasattr(item.item, "metadata")
                            else item.item.get("metadata", {}).get("case_id")
                        ),
                        "trace_id": item.trace_id,
                        "session_id": (
                            item.output.get("session_id")
                            if isinstance(item.output, dict)
                            else None
                        ),
                        "execution_error": (
                            item.output.get("execution_error")
                            if isinstance(item.output, dict)
                            else "unknown output type"
                        ),
                    }
                    for item in result.item_results
                ],
                "run_evaluations": [
                    {"name": item.name, "value": item.value, "comment": item.comment}
                    for item in result.run_evaluations
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
