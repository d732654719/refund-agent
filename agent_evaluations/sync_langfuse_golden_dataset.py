"""CLI entry point for syncing local refund Golden cases to Langfuse."""

from __future__ import annotations

import argparse
import json

from .langfuse_golden_experiment import (
    DEFAULT_DATASET_NAME,
    cloud_client_from_environment,
    sync_golden_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="同步退款 Agent Golden 到 Langfuse Dataset")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()
    summary = sync_golden_dataset(
        client=cloud_client_from_environment(), dataset_name=args.dataset_name
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
