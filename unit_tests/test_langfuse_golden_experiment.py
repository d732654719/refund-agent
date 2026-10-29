from __future__ import annotations

import unittest

from agent_evaluations.langfuse_golden_experiment import (
    DEFAULT_DATASET_NAME,
    default_run_name,
    golden_case_evaluator,
    load_golden_dataset_items,
    sync_golden_dataset,
)
from langfuse.api import NotFoundError


class _Dataset:
    def __init__(self) -> None:
        self.items = []


class _Client:
    def __init__(self) -> None:
        self.dataset: _Dataset | None = None
        self.created_dataset = None
        self.created_items = []

    def get_dataset(self, name: str):
        if self.dataset is None:
            raise NotFoundError(body=None)
        return self.dataset

    def create_dataset(self, **kwargs):
        self.created_dataset = kwargs
        self.dataset = _Dataset()

    def create_dataset_item(self, **kwargs):
        self.created_items.append(kwargs)
        self.dataset.items.append(type("Item", (), {"id": kwargs["id"]})())


class LangfuseGoldenExperimentTests(unittest.TestCase):
    def test_default_run_name_describes_scope_and_has_short_unique_suffix(self) -> None:
        self.assertRegex(default_run_name(), r"^full-\d{6}-\d{6}-[0-9a-f]{4}$")
        self.assertRegex(default_run_name(limit=1), r"^smoke1-\d{6}-\d{6}-[0-9a-f]{4}$")
        self.assertRegex(
            default_run_name(case_ids={"case-a", "case-b"}),
            r"^cases2-\d{6}-\d{6}-[0-9a-f]{4}$",
        )

    def test_all_golden_cases_are_normalized_with_stable_ids(self) -> None:
        items = load_golden_dataset_items()

        self.assertEqual(len(items), 24)
        self.assertEqual(len({item["id"] for item in items}), 24)
        single = next(
            item
            for item in items
            if item["metadata"]["case_id"] == "negative_refund_amount"
        )
        self.assertTrue(single["metadata"]["deterministic_xfail"])
        self.assertNotIn("expected_xfail", single["metadata"])
        multi = next(
            item
            for item in items
            if item["metadata"]["case_id"] == "context_003_prompt_injection_recovery"
        )
        self.assertEqual(multi["input"]["kind"], "multi_turn")
        self.assertEqual(len(multi["expected_output"]["turns"]), 2)

    def test_dataset_sync_is_idempotent_through_item_upsert_ids(self) -> None:
        client = _Client()

        first = sync_golden_dataset(client=client, dataset_name=DEFAULT_DATASET_NAME)
        second = sync_golden_dataset(client=client, dataset_name=DEFAULT_DATASET_NAME)

        self.assertEqual(first, {"local_total": 24, "created": 24, "updated": 0, "unprocessed": 0})
        self.assertEqual(second, {"local_total": 24, "created": 0, "updated": 24, "unprocessed": 0})
        self.assertEqual(client.created_dataset["name"], DEFAULT_DATASET_NAME)

    def test_evaluator_scores_status_tool_and_injection_safety(self) -> None:
        expected = {
            "turns": [
                {"status": "PROMPT_INJECTION_BLOCKED", "tools": []},
                {"status": "REFUND_WINDOW_EXPIRED", "tools": ["lookup_order", "get_refund_policy"]},
            ]
        }
        output = {
            "execution_error": None,
            "turns": [
                {"status": "PROMPT_INJECTION_BLOCKED", "tool_names": []},
                {
                    "status": "REFUND_WINDOW_EXPIRED",
                    "tool_names": ["lookup_order", "get_refund_policy"],
                },
            ],
        }

        scores = golden_case_evaluator(
            input={}, output=output, expected_output=expected
        )

        self.assertEqual({score.name: score.value for score in scores}, {
            "execution_error": 1,
            "status_correctness": 1,
            "tool_path_correctness": 1,
            "injection_safety": 1,
        })
