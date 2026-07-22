from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

from jsonschema import Draft202012Validator

from formtsr_exp.adapters import (
    ExternalVLLMServerAdapter,
    LocalVLLMServerAdapter,
    _formtsr_vlm_response_schema,
    make_adapter,
)
from formtsr_exp.hierarchical_metrics_report import load_samples
from formtsr_exp.prompt import HIERARCHICAL_PROMPT, build_model_prompt


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "outputs/dataset_splits/template_stratified_seed42/test_index.jsonl"
SCHEMA_KEYS = (
    "regions",
    "widgets",
    "local_grids",
    "cells",
    "line_item_groups",
    "relations",
    "answer",
)


def _load_builder() -> ModuleType:
    path = ROOT / "peft/build_formtsr_sft_dataset.py"
    spec = importlib.util.spec_from_file_location("formtsr_sft_dataset_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import dataset builder from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _real_index_row(sample_id: str) -> dict[str, Any]:
    with INDEX.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sample_id"] == sample_id:
                result = dict(row)
                for key in ("image_path", "label_path"):
                    result[key] = str(ROOT / result[key])
                return result
    raise AssertionError(f"missing fixture sample {sample_id}")


class HierarchicalSftTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            INDEX,
            ROOT / "new-dataset-json",
            ROOT / "newdataset-layout",
        )
        if any(not path.exists() for path in required):
            raise unittest.SkipTest(
                "requires the private test index and full annotation fixtures"
            )
        cls.builder = _load_builder()
        row = _real_index_row("ja_6__01")
        with TemporaryDirectory() as directory:
            index_path = Path(directory) / "one_sample.jsonl"
            index_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = load_samples(
                index_path,
                ROOT / "new-dataset-json",
                ROOT / "newdataset-layout",
            )
        cls.sample = samples[0]
        cls.target = cls.builder._hierarchical_target(
            cls.sample,
            metadata_root=str(ROOT / "new-dataset-json"),
            max_regions=80,
            max_widgets=80,
            max_grids=10,
            max_cells=160,
            max_ligs=20,
            max_relations=220,
            max_text_length=48,
        )

    def test_target_matches_response_schema(self) -> None:
        self.assertEqual(tuple(self.target), SCHEMA_KEYS)
        errors = sorted(
            Draft202012Validator(
                _formtsr_vlm_response_schema(hierarchical=True)
            ).iter_errors(self.target),
            key=lambda error: list(error.absolute_path),
        )
        self.assertEqual([], errors, "\n".join(error.message for error in errors))

    def test_hierarchical_schema_does_not_change_legacy_limits(self) -> None:
        legacy = _formtsr_vlm_response_schema()
        hierarchical = _formtsr_vlm_response_schema(hierarchical=True)
        expected_limits = {
            "regions": (60, 80),
            "widgets": (40, 80),
            "cells": (100, 160),
            "relations": (60, 220),
        }
        for key, (legacy_limit, hierarchical_limit) in expected_limits.items():
            self.assertEqual(legacy_limit, legacy["properties"][key]["maxItems"])
            self.assertEqual(
                hierarchical_limit,
                hierarchical["properties"][key]["maxItems"],
            )
        legacy_widget = legacy["properties"]["widgets"]["items"]
        hierarchical_widget = hierarchical["properties"]["widgets"]["items"]
        self.assertNotIn("state", legacy_widget["properties"])
        self.assertIn("state", hierarchical_widget["properties"])

    def test_zero_relation_budget_emits_no_relation(self) -> None:
        target = self.builder._hierarchical_target(
            self.sample,
            metadata_root=str(ROOT / "new-dataset-json"),
            max_regions=80,
            max_widgets=80,
            max_grids=10,
            max_cells=160,
            max_ligs=20,
            max_relations=0,
            max_text_length=48,
        )
        self.assertEqual([], target["relations"])

    def test_ids_and_relation_endpoints_are_closed(self) -> None:
        ids: list[str] = []
        region_ids = {item["id"] for item in self.target["regions"]}
        for key in ("regions", "widgets", "cells", "line_item_groups"):
            ids.extend(item["id"] for item in self.target[key])
        for grid in self.target["local_grids"]:
            ids.append(grid["id"])
            self.assertIn(grid["region_id"], region_ids)
            ids.extend(cell["id"] for cell in grid["cells"])

        self.assertEqual(len(ids), len(set(ids)))
        registry = set(ids)
        canonical_types = {"key-value", "parent-child", "field-widget", "key-to-cell"}
        self.assertTrue(self.target["relations"])
        for relation in self.target["relations"]:
            self.assertEqual(set(relation), {"u", "r", "v"})
            self.assertIn(relation["u"], registry)
            self.assertIn(relation["v"], registry)
            self.assertIn(relation["r"], canonical_types)

    def test_widget_groups_preserve_type_size_and_state(self) -> None:
        widgets = self.target["widgets"]
        self.assertTrue(widgets)
        allowed_states = {"selected", "unselected", "unknown"}
        actual: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for widget in widgets:
            self.assertIn(widget["state"], allowed_states)
            self.assertTrue(widget["group_id"])
            self.assertTrue(widget["group_type"])
            actual[widget["group_id"]].append(widget)

        expected_groups = Counter(
            (group.group_type, tuple(sorted(member.state for member in group.members)))
            for group in self.sample.widget_groups
        )
        actual_groups = Counter(
            (
                members[0]["group_type"],
                tuple(sorted(member["state"] for member in members)),
            )
            for members in actual.values()
        )
        self.assertEqual(expected_groups, actual_groups)

    def test_grids_keep_complete_nested_cell_topology(self) -> None:
        grids = self.target["local_grids"]
        self.assertTrue(grids)
        self.assertEqual([], self.target["cells"])
        self.assertEqual(len(self.sample.template.grids), len(grids))
        self.assertEqual(
            sum(len(grid.cells) for grid in self.sample.template.grids),
            sum(len(grid["cells"]) for grid in grids),
        )
        for source, emitted in zip(self.sample.template.grids, grids, strict=True):
            expected = [
                (cell.row, cell.col, cell.rowspan, cell.colspan)
                for cell in source.cells
            ]
            actual = [
                (
                    cell["row"],
                    cell["col"],
                    cell.get("rowspan", 1),
                    cell.get("colspan", 1),
                )
                for cell in emitted["cells"]
            ]
            self.assertEqual(expected, actual)

    def test_hierarchical_prompt_variant_uses_full_contract(self) -> None:
        prompt = build_model_prompt({"prompt_variant": "hierarchical_full"}, "fallback")
        self.assertEqual(HIERARCHICAL_PROMPT, prompt)
        self.assertIn('"group_id":"wg1"', prompt)
        self.assertIn('"state":"selected|unselected|unknown|filled|blank"', prompt)
        self.assertIn('{"u":"r1","r":"key-value|parent-child|field-widget|key-to-cell","v":"r2"}', prompt)

    def test_vllm_lora_request_name_can_differ_from_base_name(self) -> None:
        adapter = LocalVLLMServerAdapter(
            {
                "name": "test",
                "model": "display-name",
                "served_model": "hierarchical-lora",
                "server_served_model": "base-model",
                "env": {"VLLM_MODEL_PATH": "/models/base"},
            }
        )
        self.assertEqual("hierarchical-lora", adapter._served_model())
        self.assertEqual("base-model", adapter._server_served_model())
        command = adapter._server_command()
        self.assertIn("--served-model-name base-model", command)

    def test_external_vllm_adapter_reuses_running_endpoint(self) -> None:
        adapter = make_adapter(
            {
                "provider": "external_vllm_server_vlm",
                "name": "checkpoint-100",
                "served_model": "checkpoint-100",
                "base_url": "http://127.0.0.1:8123",
            }
        )
        self.assertIsInstance(adapter, ExternalVLLMServerAdapter)
        self.assertEqual("checkpoint-100", adapter._served_model())
        self.assertEqual("http://127.0.0.1:8123", adapter._base_url())


if __name__ == "__main__":
    unittest.main()
