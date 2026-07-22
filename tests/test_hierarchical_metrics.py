from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from formtsr_exp.hierarchical_metrics import (
    GridCell,
    LocalGrid,
    Relation,
    Widget,
    WidgetGroup,
    grits_top,
    lg_grits_top,
    match_widgets_for_relations,
    maximum_weight_matching,
    relation_f1,
    widget_group_f1,
)
from formtsr_exp.hierarchical_metrics_report import (
    _add_endpoint_mapping,
    EvaluationSample,
    TemplateStructure,
    evaluate_run,
    load_raw_template_structure,
    load_template_structure,
    parse_raw_annotation,
    parse_prediction,
    recover_grid_parents,
    reconcile_with_r_f1_structure,
    resolve_widget_groups,
)
from formtsr_exp.io_utils import read_json
from formtsr_exp.metrics import NA
from formtsr_exp.page_em_report import RunSpec


ROOT = Path(__file__).resolve().parents[1]


def cell(cell_id: str, row: int, col: int, rowspan: int = 1, colspan: int = 1) -> GridCell:
    return GridCell(cell_id, row, col, rowspan, colspan)


def grid(
    grid_id: str,
    parent_id: str,
    cells: tuple[GridCell, ...],
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> LocalGrid:
    return LocalGrid(grid_id, parent_id, cells, bbox)


class MatchingTests(unittest.TestCase):
    def test_weight_matching_is_global_and_optional(self) -> None:
        weights = np.asarray([[1.0, 0.6], [0.6, 0.0]])
        pairs = maximum_weight_matching(weights, np.ones_like(weights, dtype=bool))
        self.assertEqual(pairs, [(0, 1), (1, 0)])

        # Optional matching must retain the single high edge rather than force
        # two lower edges merely to maximize cardinality.
        weights = np.asarray([[1.0, 0.4], [0.4, 0.0]])
        pairs = maximum_weight_matching(weights, np.ones_like(weights, dtype=bool))
        self.assertEqual(pairs, [(0, 0)])

    def test_cardinality_first_uses_weight_as_tie_break(self) -> None:
        weights = np.asarray([[0.9, 0.8], [0.7, 0.0]])
        pairs = maximum_weight_matching(
            weights, np.ones_like(weights, dtype=bool), cardinality_first=True
        )
        self.assertEqual(pairs, [(0, 1), (1, 0)])


class GritsTopTests(unittest.TestCase):
    def test_identical_grid_scores_one_and_maps_cells(self) -> None:
        pred = grid("p", "pr", (cell("pc1", 0, 0), cell("pc2", 0, 1)))
        gt = grid("g", "gr", (cell("gc1", 0, 0), cell("gc2", 0, 1)))
        result = grits_top(pred, gt)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.f1, 1.0)
        self.assertEqual(result.cell_mapping, {"pc1": "gc1", "pc2": "gc2"})

    def test_missing_row_has_reference_fscore(self) -> None:
        gt = grid(
            "g",
            "gr",
            (
                cell("g00", 0, 0),
                cell("g01", 0, 1),
                cell("g10", 1, 0),
                cell("g11", 1, 1),
            ),
        )
        pred = grid("p", "pr", (cell("p00", 0, 0), cell("p01", 0, 1)))
        result = grits_top(pred, gt)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.f1, 2.0 / 3.0)

    def test_merged_cell_changes_topology(self) -> None:
        gt = grid("g", "gr", (cell("merged", 0, 0, colspan=2),))
        pred = grid("p", "pr", (cell("left", 0, 0), cell("right", 0, 1)))
        result = grits_top(pred, gt)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(result.f1, 1.0)
        self.assertGreater(result.f1, 0.0)

    def test_overlap_and_hole_are_invalid_core_inputs(self) -> None:
        overlap = grid(
            "p",
            "pr",
            (cell("span", 0, 0, colspan=2), cell("duplicate", 0, 1)),
        )
        valid = grid("g", "gr", (cell("g0", 0, 0), cell("g1", 0, 1)))
        self.assertIsNone(grits_top(overlap, valid))
        hole = grid("h", "pr", (cell("h0", 0, 0), cell("h2", 0, 2)))
        self.assertIsNone(grits_top(hole, valid))


class LocalGridTests(unittest.TestCase):
    def test_parent_match_is_mandatory(self) -> None:
        pred = grid("p", "pred-parent", (cell("pc", 0, 0),))
        gt = grid("g", "gt-parent", (cell("gc", 0, 0),))
        score = lg_grits_top((pred,), (gt,), {})
        self.assertEqual(score.score, 0.0)
        self.assertEqual(score.matches, ())

    def test_singleton_fallback_allows_low_grid_iou(self) -> None:
        pred = grid(
            "p",
            "pred-parent",
            (cell("pc", 0, 0),),
            (0.0, 0.0, 0.1, 0.1),
        )
        gt = grid(
            "g",
            "gt-parent",
            (cell("gc", 0, 0),),
            (0.9, 0.9, 1.0, 1.0),
        )
        score = lg_grits_top((pred,), (gt,), {"pred-parent": "gt-parent"})
        self.assertEqual(score.score, 1.0)

    def test_fallback_is_disabled_with_multiple_grids(self) -> None:
        pred_one = grid(
            "p1", "pred-parent", (cell("pc1", 0, 0),), (0.0, 0.0, 0.2, 0.2)
        )
        pred_two = grid(
            "p2", "pred-parent", (cell("pc2", 0, 0),), (0.4, 0.4, 0.6, 0.6)
        )
        gt = grid(
            "g", "gt-parent", (cell("gc", 0, 0),), (0.8, 0.8, 1.0, 1.0)
        )
        score = lg_grits_top(
            (pred_one, pred_two), (gt,), {"pred-parent": "gt-parent"}
        )
        self.assertEqual(score.score, 0.0)

    def test_extra_prediction_is_in_page_denominator(self) -> None:
        pred_one = grid("p1", "pred-parent", (cell("pc1", 0, 0),))
        pred_two = grid(
            "p2", "pred-parent", (cell("pc2", 0, 0),), (0.0, 0.0, 0.2, 0.2)
        )
        gt = grid("g", "gt-parent", (cell("gc", 0, 0),))
        score = lg_grits_top(
            (pred_one, pred_two), (gt,), {"pred-parent": "gt-parent"}
        )
        self.assertAlmostEqual(float(score.score), 2.0 / 3.0)

    def test_both_empty_is_na(self) -> None:
        self.assertEqual(lg_grits_top((), (), {}).score, NA)


class WidgetGroupTests(unittest.TestCase):
    def test_two_layer_score_with_one_correct_member(self) -> None:
        pred = WidgetGroup(
            "pg",
            "checkbox",
            (
                Widget("p1", (0.0, 0.0, 0.2, 0.2), "checkbox", "selected"),
                Widget("p2", (0.5, 0.5, 0.7, 0.7), "checkbox", "selected"),
            ),
        )
        gt = WidgetGroup(
            "gg",
            "checkbox",
            (
                Widget("g1", (0.0, 0.0, 0.2, 0.2), "checkbox", "selected"),
                Widget("g2", (0.8, 0.8, 1.0, 1.0), "checkbox", "selected"),
            ),
        )
        score = widget_group_f1((pred,), (gt,))
        self.assertAlmostEqual(float(score.score), 0.5)
        self.assertEqual(score.member_matches, 1)

    def test_group_type_state_and_iou_are_strict(self) -> None:
        gt = WidgetGroup(
            "gg",
            "checkbox",
            (Widget("g", (0.0, 0.0, 0.5, 0.5), "checkbox", "unknown"),),
        )
        matching_unknown = WidgetGroup(
            "pg",
            "checkbox",
            (Widget("p", (0.0, 0.0, 0.5, 0.5), "checkbox", "unknown"),),
        )
        selected = WidgetGroup(
            "pg",
            "checkbox",
            (Widget("p", (0.0, 0.0, 0.5, 0.5), "checkbox", "selected"),),
        )
        wrong_group = WidgetGroup(
            "pg",
            "checkbox_multi",
            (Widget("p", (0.0, 0.0, 0.5, 0.5), "checkbox", "unknown"),),
        )
        self.assertEqual(widget_group_f1((matching_unknown,), (gt,)).score, 1.0)
        self.assertEqual(widget_group_f1((selected,), (gt,)).score, 0.0)
        self.assertEqual(widget_group_f1((wrong_group,), (gt,)).score, 0.0)

    def test_relation_widget_mapping_ignores_state(self) -> None:
        pred = Widget("p", (0.0, 0.0, 0.5, 0.5), "checkbox", "selected")
        gt = Widget("g", (0.0, 0.0, 0.5, 0.5), "checkbox", "unselected")
        self.assertEqual(match_widgets_for_relations((pred,), (gt,)), {"p": "g"})


class RelationTests(unittest.TestCase):
    def test_direction_type_and_fixed_endpoints(self) -> None:
        gt = (Relation("gu", "key-value", "gv"),)
        mapping = {"pu": "gu", "pv": "gv"}
        correct = relation_f1((Relation("pu", "key-value", "pv"),), gt, mapping)
        reversed_edge = relation_f1(
            (Relation("pv", "key-value", "pu"),), gt, mapping
        )
        wrong_type = relation_f1((Relation("pu", "parent-child", "pv"),), gt, mapping)
        self.assertEqual(correct.counts.f1, 1.0)
        self.assertEqual(reversed_edge.counts.f1, 0.0)
        self.assertEqual(wrong_type.counts.f1, 0.0)

    def test_relation_type_and_endpoint_ids_are_exact(self) -> None:
        gt = (Relation("gu", "key-value", "gv"),)
        mapping = {"pu": "gu", "pv": "gv"}

        wrong_case = relation_f1(
            (Relation("pu", "Key-Value", "pv"),), gt, mapping
        )
        padded_endpoint = relation_f1(
            (Relation(" pu", "key-value", "pv"),), gt, mapping
        )

        self.assertEqual(wrong_case.counts.f1, 0.0)
        self.assertEqual(padded_endpoint.counts.f1, 0.0)
        self.assertEqual(padded_endpoint.matched_endpoint_counts.pred, 0)

    def test_symmetric_type_is_canonicalized(self) -> None:
        score = relation_f1(
            (Relation("pb", "adjacent", "pa"),),
            (Relation("ga", "adjacent", "gb"),),
            {"pa": "ga", "pb": "gb"},
            symmetric_types={"adjacent"},
        )
        self.assertEqual(score.counts.f1, 1.0)

    def test_duplicate_edges_are_set_semantics(self) -> None:
        edge = Relation("pu", "key-value", "pv")
        score = relation_f1(
            (edge, edge),
            (Relation("gu", "key-value", "gv"),),
            {"pu": "gu", "pv": "gv"},
        )
        self.assertEqual(score.counts.pred, 1)
        self.assertEqual(score.counts.f1, 1.0)

    def test_matched_endpoint_metric_has_conditional_denominators(self) -> None:
        score = relation_f1(
            (
                Relation("pu", "key-value", "pv"),
                Relation("unmatched", "key-value", "pv"),
            ),
            (
                Relation("gu", "key-value", "gv"),
                Relation("gx", "key-value", "gv"),
            ),
            {"pu": "gu", "pv": "gv"},
        )
        self.assertAlmostEqual(float(score.counts.f1), 0.5)
        self.assertEqual(score.matched_endpoint_counts.f1, 1.0)


class MetadataRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            ROOT / "new-dataset-json",
            ROOT / "newdataset-layout",
            ROOT / "FormTSR" / "datasets",
        )
        if any(not path.exists() for path in required):
            raise unittest.SkipTest("requires full dataset annotation fixtures")

    def test_raw_metadata_preserves_groups_and_instance_state(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        answer = read_json(
            ROOT / "FormTSR" / "datasets" / "Arabic-1" / "01" / "answer.json"
        )
        groups, unknown, _sources = resolve_widget_groups(template, answer)
        self.assertEqual(len(groups), 6)
        self.assertEqual(sum(len(group.members) for group in groups), 23)
        self.assertEqual(unknown, 0)
        states = [widget.state for group in groups for widget in group.members]
        self.assertEqual(states.count("selected"), 6)

    def test_raw_merged_cells_form_valid_grits_grids(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "zn_4.json", "zn_4"
        )
        self.assertTrue(template.grids)
        self.assertTrue(any(cell.rowspan > 1 for grid in template.grids for cell in grid.cells))
        for local_grid in template.grids:
            result = grits_top(local_grid, local_grid)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertAlmostEqual(result.f1, 1.0)

    def test_incomplete_cell_excludes_its_entire_parent_grid(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "en_4.json", "en_4"
        )
        self.assertEqual(template.grids, ())
        self.assertEqual(template.audit["incomplete_cells"], 3)
        self.assertEqual(template.audit["invalid_incomplete_grids_excluded"], 2)

    def test_raw_grids_are_reparented_to_exact_r_f1_universe(self) -> None:
        raw = load_raw_template_structure(
            ROOT / "new-dataset-json" / "en_6.json", "en_6"
        )
        r_f1 = load_template_structure(
            ROOT / "newdataset-layout" / "en_6.json", "en_6"
        )
        reconciled = reconcile_with_r_f1_structure(raw, r_f1)

        self.assertEqual(reconciled.regions, r_f1.regions)
        self.assertTrue(reconciled.grids)
        region_ids = {region.id for region in reconciled.regions}
        self.assertTrue(
            all(grid.parent_region_id in region_ids for grid in reconciled.grids)
        )
        self.assertEqual(
            reconciled.audit["grid_parents_without_r_f1_correspondence"], 0
        )

    def test_nested_selected_option_uses_member_path_presence(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        answer = read_json(
            ROOT / "FormTSR" / "datasets" / "Arabic-1" / "100" / "answer.json"
        )
        groups, unknown, _sources = resolve_widget_groups(template, answer)
        states = {
            widget.id: widget.state for group in groups for widget in group.members
        }
        self.assertEqual(states["-GLHGoXMiF"], "selected")
        self.assertEqual(unknown, 0)

    def test_repeated_widget_paths_align_to_answer_row_indices(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "en_6.json", "en_6"
        )
        answer = read_json(
            ROOT / "FormTSR" / "datasets" / "en_6" / "97" / "answer.json"
        )
        groups, unknown, _sources = resolve_widget_groups(template, answer)
        states = {
            widget.id: widget.state for group in groups for widget in group.members
        }
        specs_by_label: dict[str, list[object]] = {}
        for spec in template.widget_specs:
            if spec.option_labels and spec.option_labels[0] in {
                "Academic",
                "Arts",
                "Sports",
                "Service",
                "Others",
            }:
                specs_by_label.setdefault(spec.option_labels[0], []).append(spec)
        state_rows = {
            label: [
                states[spec.id]
                for spec in sorted(specs, key=lambda item: (item.bbox[1], item.bbox[0]))
            ]
            for label, specs in specs_by_label.items()
        }
        self.assertEqual(state_rows["Academic"].count("selected"), 1)
        self.assertEqual(state_rows["Arts"].count("selected"), 1)
        self.assertEqual(state_rows["Sports"].count("selected"), 0)
        self.assertEqual(state_rows["Service"].count("selected"), 2)
        self.assertEqual(state_rows["Others"].count("selected"), 1)
        self.assertEqual(unknown, 0)

    def test_ambiguous_field_schema_paths_are_not_mapped(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "en_13.json", "en_13"
        )
        self.assertNotIn(("beneficiary information", "name"), template.field_paths)
        self.assertGreater(template.audit["ambiguous_field_schema_paths"], 0)

    def test_widget_self_relation_does_not_split_explicit_group(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "ja_14.json", "ja_14"
        )
        target = next(spec for spec in template.widget_specs if spec.id == "5nbxGlCrd9")
        self.assertEqual(
            sum(spec.group_id == target.group_id for spec in template.widget_specs), 4
        )
        self.assertEqual(template.audit["self_widget_relations_ignored"], 1)

    def test_left_label_studio_relation_reverses_endpoints(self) -> None:
        rectangle = {
            "from_name": "bbox",
            "type": "rectanglelabels",
            "original_width": 100,
            "original_height": 100,
            "value": {
                "x": 0,
                "y": 0,
                "width": 10,
                "height": 10,
                "rectanglelabels": ["key"],
            },
        }
        raw = [
            {
                "annotations": [
                    {
                        "result": [
                            rectangle | {"id": "left"},
                            rectangle | {"id": "right"},
                            {
                                "type": "relation",
                                "from_id": "left",
                                "to_id": "right",
                                "direction": "left",
                            },
                        ]
                    }
                ]
            }
        ]
        _items, relations, _width, _height = parse_raw_annotation(raw)
        self.assertEqual(relations, [("right", "left", "left")])

    def test_unrecoverable_partial_scope_is_not_assumed_to_be_index_prefix(self) -> None:
        template = TemplateStructure(
            "template",
            1.0,
            1.0,
            (),
            (),
            (),
            (),
            (),
            {},
            {},
            {},
            {},
        )
        samples = [
            EvaluationSample(
                sample_id,
                "template",
                {},
                template,
                (),
                0,
                {"answer_presence": 0, "option_membership": 0, "unknown": 0},
            )
            for sample_id in ("sample-a", "sample-b")
        ]
        with TemporaryDirectory() as directory:
            pred_root = Path(directory)
            (pred_root / "test-run").mkdir()
            row, _types, _states = evaluate_run(
                RunSpec("test-run", "Test", "test", 1),
                samples,
                pred_root,
                {"runs": {"test-run": {"source_space": "none"}}},
                set(),
                None,
            )
        self.assertEqual(row["sample_scope"], "unrecoverable_partial")
        self.assertEqual(row["n_missing_prediction"], 1)
        self.assertEqual(row["LG-GriTS-Top"], NA)
        self.assertEqual(row["WG-F1"], NA)
        self.assertEqual(row["Rel-F1"], NA)


class PredictionMetadataRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            ROOT / "new-dataset-json",
            ROOT / "newdataset-layout",
        )
        if any(not path.exists() for path in required):
            raise unittest.SkipTest("requires full dataset annotation fixtures")

    def test_legacy_relation_fields_ids_and_types_are_canonicalized(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "regions": [
                    {
                        "id": "field-1",
                        "type": "field",
                        "bbox": [0, 0, 100, 100],
                    },
                    {
                        "id": "value-1",
                        "type": "value",
                        "bbox": [110, 0, 210, 100],
                    },
                ],
                "widgets": [
                    {
                        "id": "widget-1",
                        "type": "checkbox",
                        "bbox": [220, 0, 240, 20],
                        "selected": True,
                    }
                ],
                "relations": [
                    {
                        "from": "field-1",
                        "to": "value-1",
                        "type": "label-value",
                    },
                    {"parent": "field-1", "child": "widget-1"},
                ],
            },
            "normalized_1000",
            template,
        )
        self.assertEqual(
            prediction.relations,
            (
                Relation("field-1", "field-widget", "widget-1"),
                Relation("field-1", "key-value", "value-1"),
            ),
        )
        self.assertEqual(prediction.audit["adapter_relation_type_aliases"], 1)
        self.assertEqual(prediction.audit["adapter_relation_types_inferred"], 1)

    def test_malformed_relation_items_and_conflicting_aliases_are_rejected(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "relations": [
                    {"id": "not-an-edge", "text": "field"},
                    {
                        "source": "r1",
                        "from": "r2",
                        "target": "r3",
                        "type": "parent-child",
                    },
                ]
            },
            "normalized_1000",
            template,
        )
        self.assertEqual(prediction.relations, ())
        self.assertEqual(prediction.audit["adapter_relation_declared_items"], 2)
        self.assertEqual(prediction.audit["adapter_relation_rejected_items"], 2)
        self.assertEqual(prediction.audit["adapter_relation_alias_conflicts"], 1)

    def test_explicit_endpoint_namespace_resolves_but_ambiguous_bare_id_does_not(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "regions": [
                    {"id": "key", "type": "field", "bbox": [0, 0, 100, 100]},
                    {"id": "c1", "type": "field", "bbox": [0, 110, 100, 210]},
                ],
                "local_grids": [
                    {
                        "id": "g1",
                        "region_id": "c1",
                        "cells": [
                            {"id": "c1", "row": 0, "col": 0, "bbox": [0, 0, 10, 10]}
                        ],
                    }
                ],
                "relations": [
                    {"source": "key", "target": "cells.c1", "type": "key_to_cell"},
                    {"source": "key", "target": "c1", "type": "key-to-cell"},
                ],
            },
            "normalized_1000",
            template,
        )
        self.assertIn(
            Relation("key", "key-to-cell", "cells.c1"), prediction.relations
        )
        unresolved = [
            relation
            for relation in prediction.relations
            if relation.target.startswith("__legacy_ambiguous_target_")
        ]
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(prediction.audit["adapter_relation_endpoint_aliases"], 1)
        self.assertEqual(prediction.audit["adapter_relation_ambiguous_endpoints"], 1)
        endpoint_mapping: dict[str, str] = {}
        _add_endpoint_mapping(endpoint_mapping, "regions", {"key": "gt-key"})
        _add_endpoint_mapping(endpoint_mapping, "cells", {"c1": "gt-cell"})
        explicit = next(
            relation
            for relation in prediction.relations
            if relation.target == "cells.c1"
        )
        score = relation_f1(
            (explicit,),
            (Relation("gt-key", "key-to-cell", "gt-cell"),),
            endpoint_mapping,
        )
        self.assertEqual(score.counts.f1, 1.0)

    def test_grid_parser_enriches_exact_cells_and_rejects_flat_pollution(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "cells": [
                    {"id": "c1", "row_start": 0, "row_end": 0, "col_start": 0, "col_end": 1, "bbox": [0, 0, 100, 100]}
                ],
                "local_grids": [
                    {"id": "g1", "cells": [{"id": "c1", "text": "value"}]},
                    {"id": "flat-cell", "row": 0, "col": 0, "bbox": [0, 0, 10, 10]},
                ],
            },
            "normalized_1000",
            template,
        )
        self.assertEqual(len(prediction.grids), 1)
        self.assertEqual(prediction.grids[0].cells[0].colspan, 2)
        self.assertEqual(prediction.grids[0].bbox, (0.0, 0.0, 0.1, 0.1))
        self.assertEqual(prediction.audit["adapter_grid_cells_enriched"], 1)
        self.assertEqual(prediction.audit["adapter_grid_rejected_items"], 1)

    def test_explicit_split_grid_references_merge_complete_fragments(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "regions": [
                    {"id": "r1", "type": "table", "bbox": [0, 0, 500, 500]}
                ],
                "local_grids": [
                    {
                        "id": "g1",
                        "region_id": "r1",
                        "bbox": [0, 0, 100, 100],
                        "cells": [
                            {"id": "c1", "row": 0, "col": 0, "bbox": [0, 0, 100, 100]}
                        ],
                    },
                    {
                        "id": "g2",
                        "region_id": "g1",
                        "cells": [
                            {"id": "c2", "row": 1, "col": 0, "bbox": [0, 100, 100, 200]}
                        ],
                    },
                ],
                "relations": [
                    {"source": "local_grids.g2", "target": "c2", "type": "reading-order"}
                ],
            },
            "normalized_1000",
            template,
        )
        self.assertEqual(len(prediction.grids), 1)
        self.assertEqual(prediction.grids[0].id, "g1")
        self.assertEqual([cell.row for cell in prediction.grids[0].cells], [0, 1])
        self.assertEqual(prediction.grids[0].bbox, (0.0, 0.0, 0.1, 0.2))
        self.assertEqual(
            prediction.relations,
            (Relation("local_grids.g1", "reading-order", "c2"),),
        )
        self.assertEqual(prediction.audit["adapter_grid_fragments_merged"], 1)
        self.assertEqual(prediction.audit["adapter_relation_endpoint_aliases"], 1)

    def test_grid_parent_recovery_requires_unique_matched_containment(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "regions": [
                    {"id": "r1", "type": "section", "bbox": [0, 0, 500, 500]}
                ],
                "local_grids": [
                    {
                        "id": "g1",
                        "cells": [
                            {"id": "c1", "row": 0, "col": 0, "bbox": [100, 100, 200, 200]}
                        ],
                    }
                ],
            },
            "normalized_1000",
            template,
        )
        unchanged, recovered = recover_grid_parents(prediction, {})
        self.assertEqual(recovered, 0)
        self.assertEqual(unchanged.grids[0].parent_region_id, "")

        adapted, recovered = recover_grid_parents(prediction, {"r1": "gt-r1"})
        self.assertEqual(recovered, 1)
        self.assertEqual(adapted.grids[0].parent_region_id, "r1")
        self.assertEqual(adapted.audit["adapter_grid_parent_inferred"], 1)

    def test_prediction_widgets_without_groups_become_singletons(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "widgets": [
                    {
                        "id": "w1",
                        "type": "checkbox",
                        "bbox": [0, 0, 100, 100],
                        "selected": True,
                    },
                    {
                        "id": "w2",
                        "type": "checkbox",
                        "bbox": [200, 0, 300, 100],
                        "selected": False,
                    },
                ]
            },
            "normalized_1000",
            template,
        )
        self.assertEqual(len(prediction.widget_groups), 2)
        self.assertTrue(all(len(group.members) == 1 for group in prediction.widget_groups))

    def test_lig_endpoint_candidates_match_corrected_lig_sources(self) -> None:
        template = load_raw_template_structure(
            ROOT / "new-dataset-json" / "Arabic-1.json", "Arabic-1"
        )
        prediction = parse_prediction(
            {
                "line_item_groups": [
                    {"id": "explicit", "bbox": [0, 0, 100, 100]}
                ],
                "regions": [
                    {
                        "id": "region-lig",
                        "type": "line_item_group",
                        "bbox": [200, 0, 300, 100],
                    }
                ],
                "local_grids": [
                    {
                        "id": "grid-lig",
                        "region_id": "parent",
                        "cells": [
                            {"row": 0, "col": 0, "bbox": [400, 0, 500, 100]}
                        ],
                    }
                ],
            },
            "normalized_1000",
            template,
        )
        self.assertEqual(
            [group.id for group in prediction.line_item_groups],
            ["explicit", "region-lig", "grid-lig"],
        )


if __name__ == "__main__":
    unittest.main()
