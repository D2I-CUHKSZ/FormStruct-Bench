import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmark_stats.run_representativeness_analysis import (
    Unit,
    allocate_stratified_counts,
    canonical_source_alias,
    extract_formstruct_graph,
    fit_gower_config,
    gower_distance_matrix,
    jensen_shannon_divergence,
    parse_srfund_relation_tree,
    sample_reference_sets,
)
from benchmark_stats.run_representativeness_analysis import DIRECT_GOWER_GROUPS


def make_unit(unit_id: str, language: str, features: dict[str, object]) -> Unit:
    payload = {
        "language_code": language,
        "script": "Latin",
        "writing_direction": "LTR",
        "max_hierarchy_depth": 2,
        "mean_branching_factor": 1.0,
        "num_root_children": 1,
        "num_entities": 2,
        "header_entity_ratio": 0.0,
        "question_entity_ratio": 0.5,
        "answer_entity_ratio": 0.5,
        "other_entity_ratio": 0.0,
        "num_item_tables": 0,
        "table_presence": False,
        "num_relation_links": 1,
        "relation_density": 0.5,
        "bbox_area_mean": 0.1,
        "bbox_width_mean": 0.2,
        "bbox_height_mean": 0.1,
        "bbox_center_x_mean": 0.5,
        "bbox_center_y_mean": 0.5,
        "bbox_center_x_std": 0.1,
        "bbox_center_y_std": 0.1,
        "spatial_layout_density": 0.2,
        "hierarchy_depth_bin": "shallow_0_2",
    }
    payload.update(features)
    return Unit(
        unit_id=unit_id,
        dataset="test",
        source_id=unit_id,
        language_code=language,
        split="test",
        image_path=Path("unused.png"),
        annotation_path=Path("unused.json"),
        features=payload,
        boxes=[],
        text="",
    )


def test_parse_srfund_relation_tree_counts_unique_edges_and_depth() -> None:
    tree = {
        "1": {
            "note": 2,
            "3": {"4": 5},
            "item_table_0": [{"6": 7}, {"6": 8}],
        },
        "other": [9, 10],
    }
    result = parse_srfund_relation_tree(tree)
    assert result["relation_graph_nodes"] == 10
    assert result["num_root_children"] == 3
    assert result["num_relation_links"] == 7
    assert result["max_hierarchy_depth"] == 4


def test_formstruct_graph_uses_same_entity_depth_convention() -> None:
    fields = [
        {
            "original_label": "Section",
            "bbox": [0, 0, 50, 20],
            "value": None,
            "keys": [
                {
                    "original_label": "Question",
                    "bbox": [0, 20, 40, 40],
                    "value": {"bbox": [40, 20, 80, 40]},
                }
            ],
        }
    ]
    features, boxes, text = extract_formstruct_graph(fields, 100, 100)
    assert features["num_entities"] == 3
    assert features["num_header_entities"] == 1
    assert features["num_question_entities"] == 1
    assert features["num_answer_entities"] == 1
    assert features["num_relation_links"] == 2
    assert features["max_hierarchy_depth"] == 3
    assert len(boxes) == 3
    assert "Section" in text and "Question" in text


def test_js_divergence_uses_union_support_and_is_symmetric() -> None:
    first = {"a": 1.0}
    second = {"b": 1.0}
    distance = jensen_shannon_divergence(first, second)
    assert np.isclose(distance, 1.0, atol=1e-9)
    assert np.isclose(distance, jensen_shannon_divergence(second, first))
    assert jensen_shannon_divergence(first, first) == 0.0


def test_gower_groups_are_equal_weighted() -> None:
    reference = [
        make_unit("r0", "en", {"num_entities": 0, "bbox_area_mean": 0.0}),
        make_unit("r1", "en", {"num_entities": 10, "bbox_area_mean": 1.0}),
    ]
    template = [make_unit("t", "de", {"num_entities": 10, "bbox_area_mean": 0.0})]
    config = fit_gower_config(reference)
    distances, groups = gower_distance_matrix(template, reference, config)
    assert distances.shape == (1, 2)
    assert set(groups) == set(config.active_groups)
    assert np.all((distances >= 0) & (distances <= 1))
    assert groups["language_script"][0, 0] > 0


def test_direct_only_gower_configuration_excludes_conditional_groups() -> None:
    reference = [make_unit("r0", "en", {}), make_unit("r1", "de", {})]
    config = fit_gower_config(reference, DIRECT_GOWER_GROUPS)
    assert set(config.active_groups) == {"hierarchy", "relation_structure", "spatial_layout", "language"}
    assert "table_presence" not in config.groups
    assert "writing_direction" not in config.groups.get("language", {}).get("categorical", [])


def test_stratified_reference_sets_are_disjoint_and_size_matched() -> None:
    form = [make_unit(f"t{i}", "en" if i < 2 else "de", {}) for i in range(4)]
    reference = [make_unit(f"r{i}", "en" if i < 8 else "de", {}) for i in range(16)]
    first, second, stratum = sample_reference_sets(
        form, reference, 4, np.random.default_rng(42), "stratified"
    )
    assert stratum == "language"
    assert len(first) == len(second) == 4
    assert set(first).isdisjoint(set(second))


def test_stratified_allocation_respects_pair_capacity() -> None:
    allocation = allocate_stratified_counts(
        ["a", "a", "a", "b"], ["a"] * 6 + ["b"] * 4, 4
    )
    assert sum(allocation.values()) == 4
    assert allocation["a"] <= 3
    assert allocation["b"] <= 2


def test_source_alias_handles_srfund_split_names() -> None:
    assert canonical_source_alias("ja_train_17") == "ja_17"
    assert canonical_source_alias("ja_val_17.jpg") == "ja_17"
    assert canonical_source_alias("0000989556") == "0000989556"
