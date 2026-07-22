from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


_SOURCE_KEYS = ("source", "from", "u", "parent")
_TARGET_KEYS = ("target", "to", "v", "child")
_TYPE_KEYS = ("type", "relation_type", "r")

_RELATION_TYPE_ALIASES = {
    "field-to-widget": "field-widget",
    "field_widget": "field-widget",
    "label-to-widget": "field-widget",
    "label_to_widget": "field-widget",
    "field-to-value": "key-value",
    "field_value": "key-value",
    "label-to-value": "key-value",
    "label-value": "key-value",
    "label_value": "key-value",
    "key_value": "key-value",
    "parent_child": "parent-child",
    "key_to_cell": "key-to-cell",
    "section_membership": "section-membership",
    "line_item_membership": "line-item-membership",
}

_CANONICAL_RELATION_TYPES = {
    "key-value",
    "parent-child",
    "field-widget",
    "key-to-cell",
    "key-to-field",
    "section-membership",
    "line-item-membership",
    "reading-order",
}

_NAMESPACE_ALIASES = {
    "region": "regions",
    "regions": "regions",
    "widget": "widgets",
    "widgets": "widgets",
    "grid": "local_grids",
    "grids": "local_grids",
    "local_grid": "local_grids",
    "local_grids": "local_grids",
    "cell": "cells",
    "cells": "cells",
    "line_item_group": "line_item_groups",
    "line_item_groups": "line_item_groups",
    "lig": "line_item_groups",
    "ligs": "line_item_groups",
}


def _token(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _scalar(value: Any) -> str | None:
    if value in (None, "") or isinstance(value, (dict, list, tuple, set, bool)):
        return None
    return str(value)


def _one_alias(item: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, bool]:
    values = [_scalar(item.get(key)) for key in keys]
    present = {value for value in values if value is not None}
    if len(present) > 1:
        return None, True
    return (next(iter(present)) if present else None), False


def _raw_type(item: Mapping[str, Any]) -> str:
    return _token(item.get("type") or item.get("region_type") or item.get("data_type")).replace(
        "-", "_"
    ).replace(" ", "_")


def _region_role(item: Mapping[str, Any]) -> str:
    raw = _raw_type(item)
    if raw in {"title", "section", "section_header", "field_group", "region"}:
        return "section"
    if raw in {"field", "field_label", "label", "key", "question"}:
        return "key"
    if raw in {"value", "field_value", "number", "date", "answer"}:
        return "value"
    if raw in {
        "checkbox",
        "checkbox_multi",
        "check_box",
        "radio",
        "radio_button",
        "widget",
    }:
        return "widget"
    if raw == "table":
        return "table"
    return "unknown"


def _relation_type_from_roles(source_role: str, target_role: str) -> str | None:
    if target_role == "widget" and source_role in {"key", "section"}:
        return "field-widget"
    if source_role == "key" and target_role == "key":
        return "parent-child"
    if source_role == "key" and target_role == "value":
        return "key-value"
    if source_role == "key" and target_role == "cell":
        return "key-to-cell"
    if source_role == "section" and target_role in {
        "section",
        "key",
        "value",
        "widget",
        "table",
        "grid",
        "cell",
        "line_item_group",
    }:
        return "section-membership"
    if "line_item_group" in {source_role, target_role}:
        return "line-item-membership"
    return None


def _relation_type_alias(value: str) -> str:
    token = _token(value)
    if token in _RELATION_TYPE_ALIASES:
        return _RELATION_TYPE_ALIASES[token]
    return token if token in _CANONICAL_RELATION_TYPES else value


def _items(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _merge_top_level_cells(prediction: dict[str, Any], audit: Counter[str]) -> None:
    top_cells = _items(prediction.get("cells"))
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in top_cells:
        cell_id = _scalar(cell.get("id"))
        if cell_id is not None:
            by_id[cell_id].append(cell)

    for grid in _items(prediction.get("local_grids")):
        cells = grid.get("cells")
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            cell_id = _scalar(cell.get("id"))
            candidates = by_id.get(cell_id or "", [])
            if len(candidates) != 1:
                continue
            enriched = False
            for key in (
                "row",
                "col",
                "column",
                "row_index",
                "col_index",
                "rowspan",
                "colspan",
                "row_start",
                "row_end",
                "col_start",
                "col_end",
                "bbox",
                "box",
                "bounds",
            ):
                if key not in cell and key in candidates[0]:
                    cell[key] = copy.deepcopy(candidates[0][key])
                    enriched = True
            if enriched:
                audit["adapter_grid_cells_enriched"] += 1


def _integer(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cell_rectangle(cell: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    row = _integer(cell.get("row", cell.get("row_index", cell.get("row_start"))))
    col = _integer(
        cell.get("col", cell.get("column", cell.get("col_index", cell.get("col_start"))))
    )
    if row is None or col is None or row < 0 or col < 0:
        return None
    rowspan = _integer(cell.get("rowspan"), 1)
    colspan = _integer(cell.get("colspan"), 1)
    if cell.get("rowspan") is None and cell.get("row_end") is not None:
        row_end = _integer(cell.get("row_end"))
        rowspan = row_end - row + 1 if row_end is not None and row_end >= row else None
    if cell.get("colspan") is None and cell.get("col_end") is not None:
        col_end = _integer(cell.get("col_end"))
        colspan = col_end - col + 1 if col_end is not None and col_end >= col else None
    if rowspan is None or colspan is None or rowspan < 1 or colspan < 1:
        return None
    return row, col, row + rowspan, col + colspan


def _valid_cell_matrix(cells: Sequence[Mapping[str, Any]]) -> bool:
    rectangles = [_cell_rectangle(cell) for cell in cells]
    if not rectangles or any(rectangle is None for rectangle in rectangles):
        return False
    parsed = [rectangle for rectangle in rectangles if rectangle is not None]
    min_row = min(rectangle[0] for rectangle in parsed)
    min_col = min(rectangle[1] for rectangle in parsed)
    max_row = max(rectangle[2] for rectangle in parsed)
    max_col = max(rectangle[3] for rectangle in parsed)
    total_area = sum(
        (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])
        for rectangle in parsed
    )
    if total_area != (max_row - min_row) * (max_col - min_col):
        return False
    for left_index, left in enumerate(parsed):
        for right in parsed[left_index + 1 :]:
            if max(left[0], right[0]) < min(left[2], right[2]) and max(
                left[1], right[1]
            ) < min(left[3], right[3]):
                return False
    return True


def _merge_split_grids(
    prediction: dict[str, Any], audit: Counter[str]
) -> dict[str, str]:
    original_grids = (
        prediction.get("local_grids")
        if isinstance(prediction.get("local_grids"), list)
        else []
    )
    raw_grids = _items(original_grids)
    grid_id_counts = Counter(
        grid_id
        for grid in raw_grids
        for grid_id in [_scalar(grid.get("id"))]
        if grid_id is not None
    )
    grid_by_id = {
        str(grid["id"]): grid
        for grid in raw_grids
        if _scalar(grid.get("id")) is not None
        and grid_id_counts[str(grid["id"])] == 1
    }
    region_ids = {
        item_id
        for region in _items(prediction.get("regions"))
        for item_id in [_scalar(region.get("id"))]
        if item_id is not None
    }

    def parent_ref(grid: Mapping[str, Any]) -> str | None:
        return _scalar(
            grid.get("region_id")
            or grid.get("parent_region_id")
            or grid.get("parent_id")
            or grid.get("parent")
        )

    def root_id(grid_id: str) -> str | None:
        current = grid_id
        visiting: set[str] = set()
        while True:
            if current in visiting:
                return None
            visiting.add(current)
            parent = parent_ref(grid_by_id[current])
            if parent not in grid_by_id or parent in region_ids:
                return current
            current = parent

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for grid_id in grid_by_id:
        root = root_id(grid_id)
        if root is not None:
            members_by_root[root].append(grid_id)

    aliases: dict[str, str] = {}
    replacements: dict[str, dict[str, Any]] = {}
    removed: set[str] = set()
    order = {
        str(grid.get("id")): index
        for index, grid in enumerate(raw_grids)
        if grid.get("id") is not None
    }
    for root, member_ids in members_by_root.items():
        if len(member_ids) < 2:
            continue
        ordered_ids = sorted(member_ids, key=lambda item_id: order[item_id])
        combined_cells = [
            copy.deepcopy(cell)
            for item_id in ordered_ids
            for cell in _items(grid_by_id[item_id].get("cells"))
        ]
        explicit_cell_ids = [
            cell_id
            for cell in combined_cells
            for cell_id in [_scalar(cell.get("id"))]
            if cell_id is not None
        ]
        if len(explicit_cell_ids) != len(set(explicit_cell_ids)) or not _valid_cell_matrix(
            combined_cells
        ):
            audit["adapter_grid_fragment_merges_rejected"] += 1
            continue
        merged = copy.deepcopy(grid_by_id[root])
        merged["cells"] = combined_cells
        for bbox_key in ("bbox", "box", "region_box", "bounds"):
            merged.pop(bbox_key, None)
        replacements[root] = merged
        for child_id in ordered_ids:
            if child_id == root:
                continue
            aliases[child_id] = root
            removed.add(child_id)
        audit["adapter_grid_fragment_groups_merged"] += 1
        audit["adapter_grid_fragments_merged"] += len(ordered_ids) - 1

    if replacements:
        normalized_grids: list[Any] = []
        for grid in original_grids:
            if not isinstance(grid, dict):
                normalized_grids.append(grid)
                continue
            grid_id = str(grid.get("id"))
            if grid_id not in removed:
                normalized_grids.append(replacements.get(grid_id, grid))
        prediction["local_grids"] = normalized_grids
    return aliases


def _explicit_widget_groups(prediction: dict[str, Any], audit: Counter[str]) -> None:
    if isinstance(prediction.get("widget_groups"), list):
        return

    legacy_fields = _items(prediction.get("fields"))
    groups = [
        copy.deepcopy(item)
        for item in legacy_fields
        if isinstance(item.get("members") or item.get("widgets") or item.get("items"), list)
    ]
    if groups:
        prediction["widget_groups"] = groups
        audit["adapter_widget_groups_from_fields"] += len(groups)
        return

    grouped: dict[str, list[str]] = defaultdict(list)
    group_types: dict[str, str] = {}
    for widget in _items(prediction.get("widgets")):
        widget_id = _scalar(widget.get("id"))
        group_id = _scalar(widget.get("widget_group_id") or widget.get("group_id"))
        if widget_id is None or group_id is None:
            continue
        grouped[group_id].append(widget_id)
        group_type = _scalar(widget.get("group_type"))
        if group_type is not None:
            group_types.setdefault(group_id, group_type)
    if grouped:
        prediction["widget_groups"] = [
            {
                "id": group_id,
                "group_type": group_types.get(group_id, ""),
                "members": member_ids,
            }
            for group_id, member_ids in sorted(grouped.items())
        ]
        audit["adapter_widget_groups_from_member_ids"] += len(grouped)


def _node_registry(
    prediction: Mapping[str, Any],
) -> tuple[
    dict[str, Counter[str]],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    namespace_counts: dict[str, Counter[str]] = defaultdict(Counter)
    roles_by_id: dict[str, set[str]] = defaultdict(set)
    roles_by_namespaced_id: dict[str, set[str]] = defaultdict(set)
    namespaces_by_id: dict[str, set[str]] = defaultdict(set)

    def add(namespace: str, item: Mapping[str, Any], role: str) -> None:
        item_id = _scalar(item.get("id"))
        if item_id is None:
            return
        namespace_counts[namespace][item_id] += 1
        namespaces_by_id[item_id].add(namespace)
        roles_by_id[item_id].add(role)
        roles_by_namespaced_id[f"{namespace}.{item_id}"].add(role)

    for item in _items(prediction.get("regions")):
        add("regions", item, _region_role(item))
    for item in _items(prediction.get("widgets")):
        add("widgets", item, "widget")
    for item in _items(prediction.get("line_item_groups")):
        add("line_item_groups", item, "line_item_group")
    for grid in _items(prediction.get("local_grids")):
        add("local_grids", grid, "grid")
        for cell in _items(grid.get("cells")):
            add("cells", cell, "cell")
    nested_cell_ids = set(namespace_counts["cells"])
    for cell in _items(prediction.get("cells")):
        cell_id = _scalar(cell.get("id"))
        if cell_id in nested_cell_ids:
            continue
        add("cells", cell, "cell")
    return namespace_counts, roles_by_id, roles_by_namespaced_id, namespaces_by_id


def _answer_aliases(answer: Any) -> dict[str, str | None]:
    aliases: dict[str, str | None] = {}

    def add(alias: str, target: str) -> None:
        aliases[alias] = target if alias not in aliases else None

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + (str(key),))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (str(index),))
        elif path:
            dot_path = ".".join(path)
            slash_path = "/".join(path)
            add(f"answer.{dot_path}", slash_path)
            add(f"answer/{slash_path}", slash_path)

    visit(answer, ())
    return aliases


def _explicit_namespace(value: str) -> tuple[str, str] | None:
    for separator in (".", "/", ":"):
        if separator not in value:
            continue
        prefix, suffix = value.split(separator, 1)
        namespace = _NAMESPACE_ALIASES.get(_token(prefix).replace("-", "_"))
        if namespace and suffix:
            return namespace, suffix
    return None


def _resolve_endpoint(
    value: str,
    side: str,
    relation_index: int,
    namespace_counts: Mapping[str, Counter[str]],
    namespaces_by_id: Mapping[str, set[str]],
    answer_aliases: Mapping[str, str | None],
    node_aliases: Mapping[str, str],
    audit: Counter[str],
) -> str:
    answer_target = answer_aliases.get(value)
    if answer_target is not None:
        audit["adapter_relation_endpoint_aliases"] += 1
        return answer_target

    explicit = _explicit_namespace(value)
    if explicit is not None:
        namespace, item_id = explicit
        if namespace == "local_grids" and item_id in node_aliases:
            audit["adapter_relation_endpoint_aliases"] += 1
            return f"local_grids.{node_aliases[item_id]}"
        if namespace_counts.get(namespace, Counter()).get(item_id) == 1:
            audit["adapter_relation_endpoint_aliases"] += 1
            return f"{namespace}.{item_id}"
        audit["adapter_relation_unresolved_endpoints"] += 1
        return f"__legacy_unresolved_{side}_{relation_index}__"

    namespaces = namespaces_by_id.get(value, set())
    if value in node_aliases and not namespaces:
        audit["adapter_relation_endpoint_aliases"] += 1
        return f"local_grids.{node_aliases[value]}"
    if len(namespaces) <= 1:
        if namespaces:
            namespace = next(iter(namespaces))
            if namespace_counts.get(namespace, Counter()).get(value) != 1:
                audit["adapter_relation_ambiguous_endpoints"] += 1
                return f"__legacy_ambiguous_{side}_{relation_index}__"
        return value

    audit["adapter_relation_ambiguous_endpoints"] += 1
    return f"__legacy_ambiguous_{side}_{relation_index}__"


def _endpoint_role(
    endpoint: str,
    roles_by_id: Mapping[str, set[str]],
    roles_by_namespaced_id: Mapping[str, set[str]],
    answer_aliases: Mapping[str, str | None],
) -> str:
    namespaced_roles = roles_by_namespaced_id.get(endpoint, set())
    if len(namespaced_roles) == 1:
        return next(iter(namespaced_roles))
    roles = roles_by_id.get(endpoint, set())
    if len(roles) == 1:
        return next(iter(roles))
    if endpoint in {target for target in answer_aliases.values() if target is not None}:
        return "value"
    return "unknown"


def _adapt_relations(
    prediction: dict[str, Any], audit: Counter[str], node_aliases: Mapping[str, str]
) -> None:
    raw_relations = prediction.get("relations")
    if not isinstance(raw_relations, list):
        raw_relations = prediction.get("edges")
    if not isinstance(raw_relations, list):
        raw_relations = prediction.get("relation_edges")
    if not isinstance(raw_relations, list):
        raw_relations = []
    audit["adapter_relation_declared_items"] += len(raw_relations)

    (
        namespace_counts,
        roles_by_id,
        roles_by_namespaced_id,
        namespaces_by_id,
    ) = _node_registry(prediction)
    answer_aliases = _answer_aliases(prediction.get("answer"))
    adapted: list[dict[str, str]] = []
    for index, item in enumerate(raw_relations):
        if not isinstance(item, dict):
            audit["adapter_relation_rejected_items"] += 1
            continue
        source, source_conflict = _one_alias(item, _SOURCE_KEYS)
        target, target_conflict = _one_alias(item, _TARGET_KEYS)
        relation_type, type_conflict = _one_alias(item, _TYPE_KEYS)
        if source_conflict or target_conflict or type_conflict or source is None or target is None:
            audit["adapter_relation_rejected_items"] += 1
            if source_conflict or target_conflict or type_conflict:
                audit["adapter_relation_alias_conflicts"] += 1
            continue

        resolved_source = _resolve_endpoint(
            source,
            "source",
            index,
            namespace_counts,
            namespaces_by_id,
            answer_aliases,
            node_aliases,
            audit,
        )
        resolved_target = _resolve_endpoint(
            target,
            "target",
            index,
            namespace_counts,
            namespaces_by_id,
            answer_aliases,
            node_aliases,
            audit,
        )
        if relation_type is None:
            source_role = _endpoint_role(
                resolved_source, roles_by_id, roles_by_namespaced_id, answer_aliases
            )
            target_role = _endpoint_role(
                resolved_target, roles_by_id, roles_by_namespaced_id, answer_aliases
            )
            relation_type = _relation_type_from_roles(source_role, target_role)
            if relation_type is None:
                audit["adapter_relation_rejected_items"] += 1
                audit["adapter_relation_missing_type_unresolved"] += 1
                continue
            audit["adapter_relation_types_inferred"] += 1
        else:
            canonical_type = _relation_type_alias(relation_type)
            if canonical_type != relation_type:
                audit["adapter_relation_type_aliases"] += 1
            relation_type = canonical_type

        adapted.append(
            {
                "source": resolved_source,
                "target": resolved_target,
                "type": relation_type,
            }
        )
        audit["adapter_relation_accepted_items"] += 1
    prediction["relations"] = adapted


def adapt_legacy_prediction(prediction: Any) -> tuple[dict[str, Any], dict[str, int]]:
    """Normalize legacy prediction spellings without using GT relation edges."""
    audit: Counter[str] = Counter()
    if not isinstance(prediction, dict):
        audit["adapter_non_object_prediction"] += 1
        return {}, dict(audit)
    adapted = copy.deepcopy(prediction)

    if not isinstance(adapted.get("regions"), list) and isinstance(
        adapted.get("region_boxes"), list
    ):
        adapted["regions"] = copy.deepcopy(adapted["region_boxes"])
        audit["adapter_top_level_aliases"] += 1
    if not isinstance(adapted.get("local_grids"), list) and isinstance(
        adapted.get("grids"), list
    ):
        adapted["local_grids"] = copy.deepcopy(adapted["grids"])
        audit["adapter_top_level_aliases"] += 1

    _merge_top_level_cells(adapted, audit)
    grid_aliases = _merge_split_grids(adapted, audit)
    _explicit_widget_groups(adapted, audit)
    _adapt_relations(adapted, audit, grid_aliases)
    return adapted, dict(audit)
