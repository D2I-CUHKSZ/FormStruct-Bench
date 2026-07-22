from __future__ import annotations


PROMPT = """You are extracting the visible form structure and semantic answer tree needed for FormTSR-Bench evaluation.

Return only strict JSON. Do not include markdown fences, prose, comments, or explanations.
Do not include thinking tags such as <think> or </think>. Do not output hidden reasoning.

JSON validity requirements:
- The response must be parseable by a standard JSON parser.
- Every object member must be a "key": value pair. Do not put a bare string inside an object.
- Escape quotation marks and line breaks inside strings.
- Close every object and array that you open.
- Do not repeat the same object key in the same object. Do not repeat the same key/value sequence.
- If the response is getting long, preserve valid JSON by omitting lower-priority optional structure items and closing the JSON object.
- Output the top-level keys exactly in this order: "regions", "widgets", "local_grids", "cells", "line_item_groups", "relations", "answer".
- Priority order is: valid top-level skeleton first, then a non-empty and complete "answer", then concise "regions", then "widgets", then "line_item_groups". Keep "local_grids", "cells", and "relations" empty unless very short.
- Do not leave "answer" empty when any readable field value, selected option, signature, date, or paragraph value is visible.
- Do not cap, summarize, or intentionally omit visible fields from "answer". If the response is long, reduce optional visual structure first, not "answer".
- Keep the structure concise: use at most 60 regions and at most 40 widgets. Preserve enough structure for localization, but reserve output budget for "answer".

Required top-level schema:
{
  "regions": [
    {"id": "r1", "type": "title|section|field|value|text|widget|table|other", "bbox": [x1, y1, x2, y2], "text": "visible label or text"}
  ],
  "widgets": [
    {"id": "w1", "type": "checkbox|radio|input|signature|other", "bbox": [x1, y1, x2, y2], "label": "visible option or field label", "selected": true}
  ],
  "local_grids": [
    {"id": "g1", "region_id": "r_table", "cells": [{"id": "c1", "row": 0, "col": 0, "rowspan": 1, "colspan": 1, "bbox": [x1, y1, x2, y2], "text": "visible cell text"}]}
  ],
  "cells": [
    {"id": "c1", "row": 0, "col": 0, "rowspan": 1, "colspan": 1, "bbox": [x1, y1, x2, y2], "text": "visible cell text"}
  ],
  "line_item_groups": [
    {"id": "lig1", "bbox": [x1, y1, x2, y2], "text": "repeated row or grouped line-item area"}
  ],
  "relations": [],
  "answer": {
    "visible form title or top-level section": {
      "visible field label": "visible filled value",
      "nested visible section": {
        "visible field label": "visible filled value"
      }
    }
  }
}

Answer rules:
- The "answer" object is mandatory. It must contain the visible filled form values organized under the visible form title or top-level section.
- Include all readable answer fields. Do not limit "answer" by field count, nesting depth, string length, or section count.
- Preserve visible labels as keys as closely as possible, including the original language/script.
- Preserve nested section/field hierarchy.
- Use strings for filled values. Use objects for nested sections and arrays for repeated line-item rows.
- For selected checkboxes/radio buttons, output the selected option label or value in the corresponding answer field.
- For groups of checkbox/radio options, do not repeat every unselected option as separate answer keys. Use one answer field with the selected option/value, or an array of selected options when multiple are selected. The visual option boxes belong in "widgets".
- If a visible field is blank, use an empty string.
- If a section contains a paragraph/comment without a clear field label, represent it as {"value": "the visible text"} rather than as a bare string.
- If several visible labels are identical, merge them into one key or add a short disambiguating suffix. Never emit duplicate keys in the same object.

Structure rules:
- Use normalized page coordinates from 0 to 1000 as [left, top, right, bottom]. Scale x by page width and y by page height independently; require 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000.
- Include visible titles, section headers, field labels, filled value boxes/text areas, and checkbox/radio/input widgets as regions.
- Each region must have id, type, bbox, and text.
- Include widgets separately in "widgets" with selected=true/false when visually determinable.
- Use stable ids.
- Add line_item_groups for visible repeated item groups, repeated row blocks, or grouped table/list item areas. Use one bbox per repeated group.
- Add local_grids/cells only for table-like repeated rows when row/column positions are visually clear. Each cell must have row, col, rowspan, colspan, bbox, and text.
- Set "relations": [] unless a few relation edges are obvious and short.
- Do not invent unreadable text or uncertain boxes. Omit items that are not visually supported.
- Keep widgets/line_item_groups/local_grids/cells/relations empty if including them would risk invalid or incomplete JSON.

Output a single JSON object with all top-level keys shown above, in the required order. Empty arrays are allowed when no such structure is visible.
"""


HIERARCHICAL_PROMPT = """Extract the visible form content and its explicit structural relationships for FormTSR-Bench.

Return only one strict JSON object. Do not output markdown, prose, comments, thinking tags, or hidden reasoning. The JSON must parse with a standard parser and must use these top-level keys exactly in this order:
"regions", "widgets", "local_grids", "cells", "line_item_groups", "relations", "answer".

Output contract:
{
  "regions": [
    {"id":"r1","type":"title|section|field|value|text|widget|table|other","bbox":[x1,y1,x2,y2],"text":"visible label"}
  ],
  "widgets": [
    {"id":"w1","type":"checkbox|radio|input|signature|other","bbox":[x1,y1,x2,y2],"label":"visible option","state":"selected|unselected|unknown|filled|blank","group_id":"wg1","group_type":"checkbox|checkbox_multi|radio|input|signature|mixed"}
  ],
  "local_grids": [
    {"id":"g1","region_id":"r_table","cells":[{"id":"c1","row":0,"col":0,"rowspan":1,"colspan":1,"bbox":[x1,y1,x2,y2]}]}
  ],
  "cells": [],
  "line_item_groups": [
    {"id":"l1","bbox":[x1,y1,x2,y2]}
  ],
  "relations": [
    {"u":"r1","r":"key-value|parent-child|field-widget|key-to-cell","v":"r2"}
  ],
  "answer": {"visible section":{"visible field":"visible value"}}
}

Validity and budget rules:
- Close every object and array. Escape quotes and line breaks in strings. Never repeat a key in one object.
- Use at most 80 regions, 80 widgets, 10 local grids, 160 nested cells, 20 line-item groups, and 220 relations.
- If space is limited, preserve a valid complete answer first. Then preserve whole widget groups, whole grids with their parent regions and cells, and relation-bearing regions with their relations. Omit a whole low-priority structure group instead of leaving dangling IDs or partial grids.
- Keep labels concise. Do not duplicate nested grid cells in the top-level cells array; leave top-level cells empty when cells are already nested under local_grids.

Answer rules:
- answer is mandatory and must contain all readable filled values, selected options, dates, signatures, and paragraph values.
- Preserve visible labels and the original language/script. Preserve nested section hierarchy.
- Use strings for values, objects for nested sections, arrays for repeated rows, and an empty string for a visibly blank field.
- For a paragraph without a clear field label, use {"value":"visible text"}. Never invent unreadable content.

Structure rules:
- Coordinates are normalized independently to the page width and height and must satisfy 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
- IDs must be unique across every emitted region, widget, grid, cell, and line-item group.
- Every widget must include group_id and group_type. Widgets belonging to one option set share the same group_id. Use state=unknown only when the mark state cannot be determined.
- local_grids must contain complete zero-based row/column topology. Keep implicit blank cells needed to make the grid rectangular; bbox may be omitted only for an implicit cell.
- Every grid region_id and every relation u/v endpoint must reference an emitted ID. Emit only the four canonical relation types shown above.
- Include visible relation-bearing fields/values and widget groups before unrelated decorative regions.

Output only the JSON object.
"""


DEEPSEEK_OCR_JSON_PROMPT = """Read the form image and extract visible form content for evaluation.

Return only one strict JSON object. Do not include markdown fences, prose, comments, explanations, or hidden reasoning.

The JSON object must contain exactly these top-level keys in this order:
regions, widgets, local_grids, cells, line_item_groups, relations, answer.

Use these rules:
- answer is mandatory and must contain all readable filled form values, selected options, dates, signatures, and paragraph values.
- Preserve visible labels as JSON keys as closely as possible, including the original language or script.
- Preserve nested section hierarchy when it is visually clear.
- Use strings for filled values, objects for nested sections, and arrays for repeated rows.
- Use an empty string for visible blank fields.
- Do not copy placeholder phrases from the instruction. Do not output keys or values like visible field label, visible filled value, nested section, or repeated row.
- Do not invent unreadable text.
- For regions, include concise visible titles, section headers, field labels, value areas, and widgets when possible.
- For widgets, include checkbox/radio/input/signature controls when visible, with selected true or false when determinable.
- Use normalized page coordinates from 0 to 1000 as [left, top, right, bottom]. Scale x by page width and y by page height independently. If a bbox is uncertain, omit that item instead of using dummy coordinates.
- Use empty arrays for local_grids, cells, line_item_groups, or relations unless they are visually clear.
- Keep the JSON valid. If output is getting long, reduce regions/widgets first and keep answer complete.
"""

MINIMAL_JSON_PROMPT = DEEPSEEK_OCR_JSON_PROMPT


COMPACT_VALID_JSON_PROMPT = """Extract the visible form content from the page image.

Return only one strict JSON object. Do not include markdown, prose, comments, or hidden reasoning.
The JSON must parse with a standard JSON parser.

Use exactly these top-level keys in this order:
regions, widgets, local_grids, cells, line_item_groups, relations, answer.

Hard validity rules:
- Close every array/object you open.
- Do not repeat the same key in the same object.
- Do not repeat the same field/value pair.
- If two visible labels are identical, merge them into one key, add a short suffix, or use an array.
- If the same section title appears more than once, merge all fields into the first section object. Never open the same section key repeatedly.
- Use at most one top-level key per visually distinct section. If hierarchy is unclear, use a single top-level key named "form".
- Stop after all unique visible fields are represented, then close the JSON. Do not continue with filler or repeated blank fields.
- Do not invent numbered variants such as "(1)", "(2)", "(3)" to avoid duplicate keys unless those exact numbers are visibly printed in the form.
- If a label/value or section/value pair repeats, keep it once or use one array, then move on. Never repeat the same semantic field under new synthetic names.
- If output is getting long, keep "answer" complete and reduce visual structure. Never produce partial JSON.

Visual structure rules:
- Use normalized page coordinates from 0 to 1000 as [left, top, right, bottom], scaling x and y independently by page width and height.
- Keep "regions" concise: include only page title, major section headers, and a few representative field/value areas needed to localize the form.
- Keep "widgets" concise: include selected checkbox/radio controls and the most important visible widget groups. Include selected true/false when clear.
- Use [] for local_grids, cells, line_item_groups, and relations unless a very small table or repeated group is visually obvious.
- Do not enumerate every line, every blank field, every cell, or every unselected option.

Answer rules:
- "answer" is mandatory and must contain all readable filled form values, selected options, dates, signatures, and paragraph values.
- Preserve visible labels as keys as closely as possible, including the original language/script.
- Preserve nested section hierarchy when visually clear.
- Use strings for filled values, objects for nested sections, and arrays for repeated rows or repeated selected values.
- Use an empty string only for a visible blank field that is important to the form. Do not generate many blank fields.
- For selected checkbox/radio groups, output one answer field with the selected option/value, or an array when multiple options are selected.
- Do not invent unreadable text.

Output shape:
{
  "regions": [{"id": "r1", "type": "title|section|field|value|text|widget|table|other", "bbox": [x1, y1, x2, y2], "text": "visible text"}],
  "widgets": [{"id": "w1", "type": "checkbox|radio|input|signature|other", "bbox": [x1, y1, x2, y2], "label": "visible label", "selected": true}],
  "local_grids": [],
  "cells": [],
  "line_item_groups": [],
  "relations": [],
  "answer": {"form": {"field label": "filled value"}}
}
"""


STEP3_COMPACT_JSON_PROMPT = """Extract a compact FormTSR JSON object from the page image.

Return only strict JSON. No markdown, prose, comments, or reasoning.
Use exactly these top-level keys in order:
regions, widgets, local_grids, cells, line_item_groups, relations, answer.

Validity and anti-repetition rules:
- Output one complete JSON object and stop immediately after the final }.
- Never repeat the same key, value, section, phrase, or sentence to fill space.
- If a field appears to contain repeated or unclear text, keep the shortest readable value once.
- If a value is long, keep at most 120 visible characters. For paragraphs, keep at most one concise sentence.
- For annual evaluation/history fields, write one short value or an array of at most 3 short items. Do not repeat years/scores.
- Do not enumerate blank fields or unselected options in answer.
- If uncertain, use a short empty string rather than a long guess.

Structure rules:
- Use normalized page coordinates from 0 to 1000 as [left, top, right, bottom], scaling x and y independently by page width and height.
- regions: at most 8 items; include page title and major section headers only.
- widgets: at most 8 items; include visible selected checkbox/radio controls, plus clear unselected controls only when useful.
- local_grids, cells, line_item_groups, relations: use [] unless a small repeated table/group is obvious.

Answer rules:
- answer is mandatory and should contain readable filled values, selected options, dates, signatures, and short paragraph values.
- Preserve visible labels as keys as closely as possible, including original script.
- Use one top-level answer key "form" unless the hierarchy is very clear.
- Use strings for scalar values, arrays for multiple selected options, and objects only for clear nested sections.

Output shape:
{
  "regions": [{"id": "r1", "type": "title|section|field|value|text|widget|table|other", "bbox": [x1, y1, x2, y2], "text": "visible text"}],
  "widgets": [{"id": "w1", "type": "checkbox|radio|input|signature|other", "bbox": [x1, y1, x2, y2], "label": "visible label", "selected": true}],
  "local_grids": [],
  "cells": [],
  "line_item_groups": [],
  "relations": [],
  "answer": {"form": {"field label": "short filled value"}}
}
"""


DEEPSEEK_OCR_MARKDOWN_PROMPT = "<|grounding|>Convert the document to markdown."


def build_prompt(compact_structure: bool = False) -> str:
    if not compact_structure:
        return PROMPT
    return (
        PROMPT
        + "\nCompact structure mode for local debugging only. Do not use this mode for reported test runs:\n"
        + "- Keep the answer object complete, but make the visual structure intentionally small for quick connectivity checks.\n"
        + "- Include at most 12 regions total. Prioritize the page title, section headers, 3 representative field labels, and their filled values.\n"
        + "- Include at most 8 widgets total. Prioritize selected checkboxes/radio buttons and their option labels.\n"
        + "- Use empty arrays for line_item_groups, local_grids, cells, and relations unless they can be expressed in fewer than 8 short items.\n"
        + "- Do not enumerate every field, option, or relation in compact mode.\n"
        + "- Keep JSON concise while preserving all required top-level keys.\n"
    )


def build_model_prompt(model_config: dict, default_prompt: str) -> str:
    prompt_override = model_config.get("prompt_override")
    if isinstance(prompt_override, str) and prompt_override.strip():
        return prompt_override
    prompt_variant = str(model_config.get("prompt_variant") or "").strip()
    if prompt_variant == "hierarchical_full":
        return HIERARCHICAL_PROMPT
    if prompt_variant == "minimal_json":
        return MINIMAL_JSON_PROMPT
    if prompt_variant == "compact_valid_json":
        return COMPACT_VALID_JSON_PROMPT
    if prompt_variant == "step3_compact_json":
        return STEP3_COMPACT_JSON_PROMPT
    if prompt_variant == "deepseek_ocr_json":
        return DEEPSEEK_OCR_JSON_PROMPT
    if prompt_variant == "deepseek_ocr_markdown":
        return DEEPSEEK_OCR_MARKDOWN_PROMPT
    return default_prompt
