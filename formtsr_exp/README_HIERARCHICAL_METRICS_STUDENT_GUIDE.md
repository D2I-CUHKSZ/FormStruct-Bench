# 三个新增结构指标：预测补测说明

本文档用于在已有页面预测 JSON 上补算以下三个指标：

1. `LG-GriTS-Top`：local grid 二维拓扑质量；
2. `WG-F1`：widget group 及成员质量；
3. `Rel-F1`：有向 typed relation 质量。

评测入口为：

```text
formtsr_exp/hierarchical_metrics_report.py
```

请使用仓库中的正式 evaluator，不要自行重新实现公式。提交结果时应同时提供聚合 CSV、逐页 JSONL、relation-type CSV 和 metadata audit。

## 1. 最重要的评测原则

模板 metadata 只用于构造 GT 和固定节点匹配，不会替预测补结构。因此：

- 预测没有 `local_grids`，`LG-GriTS-Top` 通常为 0；
- 预测没有 widget，`WG-F1` 通常为 0；
- 预测没有合法的 typed relations，`Rel-F1` 为 0；
- 不允许把 `new-dataset-json` 或 `newdataset-layout` 中的 GT 节点复制到预测文件；
- relation 不能反过来改变 region、widget、grid 或 cell 的节点匹配。

预测 ID 不需要和 GT ID 相同，但同一页面内应稳定且唯一。Relation endpoint 必须引用该预测 JSON 中实际声明的节点 ID。

## 2. LG-GriTS-Top

### 2.1 表示

页面的 GT local grids 和预测 local grids 分别为：

```text
G* = {g1*, ..., gn*}
G_hat = {g1_hat, ..., gm_hat}
```

每个 grid 至少包含：

- `id`；
- `region_id`：parent semantic region 的预测 ID；
- `cells`：二维 cell matrix；
- 每个 cell 的 `row`、`col`、`rowspan`、`colspan`；
- 建议提供 grid/cell bbox。

### 2.2 Grid pair 何时允许匹配

预测 grid 和 GT grid 必须先满足 parent region 已通过 corrected R-F1 建立对应，然后满足以下任一条件：

- grid bbox IoU 不低于 `0.5`；
- 二者位于同一个已匹配 parent region，且该 region 在预测和 GT 中都只有一个 local grid。

允许匹配的 grid pair 使用 `GriTS_Top` 作为相似度。页面内再用 Hungarian matching 寻找最大总相似度：

\[
s_G(\hat g_i,g_j^*)=\operatorname{GriTS}_{\mathrm{Top}}(\hat g_i,g_j^*),
\]

\[
\operatorname{LG\text{-}GriTS}_{\mathrm{Top}}
=\frac{2\sum_{(\hat g_i,g_j^*)\in\mathcal M_G}s_G(\hat g_i,g_j^*)}
{|\hat{\mathcal G}|+|\mathcal G^*|}.
\]

正式值是页面分数的 macro average；`LG-GriTS-Top-corpus` 是把所有页面的匹配相似度和预测/GT grid 数汇总后的诊断值，不要用它替代主指标。

### 2.3 Cell topology 要求

- `row`、`col` 为整数；推荐从 0 开始；
- `rowspan`、`colspan` 为正整数，缺省时按 1；
- merged cell 只写一次，并通过 span 覆盖多个 slot；
- 同一 grid 内不能有重叠 cell；
- cell matrix 不能有未声明的洞；
- `region_id` 应指向 `regions` 中实际存在的 parent；
- 如果 grid bbox 缺失，只有所有 cell bbox 都合法时 evaluator 才会取 cell bbox union。

Evaluator 会兼容一部分旧格式，例如局部 row/col offset、显式相互引用的 split-grid fragments，以及满足严格唯一条件的缺失 parent；这些转换都会写入 `adapter_*` 审计列。

## 3. WG-F1

### 3.1 Widget 和 group 表示

Widget：

```text
v = (bbox, type, state)
```

Widget group：

```text
c = (group_type, members)
```

建议使用以下 widget type：

- `checkbox`
- `radio`
- `character_box`
- `blank_line`
- `signature`

常用 state：

- checkbox/radio：`selected`、`unselected`；
- character box/blank line/signature：`filled`、`blank`；
- 无法判断时：`unknown`。

`unknown` 不是通配符，只能匹配 GT 的 `unknown`。

### 3.2 成员匹配

两个 widget 同时满足以下条件才匹配：

```text
bbox IoU >= 0.5
widget type 完全相同
state 完全相同
```

每对 group 内先做一次最大二分图成员匹配，得到成员 F1；只有 `group_type` 完全相同，group pair 才有非零相似度。页面内再对 group 做第二层 Hungarian matching：

\[
s_W(\hat c_i,c_j^*)=
\mathbbm 1[\hat\tau_i=\tau_j^*]
\frac{2|\mathcal M_V|}{|\hat V_i|+|V_j^*|},
\]

\[
\operatorname{WG\text{-}F1}
=\frac{2\sum_{(\hat c_i,c_j^*)\in\mathcal M_C}s_W(\hat c_i,c_j^*)}
{|\hat{\mathcal C}|+|\mathcal C^*|}.
\]

正式值是页面分数的 macro average；`WG-F1-corpus` 只作为跨页面汇总诊断。

没有显式 group 的 widget 会被规范化为 singleton group。不要期望 evaluator 根据相邻位置、相同 label 或 GT group 自动猜分组。

### 3.3 推荐的 group 写法

```json
"widgets": [
  {
    "id": "w_yes",
    "type": "checkbox",
    "bbox": [120, 300, 145, 325],
    "state": "selected",
    "label": "Yes"
  },
  {
    "id": "w_no",
    "type": "checkbox",
    "bbox": [220, 300, 245, 325],
    "state": "unselected",
    "label": "No"
  }
],
"widget_groups": [
  {
    "id": "wg_consent",
    "group_type": "checkbox_multi",
    "members": ["w_yes", "w_no"]
  }
]
```

也可以用布尔字段 `selected: true/false` 表示 checkbox/radio 状态。

## 4. Rel-F1

### 4.1 Relation 表示

每条关系是有向 typed triple：

```text
e = (source, relation_type, target)
```

推荐写法：

```json
"relations": [
  {"source": "r_section", "type": "parent-child", "target": "r_name_key"},
  {"source": "r_name_key", "type": "key-value", "target": "r_name_value"},
  {"source": "r_consent_key", "type": "field-widget", "target": "w_yes"},
  {"source": "r_table_key", "type": "key-to-cell", "target": "cell_0_0"}
]
```

当前 evaluator/metadata 中可能出现的 canonical type 包括：

- `parent-child`
- `key-value`
- `field-widget`
- `key-to-cell`
- `key-to-field`
- `section-membership`
- `line-item-membership`
- `reading-order`

Relation type 完全匹配且方向正确才可能成为 TP。默认没有对称 relation；只有项目负责人明确指定时才使用 `--symmetric-relations`。

旧预测中的 `from/to`、`parent/child`、`label-value`、`label_to_widget` 等有限别名会经过 adapter 规范化。不要依赖 adapter 猜任意文本或 malformed relation。

### 4.2 Endpoint mapping 在关系评分前固定

Endpoint 映射不会使用 relation 预测优化：

- region：沿用 corrected R-F1 mapping；
- widget：相同 type 且 bbox IoU 不低于 0.5，忽略 state；
- field：使用无歧义的规范化 schema path；
- line-item group：沿用 corrected LIG-F1 mapping；
- grid/cell：沿用 LG-GriTS 产生的 grid/cell alignment。

预测关系正确，当且仅当 source 和 target 都成功映射、relation type 完全相同、方向正确。

令预测 relation 映射到 GT 后的正确集合大小为 `TP`：

\[
P_{\mathrm{rel}}=\frac{TP}{|\hat{\mathcal E}|},\qquad
R_{\mathrm{rel}}=\frac{TP}{|\mathcal E^*|},
\]

\[
\operatorname{Rel\text{-}F1}
=\frac{2P_{\mathrm{rel}}R_{\mathrm{rel}}}
{P_{\mathrm{rel}}+R_{\mathrm{rel}}}
=\frac{2TP}{|\hat{\mathcal E}|+|\mathcal E^*|}.
\]

重复的完全相同 relation 按 set semantics 去重，不会增加 TP。

正式主表使用 page-macro `Rel-F1`。附录还应报告：

- `Rel-Precision-micro`
- `Rel-Recall-micro`
- `Rel-F1-micro`
- `Rel-F1-matched-endpoints`
- `Rel-F1-matched-endpoints-micro`
- 各 relation type 的 micro F1

`Rel-F1-matched-endpoints` 用于区分“节点没有识别/定位正确”和“节点正确但 relation 预测错误”。

对三个指标，若某页预测集合和 GT 集合都为空，该页记为 `NA`，不进入 macro mean；只有一侧为空时记为 0。Missing prediction 不会因此自动从正式分母中消失：只要该页 GT 非空，页面分数就是 0。

## 5. 推荐预测 JSON

下面是一个结构示例。字段内容和坐标仅用于说明格式，不是可复制到真实预测中的答案。

```json
{
  "regions": [
    {
      "id": "r_section",
      "type": "section",
      "bbox": [80, 100, 920, 180],
      "text": "Personal information"
    },
    {
      "id": "r_name_key",
      "type": "field",
      "bbox": [100, 210, 300, 250],
      "text": "Name"
    },
    {
      "id": "r_name_value",
      "type": "value",
      "bbox": [320, 210, 700, 250],
      "text": "Alice"
    },
    {
      "id": "r_table_key",
      "type": "field",
      "bbox": [100, 360, 300, 395],
      "text": "Items"
    },
    {
      "id": "r_table",
      "type": "table",
      "bbox": [100, 400, 900, 700],
      "text": "Items"
    }
  ],
  "widgets": [
    {
      "id": "w_yes",
      "type": "checkbox",
      "bbox": [120, 300, 145, 325],
      "state": "selected",
      "label": "Yes"
    }
  ],
  "widget_groups": [
    {
      "id": "wg_consent",
      "group_type": "checkbox",
      "members": ["w_yes"]
    }
  ],
  "local_grids": [
    {
      "id": "grid_items",
      "region_id": "r_table",
      "bbox": [100, 400, 900, 700],
      "cells": [
        {
          "id": "cell_0_0",
          "row": 0,
          "col": 0,
          "rowspan": 1,
          "colspan": 1,
          "bbox": [100, 400, 500, 500],
          "text": "Item"
        },
        {
          "id": "cell_0_1",
          "row": 0,
          "col": 1,
          "rowspan": 1,
          "colspan": 1,
          "bbox": [500, 400, 900, 500],
          "text": "Amount"
        }
      ]
    }
  ],
  "cells": [],
  "line_item_groups": [],
  "relations": [
    {"source": "r_section", "type": "parent-child", "target": "r_name_key"},
    {"source": "r_name_key", "type": "key-value", "target": "r_name_value"},
    {"source": "r_table_key", "type": "key-to-cell", "target": "cell_0_0"}
  ],
  "answer": {
    "Personal information": {
      "Name": "Alice"
    }
  }
}
```

建议所有节点 ID 在页面内全局唯一。确实存在跨类型同名 ID 时，relation endpoint 可显式写成 `regions.r1`、`widgets.w1`、`local_grids.g1`、`cells.c1` 或 `line_item_groups.lig1`。

## 6. 准备预测目录

运行前必须具备：

- 与负责人一致的 evaluator 代码版本；
- 项目 Python 环境 `.venv`，其中已安装 `numpy` 和 `scipy`；
- `new-dataset-json/` 和 `newdataset-layout/`；
- index 中 `label_path` 指向的 `FormTSR/datasets/.../answer.json`；
- 与所选 index 一一对应的预测文件。

以下示例使用 test split。必须先确认学生的预测确实对应这个 index：

```text
outputs/dataset_splits/template_stratified_seed42/test_index.jsonl
```

预测文件必须按 `sample_id` 命名：

```text
outputs/student_hierarchical_eval/
└── pred/
    └── student_model_v1/
        ├── Arabic-2__01.json
        ├── Arabic-2__02.json
        └── ...
```

其中 `student_model_v1` 是本次 run ID。不要在 run ID 中加入 `smoke` 或 `aligned_metadata`，否则报告会把它归类为诊断 run。

先检查单个文件是否合法：

```bash
.venv/bin/python -m json.tool \
  outputs/student_hierarchical_eval/pred/student_model_v1/Arabic-2__01.json \
  >/dev/null
```

记录代码版本并检查文件数：

```bash
git rev-parse HEAD
wc -l outputs/dataset_splits/template_stratified_seed42/test_index.jsonl
find outputs/student_hierarchical_eval/pred/student_model_v1 \
  -maxdepth 1 -type f -name '*.json' | wc -l
```

Index 中存在、预测目录中缺失的页面会按 missing prediction 处理，不应从 index 中删除。

## 7. 配置 bbox 坐标空间

新建：

```text
outputs/student_hierarchical_eval/bbox_coordinate_spaces.json
```

如果预测坐标范围是 0 到 1000：

```json
{
  "version": 1,
  "canonical_space": "normalized_0_1",
  "runs": {
    "student_model_v1": {
      "source_space": "normalized_1000",
      "evidence": "The prediction contract emits coordinates in [0, 1000]."
    }
  }
}
```

`source_space` 只能根据模型输出契约选择：

| 预测 bbox | `source_space` |
| --- | --- |
| `[0, 1000]` 归一化坐标 | `normalized_1000` |
| `[0, 1]` 归一化坐标 | `normalized_1` |
| 原图像素坐标 | `pixel` |
| 预测完全没有 bbox | `none` |

不要尝试多个坐标空间后选择分数最高者。坐标空间必须由预测生成方式确定，并在 `evidence` 中记录依据。

## 8. 运行评测

从仓库根目录 `.` 执行：

```bash
RUN_ID=student_model_v1
EVAL_ROOT=outputs/student_hierarchical_eval

.venv/bin/python -m formtsr_exp.hierarchical_metrics_report \
  --index outputs/dataset_splits/template_stratified_seed42/test_index.jsonl \
  --metadata-root new-dataset-json \
  --layout-root newdataset-layout \
  --pred-root "$EVAL_ROOT/pred" \
  --main-results "$EVAL_ROOT/main_results_not_present.csv" \
  --bbox-manifest "$EVAL_ROOT/bbox_coordinate_spaces.json" \
  --models "$RUN_ID" \
  --out "$EVAL_ROOT/report" \
  --workers 4
```

`main_results_not_present.csv` 应保持不存在。此时 evaluator 会从 `pred` 下的目录名发现 run，并使用完整 index 长度作为 `n_total`。

如果预测对应主实验 7000 页，改用：

```text
outputs/main_exp/dataset_index.jsonl
```

不要混用 test split 和主实验 index。

## 9. 应提交哪些结果

主要结果位于：

```text
outputs/student_hierarchical_eval/report/hierarchical_structure_metrics.csv
```

需要回填的三个主指标列：

```text
LG-GriTS-Top
WG-F1
Rel-F1
```

同时提供以下上下文列，不能只抄三个小数：

```text
model
run_type
comparison_status
bbox_source_space
sample_scope
n_total
n_valid_json
coverage
n_missing_prediction
n_invalid_json
n_lg_gt_applicable
n_grid_pred
n_grid_gt
n_grid_matches
n_wg_gt_applicable
n_widget_group_pred
n_widget_group_gt
n_rel_gt_applicable
n_relation_tp
n_relation_pred
n_relation_gt
Rel-F1-micro
Rel-F1-matched-endpoints
```

还应提交：

- `hierarchical_structure_metrics.md`
- `hierarchical_relation_type_metrics.csv`
- `hierarchical_structure_metrics_metadata.json`
- `hierarchical_structure_per_sample/student_model_v1.jsonl`

建议按以下格式汇报：

```text
run_id:
git commit:
evaluation index:
bbox source space:
n_valid_json / n_total:
LG-GriTS-Top:
WG-F1:
Rel-F1:
Rel-F1-micro:
Rel-F1-matched-endpoints:
notes:
```

不要手工四舍五入后覆盖 CSV。论文表格通常保留 6 位小数，但原始输出文件必须一并保留。

## 10. 0 分排查顺序

### 10.1 三项都为 0

依次检查：

1. `n_valid_json` 是否为 0；
2. 文件名是否与 index 的 `sample_id` 一致；
3. `bbox_source_space` 是否符合预测契约；
4. JSON 顶层是否真的包含非空 `regions/widgets/local_grids/relations`；
5. bbox 是否为合法 `[left, top, right, bottom]`，且 `left < right`、`top < bottom`。

### 10.2 LG-GriTS-Top 为 0

检查：

- `n_grid_pred` 是否为 0；
- grid 的 `region_id` 是否引用已声明 region；
- parent region 的 type/bbox 是否足以通过 R-F1；
- row/col/span 是否形成完整且无重叠的 matrix；
- `n_grid_matches` 是否为 0；
- `adapter_grid_rejected_items` 是否很高。

### 10.3 WG-F1 为 0

检查：

- `n_widget_group_pred` 是否为 0；
- widget bbox 是否正确；
- 是否使用了过于泛化的 `input`、`other` 或复合 type 字符串；
- `selected/unselected/filled/blank/unknown` 是否和任务定义一致；
- 是否缺失显式 group，导致所有 widget 被当成 singleton；
- `group_type` 是否与成员语义一致。

### 10.4 Rel-F1 为 0

检查：

- `adapter_relation_declared_items` 是否为 0；
- `adapter_relation_accepted_items` 是否远小于 declared；
- `adapter_relation_rejected_items` 是否很高；
- `n_relation_pred` 是否为 0；
- source/target 是否引用实际节点 ID，而不是 label 文本或 relation 自身 ID；
- relation type 和方向是否正确；
- `Rel-F1-matched-endpoints` 是否为 `NA` 或 0；
- region/widget/grid/cell 等 endpoint 自身是否先匹配成功。

如果 `Rel-F1-matched-endpoints` 为 `NA`，通常表示没有任何预测 relation 同时拥有两个已匹配 endpoint；此时首先修节点识别和定位，而不是只改 relation type。

## 11. 不允许的操作

- 不得用 GT metadata 补预测中缺失的 grid、widget、group 或 relation；
- 不得根据最终分数选择 bbox 坐标空间；
- 不得删除预测失败或缺失的 index 页面；
- 不得把 `unknown` 当作任意状态或任意 type；
- 不得反转 relation 方向来尝试命中 GT；
- 不得用 relation 结果重新优化 endpoint matching；
- 不得只报告 valid-json 子集上的均值；
- 不得修改 evaluator 后只提交数值而不提交代码和 metadata audit。

## 12. 代码与正式定义位置

- 指标核心：`formtsr_exp/hierarchical_metrics.py`
- GT 转换、聚合与 CLI：`formtsr_exp/hierarchical_metrics_report.py`
- 旧预测兼容：`formtsr_exp/hierarchical_prediction_adapter.py`
- 正式维护说明：`formtsr_exp/README_HIERARCHICAL_METRICS.md`
- 回归测试：`tests/test_hierarchical_metrics.py`
