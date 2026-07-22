# Hierarchical Structure Metrics

需要把本文档交给预测提供者执行时，请使用学生操作版：`README_HIERARCHICAL_METRICS_STUDENT_GUIDE.md`。

`hierarchical_metrics_report.py` 在已有页面预测上计算：

- `LG-GriTS-Top`：先固定 R-F1-compatible parent region matching，再按 grid bbox IoU 或 matched-region singleton 规则筛选候选，使用 reference-compatible factored 2D-MSS `GriTS_Top` 作为边权，最后做 page-level Hungarian matching。
- `WG-F1`：成员严格按 `IoU >= 0.5 + exact widget type + exact state` 匹配，再对 exact group type 的 group pair 做第二层 Hungarian matching。`unknown` 是普通状态，只能与 `unknown` 匹配。
- `Rel-F1`：先固定 region、state-agnostic widget、schema path field、LIG、grid/cell endpoint mapping，再评分 typed directed triples。关系预测不会反向优化节点匹配。

## GT provenance

grid/widget/relation GT 优先读取 `new-dataset-json/{template}.json` 的原始 Label Studio metadata；parent region 和 LIG 则复用 corrected R-F1/LIG-F1 的 `newdataset-layout` GT universe：

- cell span 来自闭区间 `row_start/row_end/col_start/col_end`；
- widget group 来自原始 parent-to-widget relation；
- widget state 按每个实例的 `answer.json` 与 option label 对齐，无法判定时保留为 `unknown`；
- widget type 以 `data_type` 为准；`mark_type=check/circle` 只描述标记样式，不改写 type 或 state；
- relation direction 来自原始 `from_id -> to_id`，relation type 根据 endpoint role 确定性派生；
- region matching 使用与既有 corrected R-F1 完全相同的类型、去重和 GT region 集合；
- LIG endpoint matching 接受与既有 LIG-F1 相同的 top-level groups、LIG regions 及 local-grid bbox/cell union。
- 重复 schema path 不会任取第一个 field；无法通过固定规则消歧时，该 field endpoint 保持未匹配。

raw metadata 中没有标出的空 grid slot 会规范化为 implicit blank cell。存在拓扑重叠或可定位 cell span 不完整的整个 GT grid 不进入正式适用集，并写入 metadata audit；无法定位到 grid 的 bbox 缺失 cell 单独审计。

## Aggregation

三个 page score 在预测和 GT 都为空时记 `NA`；只有一侧为空时记 `0`。因此 prediction-only 页面会进入 page macro，不会被静默排除。

当历史 run 的 `n_total` 小于总 index 时，只在预测文件名能够恢复全部 sample ID 的情况下评分；范围无法恢复的 run 记为 `NA`，不会假设它对应 index 前 N 页。

主报告包括 page-macro `Rel-F1`，附带：

- corpus-level micro precision/recall/F1；
- per-relation-type micro F1；
- page-macro 和 micro `Rel-F1 | matched endpoints`。

Per-type 附表保留预测中的未知/OOV relation type，并按其原字符串计为 false positive，不会静默丢弃或合并。

## Legacy prediction adapter

历史预测在内存中先经过 `hierarchical_prediction_adapter.py`，原始 JSON 文件不修改。兼容规则只使用预测自身声明和已经冻结的节点匹配，不读取 GT relation 来选择转换结果：

- relation 字段接受无冲突的 `source/from/u/parent`、`target/to/v/child`、`type/relation_type/r`；缺少端点或无法确定 type 的容器项不构成 typed triple，并计入 rejected audit；
- endpoint 接受 namespace 唯一的原始 ID，以及 `cells.c1`、`regions.r1` 等显式 namespace；不使用 node text、label、relation id 或编号相似性猜 ID；
- `label-value -> key-value`、`label_to_widget -> field-widget` 等已声明旧词汇映射到统一 ontology；缺失 type 只在预测端点角色能够唯一确定类型时补全；
- 嵌套 grid cell 可从同 ID 的顶层 cell 补齐缺失字段，但独立顶层 cell 不会被猜进任意 grid；
- `region_id` 显式指向另一 predicted grid 的片段，仅在合并后的 span matrix 完整且无重叠时合并，relation 中的片段 ID 同步指向 root grid；
- grid row/column index 会平移到局部 zero-based origin；该平移不改变二维拓扑；
- 缺失或悬空 grid parent 只在 topology 有效、所有 cell bbox 完整、恰好一个已通过 R-F1 匹配的 predicted region 完整包含 cell union 时恢复；
- 没有显式 `widget_groups`、legacy member list 或 widget `group_id` 时，仍按定义保留 singleton groups。

正式 CSV 中的 `adapter_*` 列报告每个 run 的转换和拒绝计数。无法从旧输出无歧义恢复的结构保持未匹配，不会从模板 metadata 回填成预测。

## Commands

主实验历史预测：

```bash
.venv/bin/python -m formtsr_exp.hierarchical_metrics_report \
  --index outputs/main_exp/dataset_index.jsonl \
  --pred-root outputs/main_exp/pred \
  --main-results outputs/main_exp/main_results.csv \
  --out outputs/main_exp
```

Qwen3.5 FormTSR LoRA test split：

```bash
.venv/bin/python -m formtsr_exp.hierarchical_metrics_report \
  --index outputs/dataset_splits/template_stratified_seed42/test_index.jsonl \
  --pred-root outputs/qwen35_formtsr_lora_test/pred \
  --main-results outputs/qwen35_formtsr_lora_test/main_results.csv \
  --out outputs/qwen35_formtsr_lora_test
```

产物：

- `hierarchical_structure_metrics.csv/.md`
- `hierarchical_relation_type_metrics.csv`
- `hierarchical_structure_metrics_metadata.json`
- `hierarchical_structure_per_sample/{run_id}.jsonl`
