# FormTSR-Bench 辅助实验运行手册

本文档面向负责跑实验的同学，覆盖当前已经接入 `formtsr_exp`、但不属于主实验模型总表的实验：

| 实验 | 是否需要重新推理 | 主要目的 | 推荐输出目录 |
|---|---:|---|---|
| 难度分层 | 否 | 比较 L1-L4 难度下的性能和能力边界 | `outputs/aux_exp/difficulty` |
| 约束切片 | 否 | 衡量不同表单结构约束带来的性能下降 | `outputs/aux_exp/constraints` |
| 受控诊断验证 | 否 | 验证八个指标对七类孤立结构错误的敏感性与选择性 | `outputs/aux_exp/controlled_diagnostic` |
| 视觉退化鲁棒性 | 是，只跑退化图 | 比较 clean/degraded 的最新分层指标与 drop | `outputs/robustness_exp/report_latest` |
| metadata alignment | 否，诊断用途 | 检查字段路径不对齐是否造成低分 | `outputs/aux_exp/alignment` |

本文档不包含主实验重新推理、case study、RAG/few-shot 等其他研究分支。不要把 `scripts/run_few_shot*.sh` 或 RAG 脚本产生的结果混入本文档中的实验。

## 0. 用最新 metrics 整理主实验结果

这一节是当前主实验结果整理的唯一入口。已有 prediction 不需要重新调用模型，也不要从旧 `main_results.csv` 手工复制一张新表。

### 0.1 直接使用已经生成的结果

正式主表使用：

```text
outputs/main_exp/main_experiment_results.csv
```

配套文件：

| 文件 | 用途 |
|---|---|
| `outputs/main_exp/final_reporting_metrics.md` | 便于人工检查的 Markdown 主表 |
| `outputs/main_exp/final_reporting_metrics_table.tex` | 可直接放入论文的 LaTeX 表 |
| `outputs/main_exp/final_reporting_metrics_metadata.json` | 主表来源、筛选条件和指标口径 |
| `outputs/main_exp/corrected_structure_metrics.csv` | corrected R-F1/LIG-F1、precision/recall 和 bbox 审计计数 |
| `outputs/main_exp/corrected_structure_metrics.md` | 全部 run 的可读结构指标表 |
| `outputs/main_exp/corrected_structure_per_sample/{run}.jsonl` | corrected structure 逐页结果 |
| `outputs/main_exp/hierarchical_structure_metrics.csv` | 全部 run 的 LG-GriTS-Top、WG-F1 和 Rel-F1 汇总 |
| `outputs/main_exp/hierarchical_relation_type_metrics.csv` | relation-type micro F1 附表 |

`main_experiment_results.csv` 只保留 raw/pred/error 的 sample-id 并集覆盖完整 7,000 页、且至少有一个有效预测的 run。同一 `model_id` 有多个全量 run 时，先按有效预测数、再按 `Schema-nTED / Value-nED / TSR-path / R-F1 / LIG-F1` 选择最佳结果。当前共 9 个模型，每行都有 `n_attempted=7000`。不得再加入 `_aligned_metadata`、smoke、failed 或 partial run。

### 0.2 主表保留的列

正式报告分为原有 extraction/spatial 主表和 hierarchical structure companion table。后者表头为：

```text
Model | LG-GriTS-Top | WG-F1 | Rel-F1
```

| 层级 | 列 | 解释 |
|---|---|---|
| 系统覆盖 | `Valid/7000` | 可解析预测数；不能只报指标均值而省略覆盖率 |
| 整页严格 | `Exact Pages` / `Page-EM` | 完整 answer tree 全对的页面数；正文优先写 `3/7000`，不要只写很小的小数 |
| 文档 schema | `Schema-nTED` | 只看 key、容器类型和层级的软 tree similarity |
| 文档内容 | `Value-nED` | 忽略 path 的软 value matching，同时惩罚漏报和多报 |
| 字段绑定 | `TSR-path` | path 和 value 均严格正确的 GT 字段比例 |
| region | `R-F1@0.5` | corrected 主 region 指标；统一坐标和类型后按 IoU 0.5 一对一匹配 |
| 精细定位 | `R-F1@0.75` | 更严格的 bbox 定位诊断，不替代 `R-F1@0.5` |
| 局部分组 | `LIG-F1` | corrected line-item-group 定位 F1，只在有 GT 的页面上计算 |
| local grid | `LG-GriTS-Top` | 固定 corrected R-F1 parent mapping 后的 page-macro topology score |
| widget group | `WG-F1` | strict type/state/IoU member matching 后的两层 group F1 |
| relation | `Rel-F1` | 固定 endpoint mapping 后的 page-macro directed typed-edge F1 |

不要放进正式主表的列：

- `VAcc`：与 `Value-nED` 高度重复，而且不惩罚额外预测值；只作旧口径诊断。
- `WAcc`：只适用于 widget subset；需要时另做专项表，并同时报告适用分母。
- `CDS`：旧 CDS 包含未规范化的旧 `R-F1/LIG-F1`，不得作为最新模型排名。
- `Page-EM-valid / Schema-nTED-valid / Value-nED-valid / R-F1-valid / LIG-F1-valid`：这些列排除了缺失或无效预测，只能做覆盖率诊断。
- `outputs/main_exp/main_results.csv` 中的 `R-F1/LIG-F1/CDS`：属于 legacy 结构口径，不得抄入最新表。

### 0.3 为什么 structure 指标已经重算

不同模型的旧预测混用了原图 pixel、`0-1000` 和 `0-1` bbox；旧 evaluator 直接做 IoU，并且要求预测 type 与 metadata data type 字符串完全相同。最新结构评测使用 `configs/bbox_coordinate_spaces.json` 的 run-level 显式坐标空间，统一转换为 `[0,1]`，再将 GT/预测映射到共同 region ontology。转换会 clip 越界框并丢弃 malformed、反向和零面积框，同时记录每个 run 的 `clipped/dropped` 数量。原始 `outputs/main_exp/pred` 不会被改写。

不同 adapter 的 region 输出上限仍不完全相同，因此 corrected R-F1 衡量的是“当前完整系统实际输出”，不是脱离 prompt 的纯模型检测上限。解释低分时同时查看 `R-Precision@0.5` 和 `R-Recall@0.5`：compact 输出通常首先表现为 recall 较低。

### 0.4 prediction 更新后如何一键重建

只有 prediction 或 GT/layout metadata 更新后才运行以下命令：

```bash
cd .
PY=.venv/bin/python

$PY -m formtsr_exp.page_em_report --workers 8
$PY -m formtsr_exp.document_similarity_report --workers 16
$PY -m formtsr_exp.structure_metrics_report --workers 4
$PY -m formtsr_exp.hierarchical_metrics_report --workers 4
$PY -m formtsr_exp.final_metrics_report
```

如果加入新 run，必须先在 `configs/bbox_coordinate_spaces.json` 中写明该 run 的 `pixel / normalized_1000 / normalized_1 / none`，并给出审计证据。不得按单页最大坐标自动猜，也不得把反向 bbox 的端点交换后继续评分。

### 0.5 本科生交付清单

1. 从 `main_experiment_results.csv` 生成一张 9-model 主表，不自行 join 或补数。
2. 表注写清：所有分数范围 `[0,1]`、越高越好；缺失/无效页进入主分母并记 0。
3. `Page-EM` 同时给 exact page count；所有模型都必须给 `Valid/7000`。
4. 正文分别讨论 schema、value、path binding 和 spatial structure，不用一个 composite score 代替分层分析。
5. 至少核对 `Qwen3.6` 的 `3/7000` exact pages、`R-F1@0.5=0.059498`、`LIG-F1=0.076500`；对不上说明使用了旧文件。
6. 将最终 CSV、Markdown/LaTeX 表和一页结果解读一起交付；解读中不要把相关性描述成因果关系。

目前可以直接写入结果分析的事实：Qwen3.6 在 `Schema-nTED / Value-nED / TSR-path / R-F1@0.5 / LIG-F1` 上均为最高；Qwen3.5 的语义指标接近 Qwen3.6，但结构定位较弱；GLM 的 `R-F1@0.5=0.055899` 接近 Qwen3.6，而语义分数更低；PaddleOCR 的 `R-F1@0.75=0.016526` 最高但 `TSR-path=0`，说明精细 OCR/layout 定位不能替代字段路径理解；Step3 的 `Value-nED=0.594786` 与 `TSR-path=0.0004` 反差表明其主要问题是 schema/path binding。

## 1. 总体原则

1. 难度分层和约束切片直接复用 `outputs/main_exp` 中已有预测；受控诊断验证只使用 test gold annotation。三者都不需要启动 GPU 模型。
2. 视觉鲁棒性只推理退化图片。clean baseline 必须复用主实验的 clean 预测，不要再跑一遍 clean。
3. 所有辅助实验写到独立目录。不要覆盖或手工编辑 `outputs/main_exp/main_results.csv`、`outputs/main_exp/per_sample_metrics.jsonl` 或第 0 节的最终 reporting 文件。
4. 除第 3 节最新难度报告会直接读取正式主表外，其余旧辅助命令必须显式传 `--models`。legacy metrics 文件包含 smoke、失败任务和 `_aligned_metadata` 诊断行；不筛模型会把这些行一起统计。
5. 官方结果默认使用原始模型输出。名字含 `_aligned_metadata` 的结果只能作为诊断，不得替代 raw 结果，除非负责人明确要求。
6. `NA` 表示该样本或模型不适用该指标，不得改成 0。CSV 和 LaTeX 表都由脚本生成，不要人工补数字。
7. 本地推理一次只跑一个模型。当前 vLLM/SGLang 配置通常会占用两张 GPU，不要并行启动两个双卡模型。
8. 不要升级 `.venv`、`.venv-vllm`、CUDA、PyTorch、vLLM 或 SGLang。环境问题先记录完整报错并联系负责人。

## 2. 运行前检查

进入项目目录，并固定后续命令使用的路径：

```bash
cd .

PY=.venv/bin/python
CONFIG=configs/main_experiment.yaml
INDEX=outputs/main_exp/dataset_index.jsonl
METRICS=outputs/main_exp/per_sample_metrics.jsonl
PRED_ROOT=outputs/main_exp/pred
```

确认必要文件存在：

```bash
test -f "$CONFIG"
test -f "$INDEX"
test -f "$METRICS"
test -d "$PRED_ROOT"
test -d FormTSR/datasets
test -d newdataset-layout
test -f outputs/domain_stats/normal_calibrated_difficulty_sample_levels.csv
```

视觉鲁棒性还需要：

```bash
test -d FormTSR/dataset-augment
nvidia-smi
```

先查看现有模型行，确认 run id、样本数和 invalid rate：

```bash
head -n 40 outputs/main_exp/main_results.csv
head -n 30 outputs/robustness_exp/degraded/main_results.csv
```

模型筛选使用 CSV 第一列 `model`，不是第二列 `model_id`。第 3 节不使用下面的手工列表，而是自动读取 `main_experiment_results.csv` 中 9 个最佳全量 raw run。第 4/5 节的 legacy 命令仍需显式筛选；aligned、低覆盖和旧后端只能用于单独诊断。

```bash
MAIN_MODELS="caprl_internvl3_5_8b_vllm_vlm_aligned_metadata,deepseek_ocr2_sglang_vlm,deepseek_vl2_vllm_vlm,gemma4_26b_hf_vlm,glm4_6v_flash_sglang_vlm_aligned_metadata,gpt_vlm,kimi_vl_a3b_vllm_vlm,Qwen3.6-35B-A3B_aligned_metadata,mineru2_5_pro_vllm_engine_vlm,paddleocr_vl_1_6_pipeline_sglang,qwen3_5_9b_sglang_vlm_aligned_metadata,step3_vl_10b_vllm_vlm,unlimited_ocr_hf_vlm"
ROBUST_MODELS="Qwen3.6-35B-A3B_vllm_vlm,caprl_internvl3_5_8b_vllm_vlm,deepseek_vl2_vllm_vlm,glm4_6v_flash_vllm_vlm,kimi_vl_a3b_vllm_vlm,qwen3_5_9b_vllm_vlm,step3_vl_10b_vllm_vlm"
```

完整选择及覆盖率记录在 `outputs/aux_exp/latex/model_selection.csv`。不要同时加入同一实际模型的 raw、aligned、smoke 和失败后端行。

clean 和 degraded 的 run id 可以不同。鲁棒性报告会先按 `(model, clean_sample_id)` 配对，找不到时再按 `(model_id, clean_sample_id)` 配对。

当前已有以下可复用产物，运行前不要删除：

```text
outputs/main_exp/per_sample_metrics.jsonl
outputs/main_exp/pred/{model}/
outputs/robustness_exp/degraded/main_results.csv
outputs/robustness_exp/degraded/pred/{model}/
outputs/robustness_exp/report_latest/
```

先根据 `main_results.csv` 判断负责人指定的模型是否已经覆盖全部样本。已经完成的模型直接重新生成报告；只有缺预测的模型才需要启动推理。

## 2.1 最早版、legacy 主结果与最新 reporting 指标

最早一版实验设计使用 `TSR / R-F1 / LG / WG-F1 / Rel-F1 / CDS`。随后生成的 legacy `main_results.csv` 使用 `TSR-path / VAcc / R-F1 / LIG-F1 / WAcc / CDS`。坐标审计又发现 legacy structure evaluator 混用了不同 bbox 坐标空间，因此最新正式 reporting 指标以第 0 节为准：

```text
Valid/Total / Page-EM / Schema-nTED / Value-nED / TSR-path / corrected R-F1 / corrected LIG-F1 / LG-GriTS-Top / WG-F1 / Rel-F1
```

具体变化如下：

| 最早版 | 当前版 | 变化原因与当前定义 |
|---|---|---|
| `TSR` | `TSR-path` + `VAcc` | 将字段路径正确性和值识别能力拆开，避免模型识别出值但字段命名不同而全部记错。 |
| `R-F1` | corrected `R-F1@0.5/0.75` | run-level 坐标统一到 `[0,1]`，GT/预测 type 映射到共同 ontology 后做一对一匹配。0.5 是主列，0.75 是严格定位诊断。 |
| `LG` | corrected `LIG-F1` + `LG-GriTS-Top` | LIG 继续评估 line-item-group bbox；local grid 使用 raw cell span topology、固定 R-F1 parent mapping 和 GriTS-Top。 |
| `WG-F1` | raw-metadata `WG-F1` | raw Label Studio parent edge 恢复 group membership，实例 `answer.json` 恢复 state；成员和 group 分别做两层 matching。legacy `WAcc` 仅保留作历史诊断。 |
| `Rel-F1` | fixed-endpoint `Rel-F1` | 使用 raw directed GT edge 和显式预测 relation；endpoint mapping 在评分前冻结。主表报告 page macro，micro/per-type/matched-endpoint 放附表。 |
| 四项等权 `CDS` | 主表删除 | legacy CDS 使用旧 structure 分数且动态排除 `NA`，只保留作历史诊断。 |

当前指标的精确定义：

- `TSR-path`: 保留完整 GT leaf path，逐字段精确比较；分母是所有 GT leaf 字段数。
- `VAcc`: 忽略字段路径，对非空 GT leaf value 做规范化 multiset matching；空 GT value 不进入分母，重复值按出现次数计。
- `R-F1`: 使用 `corrected_structure_metrics.csv` 中经过坐标和 type 规范化的值，主阈值为 0.5。
- `LIG-F1`: 使用同一 corrected structure report 中的 line-item-group 值。
- `LG-GriTS-Top`: 固定 corrected R-F1 parent mapping 后，对 local grid 做 GriTS-Top 和 page-level Hungarian matching。
- `WG-F1`: strict widget type/state/IoU member matching 后，再做 group-level Hungarian matching。
- `Rel-F1`: 固定 region/widget/field/LIG/grid/cell endpoint mapping 后评分 typed directed triples。
- `WAcc`: legacy widget answer-path 诊断，不替代当前 `WG-F1`。
- `CDS`: 仅解释 legacy/辅助实验，不进入最新正式主表。

不要把最早版表头和当前版数值放在同一张表中。如果需要引用旧实验，必须标注 `legacy metric definition`；论文当前表统一使用新版指标。

## 2.2 用已有预测重算 legacy 辅助指标

本节的 `formtsr_exp.evaluate` 仍服务于旧 difficulty/constraint/ablation/robustness 流程。它生成的 `TSR-path/VAcc/WAcc/CDS` 可以供这些历史辅助报告继续使用，但其中 `R-F1/LIG-F1/CDS` 不能回填第 0 节的最新主表。最新主表只按第 0.4 节的四条命令重建。

重新计算 legacy 辅助指标不需要调用模型，但必须有以下三类输入：dataset index、`pred/{model}/{sample_id}.json` 和 label/layout metadata。不能只用旧的 `main_results.csv` 推导逐页结果。

对主实验已有 prediction 重算当前指标，并写入独立目录：

```bash
CURRENT_EVAL=outputs/aux_exp/current_metric_reeval

$PY -m formtsr_exp.evaluate \
  --index "$INDEX" \
  --pred-root "$PRED_ROOT" \
  --out "$CURRENT_EVAL" \
  --config "$CONFIG" \
  --models "$MAIN_MODELS" \
  --skip-extra-reports
```

关键输出：

```text
outputs/aux_exp/current_metric_reeval/per_sample_metrics.jsonl
outputs/aux_exp/current_metric_reeval/per_model_metrics/{model}.jsonl
outputs/aux_exp/current_metric_reeval/main_results.csv
outputs/aux_exp/current_metric_reeval/main_results_table.tex
outputs/aux_exp/current_metric_reeval/main_results_metadata.json
```

`evaluate` 会对 index 中每个样本查找 prediction；缺失 prediction 会保留为 `missing_prediction`，所以必须检查 `n_total / n_valid_json / invalid_rate`。需要难度、约束和消融表时，优先使用第 3-5 节的独立命令读取这个新生成的 `per_sample_metrics.jsonl`，不要把重评结果覆盖回 `outputs/main_exp`。

对已有退化 prediction 重算当前指标：

```bash
ROBUST_REEVAL=outputs/robustness_exp/degraded_current_metrics

$PY -m formtsr_exp.evaluate \
  --index outputs/robustness_exp/robustness_degraded_index.jsonl \
  --pred-root outputs/robustness_exp/degraded/pred \
  --out "$ROBUST_REEVAL" \
  --config "$CONFIG" \
  --models "$ROBUST_MODELS" \
  --skip-extra-reports
```

最新视觉退化报告不读取 legacy CDS 汇总，而是直接组合 clean/degraded prediction、语义逐页结果和 corrected structure 逐页结果。legacy CDS 权重仍以 `configs/main_experiment.yaml` 为准，仅供重现旧表；不得将该 CDS 写入最新主表或视觉退化正式表。

## 3. 难度分层实验

本节和第 6 节使用最新正式指标。第 4-5 节仍有 legacy 辅助逻辑，其中的旧 `R-F1/LIG-F1/CDS` 不得回填本节或第 0 节主表。

### 做法

按模板的校准难度将样本分为：

- `L1`: easy
- `L2`: medium
- `L3`: hard
- `L4`: expert

难度是模板级的结构与上下文复杂度，不是单页视觉质量。正式映射冻结在 `outputs/domain_stats/normal_calibrated_difficulty_sample_levels.csv` 的 `normal_calibrated_level`：70 个模板分别为 L1=11、L2=24、L3=24、L4=11，每个正式模型对应 1100/2400/2400/1100 页。当前 `newdataset-layout` 已包含额外模板，因此不得从该目录动态重算阈值或回退分级；生成器会要求 mapping 的模板集合与主 index 完全一致，并将 SHA-256 写入 metadata。

主要报告：

- 只保留 `main_experiment_results.csv` 中 9 个最佳全量 raw run，共 36 行。
- 每个模型在 L1-L4 上报告 `Page-EM / Schema-nTED / Value-nED / TSR-path / R-F1@0.5 / R-F1@0.75 / LIG-F1`。
- `Page-EM` 同时报告 exact page count；`LIG-F1` 同时报告各级 GT-applicable 分母，当前固定为 200/600/800/800。
- `L1_to_L4_drop = metric(L1) - metric(L4)`。
- 正 drop 表示从 easy 到 expert 性能下降。

### 命令

```bash
$PY -m formtsr_exp.difficulty_metrics_report \
  --main-results outputs/main_exp/main_experiment_results.csv \
  --index "$INDEX" \
  --pred-root "$PRED_ROOT" \
  --difficulty-csv outputs/domain_stats/normal_calibrated_difficulty_sample_levels.csv \
  --out outputs/aux_exp/difficulty \
  --workers 4
```

### 检查输出

```text
outputs/aux_exp/difficulty/difficulty_results.csv
outputs/aux_exp/difficulty/difficulty_results.md
outputs/aux_exp/difficulty/difficulty_results_table.tex
outputs/aux_exp/difficulty/difficulty_diagnostic_summary.csv
outputs/aux_exp/difficulty/difficulty_diagnostic_summary_table.tex
outputs/aux_exp/difficulty/difficulty_results_metadata.json
```

检查 `difficulty_results_metadata.json`：

- `n_models=9`、`n_result_rows=36`。
- `sample_counts_by_level` 必须为 1100/2400/2400/1100，`template_counts_by_level` 必须为 11/24/24/11。
- `main_result_reconstruction` 中每个模型都应还原正式主表的 total、valid、exact 和所有指标；生成器发现不一致会直接报错。
- mapping SHA-256 应为 `4c446da7c43909d088cc9d5de71c4621dd388ff3b47739cb4bc5ea7318921a62`。
- index SHA-256 应为 `d7fcca1ea45453e8acf93476b6007cd175b8987cfae9bf91123ab1fae9247489`。

## 4. 约束切片实验

### 做法

从 `newdataset-layout` 的模板 metadata 中构造结构约束切片，并计算：

```text
Delta(c) = mean(metric | constraint absent) - mean(metric | constraint present)
```

正 `Delta(c)` 表示约束存在时模型性能更低。当前约束包括：

- `region_local_grids`: 存在局部表格或网格结构。
- `widget_grouping`: option、多值或 widget 分组信号较强。
- `key_field_relations`: key-field relation 较密集。
- `line_item_groups`: 存在 line-item group。
- `mixed_layout`: 表格区域与非表格区域混合。
- `visual_degradation`: 仅在索引包含退化样本时有效。

`weak_borderless_grids` 当前标注不可靠，脚本默认忽略。用 clean 主实验 metrics 跑切片时，`visual_degradation` 没有正样本，结果应为 `NA/insufficient_contrast`；视觉退化结论必须使用第 6 节的鲁棒性报告。

### 命令

```bash
$PY -m formtsr_exp.constraint_report \
  --metrics "$METRICS" \
  --index "$INDEX" \
  --config "$CONFIG" \
  --layout-root newdataset-layout \
  --models "$MAIN_MODELS" \
  --out outputs/aux_exp/constraints
```

### 检查输出

```text
outputs/aux_exp/constraints/constraint_slice_results.csv
outputs/aux_exp/constraints/constraint_slice_results_table.tex
outputs/aux_exp/constraints/constraint_slice_template_membership.csv
outputs/aux_exp/constraints/constraint_slice_metadata.json
```

检查 `constraint_slice_metadata.json` 中的 `missing_templates`、每个约束的 with/without 数量、q75 threshold 和 `ignored_constraints`。不要根据结果重新调整 q75 threshold。

## 5. 受控错误注入与诊断指标验证

### 做法与边界

该实验不使用模型预测。它只读取 11 个 fully reviewed test templates 的 1,100 条 gold annotation，把 gold 作为完美预测，再分别注入 value、hierarchy、region、line-item、local-grid、widget 和 relation 错误。正式强度为 10%/25%/50%，固定 seeds 为 `0,1,2,3,4`；同一 seed 的三个强度使用 nested unit selection。

运行前脚本强制执行 gold identity check。八项指标在各自 applicable pages 上必须全部为 100，否则直接终止。当前 applicable pages 为：

```text
Schema/Value/TSR/R/Rel = 1100
LIG = 400, LG = 300, WG = 1000
```

每个 corrupted output 只改变目标结构属性。line-item 的 primitive item text/bbox 保持不变，membership 调换后才重建 LIG evaluator 使用的 group envelope。region 框只在对所有 type-compatible GT 框均满足 `IoU < 0.5` 时进入可注入集合。relation 保持 endpoint pair 和 edge count 不变。

drop 定义为：

```text
Delta_pp = 100 * (clean gold score - corrupted score)
```

先逐页计算 paired drop，再在每个 template 内对页面和五个 seeds 求均值，最后对 applicable templates 做 macro-average。95% CI 使用 10,000 次 template-clustered bootstrap；不能把同一模板的 100 页作为独立 cluster。正文使用 25% response matrix、target-vs-max-unrelated selectivity 和 target monotonicity；完整 severity curves 放附录。

### 命令

```bash
$PY -m formtsr_exp.controlled_diagnostic_validation --workers 4

# paired scores 未变化时只重建 bootstrap、图和 LaTeX：
$PY -m formtsr_exp.controlled_diagnostic_validation --report-only --workers 1
```

### 检查输出

```text
outputs/aux_exp/controlled_diagnostic/gold_identity_check.csv
outputs/aux_exp/controlled_diagnostic/paired_page_drops.csv
outputs/aux_exp/controlled_diagnostic/injection_rates.csv
outputs/aux_exp/controlled_diagnostic/severity_curves.csv
outputs/aux_exp/controlled_diagnostic/metric_response_matrix_25pct.csv
outputs/aux_exp/controlled_diagnostic/diagnostic_selectivity_25pct.csv
outputs/aux_exp/controlled_diagnostic/controlled_diagnostic_figure.pdf
outputs/aux_exp/controlled_diagnostic/controlled_diagnostic_severity_curves.pdf
outputs/aux_exp/controlled_diagnostic/controlled_diagnostic_table.tex
outputs/aux_exp/controlled_diagnostic/controlled_diagnostic_metadata.json
```

正式输出应有 91,500 条 paired page-condition 记录、105 条 seed-level injection-rate 记录、168 条 metric severity curve，以及 7 行 response/selectivity summary。所有正式下降都报告 percentage points，不报告 relative drop。`controlled_diagnostic_metadata.json` 必须记录 test index hash、固定 seeds、bootstrap 协议和 relation endpoint reachability audit。

## 6. 视觉退化鲁棒性实验

### 做法

退化数据位于：

```text
FormTSR/dataset-augment/{template}/{instance}/{variant}/{level}/
```

当前包含五种退化：

- `blur_noise`
- `dilate`
- `erode`
- `occlusion_stain`
- `perspective_skew`

每种退化包含 `low / medium / high` 三档。具体操作参数以每个目录下的 `augment_meta.json` 为准，不要根据文件名自行猜参数。

最新报告同时按以下维度分层：

```text
model x degradation_variant x degradation_level x difficulty_level
```

每层报告 clean/degraded 均值、绝对 drop 和 relative drop，指标为 `Page-EM / Schema-nTED / Value-nED / TSR-path / corrected R-F1@0.5 / R-F1@0.75 / corrected LIG-F1`。完成全部 1,020 次尝试的 7 个模型统一进入正式汇总；当前报告假定 clean/degraded 运行之间的 backend 差异可忽略。

`blur_noise / erode / occlusion_stain` 不改变页面坐标，可以报告 corrected spatial metrics。`dilate` 含局部 warp，`perspective_skew` 含全局仿射和透视变换；当前没有同步变换后的 bbox GT，因此这两类的 `R-F1@0.5 / R-F1@0.75 / LIG-F1` 必须为 `NA`。

旧 CDS 已退出正式指标，因此旧 `CDS failure boundary` 也不进入最新报告，不能把旧阈值套到新指标上。

### 6.1 先跑 smoke test

只取 2 个 clean base sample、`blur_noise/low`：

```bash
SMOKE_ROOT=outputs/robustness_exp/smoke

$PY -m formtsr_exp.build_robustness_index \
  --clean-data-root FormTSR/datasets \
  --augment-root FormTSR/dataset-augment \
  --out-root "$SMOKE_ROOT" \
  --variants blur_noise \
  --levels low \
  --limit-base 2
```

选择一个配置中存在的模型 run id：

```bash
MODEL=qwen3_5_9b_vllm_vlm

$PY -m formtsr_exp.run_main \
  --config "$CONFIG" \
  --index "$SMOKE_ROOT/robustness_degraded_index.jsonl" \
  --out-dir "$SMOKE_ROOT/degraded" \
  --models "$MODEL" \
  --resume \
  --rerun-invalid \
  --skip-extra-reports
```

生成 smoke 报告：

```bash
$PY -m formtsr_exp.robustness_report \
  --clean-main-metrics "$METRICS" \
  --degraded-metrics "$SMOKE_ROOT/degraded/per_sample_metrics.jsonl" \
  --clean-index "$SMOKE_ROOT/robustness_clean_index.jsonl" \
  --degraded-index "$SMOKE_ROOT/robustness_degraded_index.jsonl" \
  --difficulty-csv outputs/domain_stats/normal_calibrated_difficulty_sample_levels.csv \
  --layout-root newdataset-layout \
  --models "$MODEL" \
  --out "$SMOKE_ROOT/report"
```

smoke 验收：raw、pred、per-sample metrics、report metadata 都存在；单条失败不能导致整个任务退出。

### 6.2 构建全量鲁棒性索引

```bash
$PY -m formtsr_exp.build_robustness_index \
  --clean-data-root FormTSR/datasets \
  --augment-root FormTSR/dataset-augment \
  --out-root outputs/robustness_exp
```

当前数据快照应得到 68 个 clean base sample 和 1020 个 degraded sample，即：

```text
68 x 5 variants x 3 levels = 1020
```

必须查看 `outputs/robustness_exp/robustness_index_metadata.json`，确认 `by_variant_level`、`skipped` 和实际样本数。以后数据更新时以 metadata 为准，不要硬编码 1020。

### 6.3 按模型顺序运行退化图

先检查已有鲁棒性结果：

```bash
head -n 30 outputs/robustness_exp/degraded/main_results.csv
```

当前索引下，完整模型的 `n_total` 应为 1020。若模型已经有 1020 行，不要从头重跑；通常可以直接进入第 6.4 节生成报告。确实缺预测时，一次只运行一个模型：

```bash
MODEL=qwen3_5_9b_vllm_vlm

$PY -m formtsr_exp.run_main \
  --config "$CONFIG" \
  --index outputs/robustness_exp/robustness_degraded_index.jsonl \
  --out-dir outputs/robustness_exp/degraded \
  --models "$MODEL" \
  --resume \
  --skip-extra-reports
```

完成后修改 `MODEL`，再运行下一个。`--resume` 会复用已存在的 prediction 和已有错误记录。先检查 `errors/{model}.jsonl`，只有确认失败属于网络、服务中断等暂时性问题后，才在同一命令中增加 `--rerun-invalid`。不要删除已有 `pred/{model}` 目录来“重新开始”。

长任务在 `tmux` 或 `screen` 中运行。重新连接后先检查进程和 GPU，再决定是否续跑：

```bash
ps -ef
nvidia-smi
```

如果负责人要求使用批处理脚本，必须显式传模型列表，不能直接使用脚本内默认值。该脚本会自动加 `--rerun-invalid`，所以运行前必须先确认确实需要重试失败样本：

```bash
MODELS="$ROBUST_MODELS" bash scripts/run_robustness_successful_models.sh
```

### 6.4 生成分层报告

```bash
$PY -m formtsr_exp.structure_metrics_report \
  --index outputs/robustness_exp/robustness_degraded_index.jsonl \
  --pred-root outputs/robustness_exp/degraded/pred \
  --main-results outputs/robustness_exp/degraded/main_results.csv \
  --bbox-manifest configs/bbox_coordinate_spaces.json \
  --models "$ROBUST_MODELS" \
  --workers 7 \
  --out outputs/robustness_exp/latest_metrics/degraded

$PY -m formtsr_exp.robustness_metrics_report \
  --clean-results outputs/main_exp/main_experiment_results.csv \
  --degraded-results outputs/robustness_exp/degraded/main_results.csv \
  --clean-index outputs/robustness_exp/robustness_clean_index.jsonl \
  --degraded-index outputs/robustness_exp/robustness_degraded_index.jsonl \
  --out outputs/robustness_exp/report_latest

$PY -m formtsr_exp.robustness_component_report \
  --pairs outputs/robustness_exp/report_latest/visual_degradation_per_sample.jsonl \
  --clean-index outputs/robustness_exp/robustness_clean_index.jsonl \
  --main-index outputs/main_exp/dataset_index.jsonl \
  --layout-root newdataset-layout \
  --out outputs/robustness_exp/report_latest
```

关键输出：

```text
outputs/robustness_exp/report_latest/visual_degradation_results.csv
outputs/robustness_exp/report_latest/visual_degradation_model_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_variant_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_by_difficulty.csv
outputs/robustness_exp/report_latest/visual_degradation_per_sample.jsonl
outputs/robustness_exp/report_latest/visual_degradation_gt_mismatches.csv
outputs/robustness_exp/report_latest/visual_degradation_results_metadata.json
outputs/robustness_exp/report_latest/visual_degradation_component_membership.csv
outputs/robustness_exp/report_latest/visual_degradation_by_component.csv
outputs/robustness_exp/report_latest/visual_degradation_component_contrast.csv
outputs/robustness_exp/report_latest/visual_degradation_component_condition_macro.csv
outputs/robustness_exp/report_latest/visual_degradation_component_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_component_excess_drop_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_component_results.md
outputs/robustness_exp/report_latest/visual_degradation_component_metadata.json
```

完整输出应满足：

- 7 个全量尝试模型、7,140 个 clean/degraded pair。
- 统一条件表 `7 x 5 x 3 = 105` 行，variant × severity 宏平均表 15 行。
- difficulty 细分表 `7 x 5 x 3 x 4 = 420` 行。
- `dilate / perspective_skew` 的三个 spatial metrics 均为 `NA`。
- component membership 为 `68 x 5 = 340` 行；统一 component 条件表为 `7 x 5 x 3 x 5 x 2 = 1,050` 行，present-vs-absent 对照为 525 行。
- component severity 和 present-vs-absent excess-drop 表各 15 行。

检查 metadata 中：

- clean/degraded index SHA-256、`n_pairs=7140`、`n_condition_rows=105` 和 `n_difficulty_rows=420`。
- `n_models=7`，且 `aggregation_policy` 明确记录 7 个完整模型统一汇总。
- clean baseline 是否只取 68 页 robustness 子集，而不是完整 7,000 页均值。
- `en_13__01` 的 15 份 augmented label 一致包含 clean label 漏掉的两个可见字段；报告器将该一致标签作为双方共享 GT，并写入 `visual_degradation_gt_mismatches.csv`。

component 标签必须由 clean instance 的 `template_name` 连接到对应模板 metadata，不能从 prediction、退化图或文件名反推。正式主索引中的 70 个模板用于冻结 widget/relation 的 q75 阈值，再选出 robustness 的 68 个模板。当前 component present 数为 local-grid 30、widget 32、dense relation 18、LIG 18、mixed-layout 10；五个切片可以重叠，不能相加当作互斥分区。

## 7. metadata alignment 诊断

该步骤只用于分析“模型识别出了值，但字段路径命名与 GT 不一致”的情况。它使用模板 metadata 中的 label/bbox 和模型预测文本做匹配，不读取 GT answer value 进行匹配。

示例：

```bash
RAW_MODEL=caprl_internvl3_5_8b_vllm_vlm
ALIGNED_MODEL=caprl_internvl3_5_8b_vllm_vlm_aligned_metadata_diagnostic
ALIGN_ROOT=outputs/aux_exp/alignment

$PY -m formtsr_exp.align_predictions \
  --index "$INDEX" \
  --pred-in "$PRED_ROOT/$RAW_MODEL" \
  --pred-out "$ALIGN_ROOT/pred/$ALIGNED_MODEL" \
  --layout-root newdataset-layout \
  --report "$ALIGN_ROOT/${ALIGNED_MODEL}.json" \
  --workers 8

$PY -m formtsr_exp.evaluate \
  --index "$INDEX" \
  --pred-root "$ALIGN_ROOT/pred" \
  --out "$ALIGN_ROOT/eval" \
  --models "$ALIGNED_MODEL" \
  --skip-extra-reports
```

报告时必须同时保留 raw 和 aligned 两行，并明确写 `post-hoc metadata alignment diagnostic`。aligned 结果不能静默替换原始结果。

## 8. 结果交付清单

每次实验结束后提交以下信息：

1. 使用的模型 run id 和实际 `model_id`。
2. 完整运行命令、配置文件路径和输出目录。
3. `n_total / n_valid_json / invalid_rate`。
4. 对应实验的 metadata JSON。
5. CSV 主表和 LaTeX 草稿。
6. 错误文件路径及失败样本数；不要只汇报成功样本。
7. 是否使用 raw prediction、aligned diagnostic 或 OCR pipeline 后处理。
8. 任何偏离本文档固定阈值、模型列表或数据索引的地方。

建议最终目录结构：

```text
outputs/aux_exp/
  difficulty/
  constraints/
  structure_ablation/
  alignment/

outputs/robustness_exp/
  robustness_clean_index.jsonl
  robustness_degraded_index.jsonl
  robustness_index_metadata.json
  degraded/
  latest_metrics/
  report_latest/
  smoke/
```

## 9. 常见问题

### 输出中出现 smoke 或失败模型

原因通常是漏传 `--models`。改用新的辅助输出目录，或备份错误目录后再处理，然后使用负责人确认的 run id 重新生成报告；不要修改主实验 metrics。

### `n_total` 不是 7000

先查看 `outputs/main_exp/main_results.csv`。该 run 可能是 smoke、未完成任务或缺预测，不能直接作为完整模型结果。

### 结构指标大部分为 `NA` 或很低

先确认 `--layout-root newdataset-layout` 正确，并检查模型预测是否真的输出 bbox、regions、widgets 或 relations。没有对应 GT/预测结构时保留 `NA`，不能改成 0 或人工补标。

### 鲁棒性报告找不到 clean pairing

检查 degraded metrics 的 `model_id` 是否与 clean 主实验中的实际模型一致，再检查 `clean_sample_id` 是否存在于 clean index。不要通过复制别的模型结果来补 pairing。

### invalid JSON 较多

查看 `outputs/robustness_exp/degraded/errors/{model}.jsonl` 和 raw response。确认服务正常后用 `--resume --rerun-invalid` 续跑，不要清空正常 prediction。

### GPU OOM 或服务启动失败

保存完整日志和 `nvidia-smi` 输出。未经负责人确认，不要先降低 `max_tokens`，因为会截断结构 JSON；通常应先降低 concurrency、batch size 或 `max_num_seqs`。

### clean constraint report 中 `visual_degradation=NA`

这是预期行为。clean 主实验没有退化正样本；视觉退化 drop 由 `robustness_metrics_report` 在配对 clean/degraded 样本上计算。旧 CDS failure boundary 不进入最新报告。

## 10. 当前实现限制

- instance-level `answer.json` 主要是语义 key-value tree；region/LIG GT 使用 `newdataset-layout`，grid/widget/relation GT 优先使用 raw `new-dataset-json` metadata。
- `R-F1` 需要 region bbox/type；`LIG-F1`、`LG-GriTS-Top` 和 `WG-F1` 分别只在对应 GT 可观测的页面上定义。
- 主实验报告 fixed-endpoint page-macro `Rel-F1`；由于多数历史模型没有输出可匹配的显式 relation，当前完整 run 的该列均为 0，需结合 micro/per-type 和 matched-endpoint 附表解释。
- 当前 `FormTSR/dataset-augment` 是 68 个 clean base sample 的鲁棒性子集，不是 7000 样本全量退化集。
- case study 尚未实现。
- 所有未运行或不适用结果必须保留为 `TBD` 或 `NA`，不得编造。
