# FormTSR-Bench Main Experiment Pipeline

This implements the main experiment pipeline and the visual degradation robustness pipeline. Case studies are not implemented.

## Data

Default data root:

```bash
./FormTSR/datasets
```

Expected valid sample layout:

```text
FormTSR/datasets/{template_name}/{instance_id}/{template_name}-{instance_id}.png
FormTSR/datasets/{template_name}/{instance_id}/answer.json
```

The indexer only includes samples where both the PNG and `answer.json` exist.

## Commands

Build the index and schema summary:

```bash
python -m formtsr_exp.build_index --data-root ./FormTSR/datasets --out outputs/main_exp/dataset_index.jsonl
```

Small smoke test:

```bash
python -m formtsr_exp.run_main --config configs/main_experiment.yaml --limit 3 --models gpt_vlm
```

The default config uses the full extraction prompt. `compact_structure: true` is only for local connectivity/debug checks and should not be used for reported test runs, because it caps the number of structural regions/widgets.

Full main experiment:

```bash
python -m formtsr_exp.run_main --config configs/main_experiment.yaml
```

Re-evaluate existing predictions:

```bash
python -m formtsr_exp.evaluate --index outputs/main_exp/dataset_index.jsonl --pred-root outputs/main_exp/pred --out outputs/main_exp
```

Rebuild the corrected region/LIG metrics, hierarchical metrics, and formal 9-model tables without rerunning inference:

```bash
python -m formtsr_exp.structure_metrics_report --workers 4
python -m formtsr_exp.hierarchical_metrics_report --workers 4
python -m formtsr_exp.final_metrics_report
```

Build visual-degradation robustness indexes from `FormTSR/dataset-augment`:

```bash
python -m formtsr_exp.build_robustness_index \
  --clean-data-root ./FormTSR/datasets \
  --augment-root ./FormTSR/dataset-augment \
  --out-root outputs/robustness_exp
```

Run only the degraded samples in an isolated output directory. Clean baselines are read from the already completed main experiment metrics by matching `clean_sample_id`; `outputs/main_exp` is not modified.

```bash
python -m formtsr_exp.run_main \
  --config configs/main_experiment.yaml \
  --index outputs/robustness_exp/robustness_degraded_index.jsonl \
  --out-dir outputs/robustness_exp/degraded \
  --models Qwen3.6-35B-A3B \
  --resume \
  --skip-extra-reports
```

Recompute corrected degraded structure metrics, then build the latest clean/degraded report:

```bash
python -m formtsr_exp.structure_metrics_report \
  --index outputs/robustness_exp/robustness_degraded_index.jsonl \
  --pred-root outputs/robustness_exp/degraded/pred \
  --main-results outputs/robustness_exp/degraded/main_results.csv \
  --bbox-manifest configs/bbox_coordinate_spaces.json \
  --models Qwen3.6-35B-A3B_vllm_vlm,caprl_internvl3_5_8b_vllm_vlm,deepseek_vl2_vllm_vlm,glm4_6v_flash_vllm_vlm,kimi_vl_a3b_vllm_vlm,qwen3_5_9b_vllm_vlm,step3_vl_10b_vllm_vlm \
  --workers 7 \
  --out outputs/robustness_exp/latest_metrics/degraded

python -m formtsr_exp.robustness_metrics_report \
  --out outputs/robustness_exp/report_latest
```

Post-hoc metadata field alignment diagnostic:

```bash
python -m formtsr_exp.align_predictions \
  --index outputs/main_exp/dataset_index.jsonl \
  --pred-in outputs/main_exp/pred/caprl_internvl3_5_8b_vllm_vlm \
  --pred-out outputs/main_exp/pred/caprl_internvl3_5_8b_vllm_vlm_aligned_metadata \
  --layout-root newdataset-layout \
  --report outputs/main_exp/diagnostics/caprl_aligned_metadata.json \
  --workers 8

python -m formtsr_exp.evaluate \
  --index outputs/main_exp/dataset_index.jsonl \
  --pred-root outputs/main_exp/pred \
  --out outputs/main_exp \
  --models caprl_internvl3_5_8b_vllm_vlm_aligned_metadata
```

This diagnostic uses template metadata labels/bboxes and prediction text to place extracted values under GT field paths. It does not use GT answer values for matching, and should be reported separately from raw model output.

Build the formal L1-L4 table for the best fully attempted raw run of each model, using the latest metrics:

```bash
.venv/bin/python -m formtsr_exp.difficulty_metrics_report \
  --main-results outputs/main_exp/main_experiment_results.csv \
  --index outputs/main_exp/dataset_index.jsonl \
  --pred-root outputs/main_exp/pred \
  --out outputs/aux_exp/difficulty \
  --workers 4
```

Build constraint-sliced Delta(c) tables from existing per-sample metrics:

```bash
python -m formtsr_exp.constraint_report --metrics outputs/main_exp/per_sample_metrics.jsonl --index outputs/main_exp/dataset_index.jsonl --out outputs/main_exp
```

Build the latest evaluation-component ablation tables from the nine formal raw runs:

```bash
python -m formtsr_exp.structure_ablation_metrics_report \
  --selected-main outputs/main_exp/main_experiment_results.csv \
  --corrected-dir outputs/main_exp/corrected_structure_per_sample \
  --out outputs/aux_exp/structure_ablation/report_latest \
  --workers 4
```

`run_main` and `evaluate` still write legacy difficulty, constraint, and structural-ablation outputs automatically. Regenerate the formal difficulty table with `difficulty_metrics_report` and the formal component ablation with `structure_ablation_metrics_report`; do not substitute the automatic legacy R-F1/LIG-F1/CDS tables.

## Outputs

```text
outputs/main_exp/dataset_index.jsonl
outputs/main_exp/schema_summary.json
outputs/main_exp/prompt.txt
outputs/main_exp/raw/{model}/{sample_id}.txt
outputs/main_exp/pred/{model}/{sample_id}.json
outputs/main_exp/errors/{model}.jsonl
outputs/main_exp/per_sample_metrics.jsonl
outputs/main_exp/main_results.csv
outputs/main_exp/main_results_table.tex
outputs/main_exp/main_results_metadata.json
outputs/main_exp/corrected_structure_metrics.csv
outputs/main_exp/corrected_structure_per_sample/{model}.jsonl
outputs/main_exp/hierarchical_structure_metrics.csv
outputs/main_exp/hierarchical_relation_type_metrics.csv
outputs/main_exp/hierarchical_structure_metrics_metadata.json
outputs/main_exp/hierarchical_structure_per_sample/{model}.jsonl
outputs/main_exp/main_experiment_results.csv
outputs/main_exp/final_reporting_metrics_table.tex
outputs/aux_exp/difficulty/difficulty_results.csv
outputs/aux_exp/difficulty/difficulty_results.md
outputs/aux_exp/difficulty/difficulty_results_table.tex
outputs/aux_exp/difficulty/difficulty_diagnostic_summary.csv
outputs/aux_exp/difficulty/difficulty_diagnostic_summary_table.tex
outputs/aux_exp/difficulty/difficulty_results_metadata.json
outputs/aux_exp/structure_ablation/report_latest/ablation_targeted_deltas.csv
outputs/aux_exp/structure_ablation/report_latest/ablation_targeted_macro.csv
outputs/aux_exp/structure_ablation/report_latest/ablation_results.md
outputs/aux_exp/structure_ablation/report_latest/ablation_results_table.tex
outputs/aux_exp/structure_ablation/report_latest/ablation_results_metadata.json
outputs/main_exp/constraint_slice_results.csv
outputs/main_exp/constraint_slice_results_table.tex
outputs/main_exp/constraint_slice_template_membership.csv
outputs/main_exp/constraint_slice_metadata.json
outputs/main_exp/structure_ablation_components.csv
outputs/main_exp/structure_ablation_variants.csv
outputs/main_exp/structure_ablation_deltas.csv
outputs/main_exp/structure_ablation_deltas_table.tex
outputs/main_exp/structure_ablation_targeted_deltas.csv
outputs/main_exp/structure_ablation_targeted_deltas_table.tex
outputs/main_exp/structure_ablation_metadata.json
```

Visual robustness outputs are separate:

```text
outputs/robustness_exp/robustness_clean_index.jsonl
outputs/robustness_exp/robustness_degraded_index.jsonl
outputs/robustness_exp/robustness_index_metadata.json
outputs/robustness_exp/degraded/raw/{model}/{sample_id}.txt
outputs/robustness_exp/degraded/pred/{model}/{sample_id}.json
outputs/robustness_exp/degraded/per_sample_metrics.jsonl
outputs/robustness_exp/latest_metrics/degraded/corrected_structure_per_sample/{model}.jsonl
outputs/robustness_exp/report_latest/visual_degradation_results.csv
outputs/robustness_exp/report_latest/visual_degradation_model_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_by_difficulty.csv
outputs/robustness_exp/report_latest/visual_degradation_variant_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_per_sample.jsonl
outputs/robustness_exp/report_latest/visual_degradation_results_metadata.json
outputs/robustness_exp/report_latest/visual_degradation_component_membership.csv
outputs/robustness_exp/report_latest/visual_degradation_by_component.csv
outputs/robustness_exp/report_latest/visual_degradation_component_contrast.csv
outputs/robustness_exp/report_latest/visual_degradation_component_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_component_excess_drop_severity.csv
outputs/robustness_exp/report_latest/visual_degradation_component_results.md
outputs/robustness_exp/report_latest/visual_degradation_component_metadata.json
```

`main_results.csv` keeps `model` as the run id used for output directories, and `model_id` as the actual backend model configured in YAML. Prefer using the actual model name as the run id, for example `model=Qwen3.6-35B-A3B`.

## Model Adapters

Configured in `configs/main_experiment.yaml`.

Supported provider names:

- `openai_vlm`, reads `OPENAI_API_KEY`
- `anthropic_vlm`, reads `ANTHROPIC_API_KEY`
- `gemini_vlm`, reads `GOOGLE_API_KEY`
- `local_hf_vlm`, runs a configured local command
- `local_sglang_server_vlm`, starts or calls a local SGLang-compatible VLM backend
- `local_vllm_server_vlm`, starts one local vLLM OpenAI-compatible server; supports vLLM internal data parallel load balancing
- `local_vllm_multi_server_vlm`, starts multiple independent local vLLM servers and distributes requests client-side
- `traditional_tsr`, runs a configured traditional TSR command

If credentials or SDKs are missing, the runner records the status in `outputs/main_exp/errors/{model}.jsonl` and continues.

### Local SGLang VLM

`Qwen3.6-35B-A3B` uses SGLang Offline Engine batch inference when `batch_command` is configured. The runner passes the selected samples to `scripts/sglang_vlm_offline_batch.py`; that script loads the model once, chunks requests by `batch_size`, and calls `Engine.generate(prompt=[...], image_data=[[...], ...])`. This is native SGLang batching, not a runner thread pool.

Run a smoke test with the configured local Qwen model:

```bash
python -m formtsr_exp.run_main --config configs/main_experiment.yaml --limit 3 --models Qwen3.6-35B-A3B
```

The older HTTP script `scripts/sglang_vlm_infer.py` is still available as a single-sample fallback when `command` is used without `batch_command`.

### Local vLLM VLM

`local_vllm_server_vlm` follows the vLLM internal load-balancing DP deployment pattern for single-node multi-GPU runs: one `vllm serve` process exposes one OpenAI-compatible endpoint, and vLLM dispatches work across DP ranks internally. For CapRL/InternVL3.5 the default config uses:

```yaml
provider: local_vllm_server_vlm
base_url: http://127.0.0.1:8000
tensor_parallel_size: 1
data_parallel_size: 2
data_parallel_size_local: 2
```

The configured `max_num_seqs` is per DP rank in vLLM. The older `local_vllm_multi_server_vlm` remains available for external client-side load balancing across multiple ports.

Resume the CapRL vLLM full run in a background shell/tmux/screen session:

```bash
bash scripts/run_caprl_vllm_loop.sh
```

#### Gemma4 Status

`gemma4_26b_vllm_vlm` is kept in the config as the intended vLLM server route, but it is not runnable on the current driver/runtime combination. The installed Gemma4-capable vLLM environment uses a newer CUDA runtime than the current NVIDIA 550.90.07 driver supports, and startup fails with `cudaGetDeviceCount failed: CUDA driver version is insufficient for CUDA runtime version`. A driver upgrade is required before using this vLLM route.

`gemma4_26b_hf_vlm` is the current working fallback. It runs Gemma4 through Transformers in the isolated CUDA 12.4 environment:

```bash
python -m formtsr_exp.run_main \
  --config configs/main_experiment.yaml \
  --limit 3 \
  --models gemma4_26b_hf_vlm \
  --resume \
  --skip-extra-reports
```

This path uses `AutoModelForMultimodalLM`, `device_map=auto`, and both A100 GPUs. It is functionally verified for smoke tests, but it is much slower than a vLLM/SGLang serving path and should not be treated as the preferred full-run engine unless runtime is acceptable.

### PaddleOCR-VL Pipeline

`paddleocr_vl_1_6_pipeline_sglang` uses PaddleOCR-VL's official pipeline with the VL recognizer served by SGLang. This is not a prompt-following FormTSR JSON model. The parser maps observed `parsing_res_list` blocks and non-duplicate `layout_det_res.boxes` into `regions`, and maps OCR text plus HTML table cells into `answer.ocr_lines`, `answer.table_cells`, and `answer.table_rows`. It does not infer missing FormTSR field paths from GT metadata, so `TSR-path` and `WAcc` should be interpreted as raw structured-output compatibility, while `VAcc` is the useful value-only signal for this OCR pipeline.

## Schema Mapping

Observed `FormTSR/datasets/*/*/answer.json` files are semantic key-value trees, usually without explicit region boxes, table cells, widget groups, or relation edges.

The pipeline therefore uses this mapping:

`run_main` / `evaluate` below describe the legacy per-sample pipeline used by existing auxiliary reports. Formal model reporting must read `outputs/main_exp/main_experiment_results.csv`: it retains only the best fully attempted raw run per model, and its R-F1/LIG-F1 columns come from `formtsr_exp.structure_metrics_report`, which converts each run's explicit bbox source space to normalized `[0,1]`, applies a canonical region-type mapping, and counts missing/invalid applicable pages as zero. The unnormalized R-F1/LIG-F1 and CDS in `main_results.csv` are not formal current results.

- `TSR-path`: strict field-level answer accuracy, computed as correct GT leaf fields divided by total GT leaf fields with the full GT path preserved. If a prediction has a top-level `answer` field, that field is compared to GT.
- `VAcc`: path-independent value accuracy over non-empty GT leaf values. It uses normalized multiset matching, so repeated values are counted correctly and blank GT leaves are ignored for this value-only metric.
- `R-F1`: uses explicit `regions` / `region_boxes` plus `bbox` and `type` when present. With `layout_root`, `newdataset-layout/{template}.json` is normalized from `fields/keys/value/values` bbox metadata into region GT.
- `LIG-F1`: line-item-group localization F1. With `layout_root`, GT comes from `metadata.layout_structure.sections[].line_item_groups`. Predictions can provide `line_item_groups`, regions with `type=line_item_group`, or local grid/cell unions. Templates without line-item-group GT remain `NA` and are excluded from the metric mean/CDS denominator.
- `WAcc`: widget answer accuracy. With `layout_root`, selectable fields are identified from checkbox/radio metadata and scored by comparing the corresponding answer values in prediction and GT.
- `CDS`: fixed weighted average of numeric `TSR-path`, `VAcc`, `R-F1`, `LIG-F1`, and `WAcc`; default weights are all `0.20` and are saved to result metadata. `NA` metrics are excluded from that sample's weighted denominator. Traditional TSR models report unsupported form-specific metrics and CDS as `NA`.

## Difficulty-Stratified Experiment

The formal difficulty experiment reports the latest metrics separately for calibrated L1-L4 sample groups: Page-EM, Schema-nTED, Value-nED, TSR-path, corrected R-F1@0.5, R-F1@0.75, and corrected LIG-F1. It includes only the best fully attempted raw run per model from `main_experiment_results.csv`.

- `L1`: easy
- `L2`: medium
- `L3`: hard
- `L4`: expert

Difficulty is a template-level structural/context measure, not an instance-level visual-quality label. The benchmark mapping is frozen in `outputs/domain_stats/normal_calibrated_difficulty_sample_levels.csv` using `normal_calibrated_level`. The reporter requires its template set to match the main dataset index exactly and records its SHA-256; it never recalibrates from the current `newdataset-layout` directory or falls back to layout metadata.

The current calibrated split contains 70 templates: L1=11, L2=24, L3=24, and L4=11. With 100 instances per template, a fully evaluated model has 1100/2400/2400/1100 samples across L1/L2/L3/L4.

`difficulty_results.csv` is the 9-model x 4-level long table. Missing/invalid predictions score zero in all full-scope metrics; LIG-F1 instead uses its GT-applicable pages, whose level counts are 200/600/800/800. `difficulty_diagnostic_summary.csv` pivots each metric into L1-L4 columns and adds `L1_to_L4_drop` plus `relative_drop_pct`. A positive drop means the model loses performance from easy to expert samples. The reporter also verifies that the weighted four-level rollup reproduces every model's formal main result.

## Constraint-Sliced Experiment

The constraint slicing experiment reports per-model performance under each structural or visual constraint and computes:

```text
Delta(c) = mean(metric | constraint absent) - mean(metric | constraint present)
```

A positive `Delta(c)` means performance drops when constraint `c` is present. The main table is `constraint_slice_results.csv`; the LaTeX table reports the CDS slice by default.

Default constraints:

- `region_local_grids`: local table/grid evidence from `metadata.S.cell_count/row_count/col_count` or `metadata.layout_structure.table_region_count`.
- `widget_grouping`: high widget/option grouping signal using the run's q75 threshold over option/multi-value grouping counts.
- `key_field_relations`: dense key-field relations using the run's q75 threshold over `metadata.S.relation_edge_count`.
- `line_item_groups`: explicit line-item group evidence from `metadata.S` or `metadata.layout_structure`.
- `mixed_layout`: table regions mixed with non-table regions/sections, or explicit multi-table context.
- `visual_degradation`: explicit visual degradation tags when available. If the evaluated set has no degraded samples, Delta is `NA` with `insufficient_contrast`.

`weak_borderless_grids` is intentionally ignored in the default report because the current template-level weak-grid annotations are known to be incorrect.

## Visual Degradation Robustness

The robustness experiment uses `FormTSR/dataset-augment`, whose per-variant samples are organized as:

```text
FormTSR/dataset-augment/{template_name}/{instance_id}/{variant}/{level}/{template_name}-{instance_id}.png
FormTSR/dataset-augment/{template_name}/{instance_id}/{variant}/{level}/answer.json
FormTSR/dataset-augment/{template_name}/{instance_id}/{variant}/{level}/augment_meta.json
```

Current variants are `blur_noise`, `dilate`, `erode`, `perspective_skew`, and `occlusion_stain`; levels are `low`, `medium`, and `high`. The index builder pairs each degraded image with the clean sample under `FormTSR/datasets` and writes a unique degraded sample id:

```text
{template_name}__{instance_id}__deg__{variant}__{level}
```

Degraded predictions are written under `outputs/robustness_exp/degraded`. The latest reporter is `formtsr_exp.robustness_metrics_report`; it scores the paired 68-page clean subset and every degraded condition with Page-EM, Schema-nTED, Value-nED, TSR-path, corrected R-F1@0.5/R-F1@0.75, and corrected LIG-F1. Missing or invalid predictions remain in each condition's denominator and score zero. A positive drop is `clean - degraded`.

Seven model ids have a fully attempted 1,020-sample degraded run and all seven enter the unified formal tables and macros. Backend differences between retained clean and degraded runs are treated as negligible for this aggregation. MinerU and PaddleOCR have no complete degraded run and are excluded.

Spatial scores are valid only when the transform preserves page geometry. `blur_noise`, `erode`, and `occlusion_stain` use the corrected spatial metrics. `dilate` includes a local warp and `perspective_skew` includes global affine/perspective transforms, but transformed bbox GT is unavailable; their R-F1@0.5, R-F1@0.75, and LIG-F1 values are therefore `NA` in the formal report.

Clean and degraded predictions are scored against the same semantic GT. For `en_13__01`, all 15 augmented labels unanimously contain two visible fields omitted by the clean label, so that augmented label is used as a shared robustness-only override for both sides and recorded in `visual_degradation_gt_mismatches.csv`. TSR-path is recomputed against this shared GT rather than copied from legacy per-sample rows.

The current robustness index has 68 clean samples and 1020 degraded samples, formed as 68 clean samples x 5 degradation variants x 3 levels. Within this subset the calibrated clean difficulty split is L1=11, L2=22, L3=24, and L4=11. These counts are recorded in `visual_degradation_results_metadata.json` and the CSV row counts; do not mix them with the full 7000-sample main experiment.

The main outputs are `report_latest/visual_degradation_results.csv` (105 unified condition rows), `visual_degradation_by_difficulty.csv` (420 rows), and `visual_degradation_variant_severity.csv` (the seven-model macro). Legacy CDS and its failure-boundary threshold are not carried into the latest report. The reporter does not modify `outputs/main_exp`.

Build component-level robustness slices after the paired report:

```bash
python -m formtsr_exp.robustness_component_report
```

The component reporter joins each clean instance's `template_name` to its template metadata. The 70 templates in the formal main index freeze the widget/relation q75 thresholds before the 68 robustness templates are selected. Labels are not inferred from predictions or degraded filenames. The five overlapping slices are region-local grids, widget grouping, dense key-field relations, line-item groups, and mixed layout. Component tables and macros use all seven completed model pairs.

## Evaluation-Component Ablation

This repository has no independent model-architecture, prompt, or training-module ablation run. The latest report is a metric-sensitivity analysis that holds predictions fixed and changes the evaluation components. Its semantic baseline is the equal-weight mean of `Schema-nTED`, `Value-nED`, and `TSR-path`; Page-EM is excluded because exact matches are too sparse.

The report adds corrected `R-F1@0.5`, substitutes `R-F1@0.75` as a strict localization diagnostic, compares global-grid topology with corrected local-grid/LIG localization, and adds metadata-derived `WAcc` or explicit-only `Rel-F1`. Missing or invalid predictions score zero whenever the GT component is applicable. Target scopes are derived from GT/template metadata and therefore have the same denominator for all nine formal raw runs.

`ablation_targeted_deltas.csv` reports `Delta = score(with) - score(without)`. Negative values mean the added or stricter component scores lower than the simpler configuration; they are partly definition-driven and must not be described as a causal reduction in model capability. Use `ablation_targeted_macro.csv` for the nine-model summary and `ablation_results_table.tex` for the paper draft. Root-level legacy `structure_ablation_*.csv` files still use VAcc and uncorrected spatial metrics and are not formal results.

## Current Limitations

- Instance labels do not include explicit structural annotations; structure metrics use template-level `newdataset-layout` metadata when available. R-F1 requires structural boxes; LIG-F1 is only defined for templates with line-item-group metadata; WAcc uses answer values for metadata-selected widget fields. Rel-F1 was removed because it largely duplicates answer-tree TSR.
- The current `FormTSR/dataset-augment` directory is a robustness subset, not the 7000-sample full test set. The index metadata records the exact number of paired clean and degraded samples used for a run.
- No experiment result is fabricated. Missing or not-run models remain `TBD` or `NA`.
