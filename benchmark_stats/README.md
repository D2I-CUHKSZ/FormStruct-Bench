# Benchmark Data Statistics

## Template-level statistics

Generate structural statistics for only the templates present in the main
dataset and generate the matching language/script/direction distribution:

```bash
.venv/bin/python -m benchmark_stats.run_hierarchical_structure_stats \
  --input newdataset-layout \
  --template-root FormTSR/datasets \
  --selection-counts config/template_selection_fields_70.csv \
  --output outputs/structure_stats

.venv/bin/python -m benchmark_stats.run_template_language_stats \
  --template-root FormTSR/datasets \
  --output outputs/template_stats
```

## Instance-level statistics

Generate instance-level semantic-structure, content, diversity, duplicate,
clean-image visual, and cross-modal statistics:

```bash
.venv/bin/python -m benchmark_stats.run_instance_level_data_stats \
  --data-root FormTSR/datasets \
  --output outputs/instance_stats \
  --workers 4
```

Use `--skip-visual` for answer-only statistics. Use `--limit N` for a smoke
test without changing the dataset.

The main paper-ready outputs are:

```text
instance_level_summary_table.tex
instance_content_structure_summary_table.tex
instance_content_structure_summary.csv
clean_visual_summary_table.tex
instance_constraint_tag_distribution_table.tex
instance_structure_coverage_table.tex
template_diversity_summary_table.tex
duplicate_summary_table.tex
exact_duplicate_answer_clusters.csv
exact_duplicate_image_clusters.csv
data_quality_summary_table.tex
cross_modal_correlations_table.tex
instance_feature_distributions.{png,pdf}
template_diversity.{png,pdf}
clean_visual_feature_distributions.{png,pdf}
cross_modal_correlation_heatmap.{png,pdf}
```

The corresponding CSV and JSON files contain the unrounded values and the
method metadata records all thresholds and feature definitions. Instance tags
are computed relative to the mode or median of their own template. Cross-modal
correlations include a template-centered coefficient to separate within-template
variation from fixed layout differences.

## Real-template representativeness against SRFUND

Download the supplied SRFUND ZIP without modifying it, retain the archive at
`raw/srfund/srfund_download.bin`, and extract its `dataset/` directory below
`raw/srfund/extracted/`. Then run the template/layout-cluster analysis:

```bash
.venv/bin/python -m benchmark_stats.run_representativeness_analysis \
  --srfund-root raw/srfund/extracted/dataset \
  --srfund-archive raw/srfund/srfund_download.bin \
  --layout-dir newdataset-layout \
  --formstruct-image-dir new-dataset \
  --template-root FormTSR/datasets \
  --source-metadata-dir metadata-test \
  --split-assignments outputs/dataset_splits/template_stratified_seed42/template_assignments.csv \
  --bootstrap-rounds 1000 \
  --seed 42 \
  --output outputs/representativeness
```

Install the dedicated dependencies from `requirements-representativeness.txt`.
The command clusters SRFUND pages before inference, excludes confirmed and
suspected FormStruct source overlap, runs unstratified and stratified Real--Real
calibration, and generates all CSV, PDF, LaTeX, manifest, validation, and final
report artifacts in one output directory.
The report treats shared-language comparisons as the primary structural view,
keeps language-composition results separate, and includes a direct-only Gower
sensitivity alongside the mapped direct+conditional configuration. SRFUND pages
are clustered before statistics; the manifest records the candidate-retrieval
limits and any source metadata that was unavailable.

## Latest four-level difficulty statistics

Recompute v0.4.1 `D_main = S_form + C_context` over all current layout
templates and calibrate L1--L4 at the normal-CDF percentiles 15.8655, 50, and
84.1345:

```bash
.venv/bin/python -m benchmark_stats.run_latest_four_level_difficulty_stats \
  --layout-dir newdataset-layout \
  --output outputs/difficulty_stats/latest_four_level
```

This command is read-only with respect to layout metadata. It writes template
assignments, level/language/domain distributions, calibration metadata, a
paper-ready LaTeX table, and PNG/PDF plots to the output directory.
