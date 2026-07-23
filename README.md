# FormStruct-Bench

FormStruct-Bench is a multilingual benchmark and evaluation toolkit for
hierarchical form understanding. The code evaluates document vision-language
models on semantic answer extraction, schema recovery, region localization,
line-item grouping, widget understanding, and visual-degradation robustness.

This repository contains code and configuration only. Dataset images,
annotations, model weights, predictions, and generated reports are distributed
or stored separately.

## Repository Layout

```text
formtsr_exp/       Main inference, parsing, metrics, and reporting pipeline
benchmark_stats/  Dataset statistics and representativeness analysis
srfund_exp/        SRFUND transfer and aligned-evaluation utilities
scripts/           Model adapters, launchers, SFT, and reporting helpers
peft/              FormStruct SFT dataset construction
configs/           Benchmark and model configuration examples
config/            Difficulty, selection, and training metadata
```

The release intentionally excludes raw data, generated outputs, caches,
checkpoints, model weights, local virtual environments, and legacy RAG/few-shot
experiments.

## Dataset Layout

Place or link the separately downloaded dataset under `data/FormStruct-Bench`:

```text
data/FormStruct-Bench/
  datasets/
    {template_name}/
      {instance_id}/
        {template_name}-{instance_id}.png
        answer.json
        answer.md
        answer.html
  dataset-augment/
  template_annotation/
```

The official benchmark scope is defined by the 70 templates in `datasets/`,
with 100 instances per template and 7,000 pages in total. The annotation folder
contains 80 JSON files; 10 are redundant metadata and must be excluded from
official statistics, splits, training scope, and evaluation. Join annotations
to the canonical dataset by template name.

The official template-disjoint split is fixed with seed 42: 49 training
templates (4,900 pages), 10 validation templates (1,000 pages), and 11 test
templates (1,100 pages). Portable assignments and sample indices are versioned
under `splits/template_stratified_seed42/`; do not create a page-random split.

## Installation

Create an isolated environment and install the core evaluation dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For dataset analysis and plots:

```bash
python -m pip install -r requirements-analysis.txt
```

Model backends such as vLLM, SGLang, PaddleOCR, PyTorch, and Transformers are
hardware and CUDA dependent. Install them in dedicated environments using the
versions appropriate for the selected model. `requirements-inference.txt`
lists the backend-independent Python packages used by API and Hugging Face
adapters; it intentionally does not pin vLLM or SGLang.

## Quick Start

Build the canonical dataset index:

```bash
python -m formtsr_exp.build_index \
  --data-root data/FormStruct-Bench/datasets \
  --out outputs/main_exp/dataset_index.jsonl
```

Run a small API-model smoke test after exporting the corresponding credential:

```bash
export OPENAI_API_KEY="..."
python -m formtsr_exp.run_main \
  --config configs/main_experiment.yaml \
  --models gpt_vlm \
  --limit 3
```

Evaluate predictions already written under `outputs/main_exp/pred`:

```bash
python -m formtsr_exp.evaluate \
  --config configs/main_experiment.yaml \
  --index outputs/main_exp/dataset_index.jsonl \
  --pred-root outputs/main_exp/pred \
  --out outputs/main_exp
```

See `formtsr_exp/README_FormTSR_main_exp.md` for the full pipeline and
`Metrics_README.md` for metric definitions and aggregation rules.

## Configuration

`configs/main_experiment.yaml` is a portable, disabled-by-default starting
point. Credentials are read from environment variables and are never stored in
configuration files.

`configs/main_experiment.full.example.yaml` preserves the complete multi-model
experiment structure. Model, CUDA, executable, and output paths in that file
are examples and must be adapted to the local environment. Other files in
`configs/` capture model-specific evaluation and transfer experiments.

Common credential variables are listed in `.env.example`. Do not commit a
populated `.env` file or embed tokens in YAML, Python, shell scripts, logs, or
generated output.

## License

This code is released under the Apache License 2.0. That license applies only
to this software repository and does not license the dataset images, source
form designs, answers, annotations, or derived augmentations. Dataset terms and
the per-template rights audit are maintained in the Hugging Face dataset
repository; the audit status must be checked before downloading or
redistributing data.
