# Controlled Diagnostic Validation 本科生运行手册

这份手册用于复现论文的受控错误注入实验。实验不调用模型、不使用 API、
不需要 GPU。它把 1,100 条测试集 gold annotation 当作完美预测，每次只注入
一种结构错误，用来验证八个指标是否能区分七类错误。

正式实验固定为：

- 数据：11 个 test templates，共 1,100 页；
- 错误：value、hierarchy、region、line-item、local-grid、widget、relation；
- 注入比例：10%、25%、50%；
- 随机种子：`0,1,2,3,4`；
- 统计：逐页 paired drop，先在模板内平均，再对 applicable templates 做
  macro-average；
- 置信区间：10,000 次 template-clustered bootstrap；
- 下降单位：absolute percentage points，不能改写成 relative drop。

不要使用 `outputs/main_exp/dataset_index.jsonl`。它是 7,000 页主实验索引，
不属于这个实验。

## 1. 进入仓库并固定 Python

```bash
cd .
PY=.venv/bin/python
export PYTHONPATH=.
```

后续所有命令都在仓库根目录执行。`PYTHONPATH=.` 不能省略，否则可能出现：

```text
ModuleNotFoundError: No module named 'formtsr_exp'
```

## 2. 运行前检查

### 2.1 检查测试索引

```bash
wc -l outputs/dataset_splits/template_stratified_seed42/test_index.jsonl
sha256sum outputs/dataset_splits/template_stratified_seed42/test_index.jsonl
```

正式输入必须满足：

```text
1100 outputs/dataset_splits/template_stratified_seed42/test_index.jsonl
e984fecd476335f785a1dbb79b44c9f4efc6613fcd2b36788f37de0485dcaa05
```

如果行数或 SHA-256 不同，先停止，不要继续生成论文结果。

### 2.2 检查三类输入

```bash
test -f FormTSR/datasets/Arabic-2/01/answer.json
test -f new-dataset-json/Arabic-2.json
test -f newdataset-layout/Arabic-2.json
```

三条命令都应静默成功。它们分别代表实例 answer、raw Label Studio
结构标注和 corrected layout 标注。

### 2.3 检查 Python 依赖和代码

```bash
$PY -m py_compile \
  formtsr_exp/controlled_diagnostic_validation.py \
  formtsr_exp/hierarchical_metrics_report.py

$PY -c "import numpy, scipy, matplotlib, apted, rapidfuzz; print('dependencies OK')"
```

然后运行相关回归测试：

```bash
$PY -m pytest -q \
  tests/test_controlled_diagnostic_validation.py \
  tests/test_hierarchical_metrics.py
```

当前预期为 `44 passed, 4 subtests passed`。测试数将来可能增加，但不能有
failed 或 error。

## 3. 正式运行

```bash
$PY -u -m formtsr_exp.controlled_diagnostic_validation --workers 4
```

这个命令已经固定正式 index、三个 severity、五个 seeds、bootstrap 次数和输出
目录。不要为了得到更好看的结果修改这些参数。

运行开始后应先看到：

```text
running gold-as-prediction identity check
gold identity check passed for all applicable metrics
```

随后会输出 `condition 1/105` 到 `condition 105/105`。最后一行应为：

```text
wrote controlled diagnostic report to outputs/aux_exp/controlled_diagnostic
```

CPU 核数较少时可以把 `--workers 4` 改成 `--workers 1`，这只影响速度，不
改变结果。不要同时启动两个正式进程写同一个输出目录。

## 4. 脚本实际做了什么

执行顺序如下：

1. 从 test index 加载 1,100 个实例和 11 个模板。
2. 将每页 gold answer、region、LIG、grid、widget 和 relation 组装成完美预测。
3. 检查八个 applicable metrics 是否全部为 100；任一失败就终止。
4. 对七类错误分别注入 10%/25%/50%，每档使用五个固定 seeds。
5. 每个 corrupted output 只和自己的 clean gold 配对。
6. 计算逐页 absolute drop，再做 template-macro 聚合。
7. 对模板 cluster 做 10,000 次 bootstrap。
8. 生成 Figure 7、Table 7、完整 severity curves 和审计 metadata。

七类错误的控制变量是：

| 错误 | 只改变什么 | 主要响应指标 |
| --- | --- | --- |
| Value | 替换等字符长度的字段值，path 不变 | Value-nED、TSR-path |
| Hierarchy | 在已有 parent 间移动完整 key/value pair | Schema-nTED、TSR-path |
| Region | 平移 box，使其对所有兼容 GT box 均 `IoU < 0.5` | R-F1；依赖项 LG/Rel |
| Line-item | 调换 primitive item membership；item text/box 不变 | LIG-F1 |
| Local-grid | 改 row/column/span；parent、cell ID/box 不变 | LG-GriTS |
| Widget | 改 state；group/member 数量不变 | WG-F1 |
| Relation | 改 direction/type；endpoint pair 和边数不变 | Rel-F1 |

Line-item 扰动会在 membership 调换后重建 evaluator 使用的 group envelope。
primitive item 的文本和框没有变化。

## 5. 输出文件

所有结果位于：

```text
outputs/aux_exp/controlled_diagnostic/
```

### 5.1 Figure 7

```text
controlled_diagnostic_figure.pdf
controlled_diagnostic_figure.png
```

Figure 7(a) 的原始矩阵：

```text
metric_response_matrix_25pct.csv
```

Figure 7(b) 的原始数据和 bootstrap CI：

```text
diagnostic_selectivity_25pct.csv
```

不要从 PNG 反抄数字；论文数字必须来自这两个 CSV。

### 5.2 Table 7

```text
controlled_diagnostic_table.tex
target_metric_summary.csv
```

LaTeX 是自动生成表，CSV 是可审计的数值来源。

### 5.3 附录完整曲线

```text
controlled_diagnostic_severity_curves.pdf
controlled_diagnostic_severity_curves.png
severity_curves.csv
```

`severity_curves.csv` 共 168 条数据行，包含每种 error、severity、metric 的
applicable pages/templates、mean drop 和 template-clustered 95% CI。

### 5.4 最底层审计数据

```text
paired_page_drops.csv
injection_rates.csv
gold_identity_check.csv
controlled_diagnostic_metadata.json
README.md
```

- `paired_page_drops.csv`：91,500 条逐页条件记录，是所有聚合的唯一数值来源；
- `injection_rates.csv`：105 条 error/severity/seed 实际注入率；
- `gold_identity_check.csv`：八个指标的 clean identity audit；
- `controlled_diagnostic_metadata.json`：index hash、模板、seeds、bootstrap、
  applicable scope 和 relation reachability audit。

## 6. 结果验收

### 6.1 检查文件行数

```bash
wc -l outputs/aux_exp/controlled_diagnostic/*.csv
```

包含表头的正式行数应为：

```text
8      diagnostic_selectivity_25pct.csv
9      gold_identity_check.csv
106    injection_rates.csv
8      metric_response_matrix_25pct.csv
91501  paired_page_drops.csv
169    severity_curves.csv
9      target_metric_summary.csv
```

### 6.2 检查 gold identity

```bash
sed -n '1,12p' \
  outputs/aux_exp/controlled_diagnostic/gold_identity_check.csv
```

必须得到以下 applicable pages：

| Metric | Pages | Templates | Min/Mean/Max |
| --- | ---: | ---: | --- |
| Schema-nTED | 1,100 | 11 | 1/1/1 |
| Value-nED | 1,100 | 11 | 1/1/1 |
| TSR-path | 1,100 | 11 | 1/1/1 |
| R-F1@0.5 | 1,100 | 11 | 1/1/1 |
| LIG-F1 | 400 | 4 | 1/1/1 |
| LG-GriTS-Top | 300 | 3 | 1/1/1 |
| WG-F1 | 1,000 | 10 | 1/1/1 |
| Rel-F1 | 1,100 | 11 | 1/1/1 |

如果任何 clean score 不是 1，不要手工改 CSV，也不要继续写论文结论。保留报错
和当前 metadata，交给负责 evaluator 的同学检查。

### 6.3 对照当前正式结果

Table 7 当前应为：

| Error | Pages | Target | Drop@10% | Drop@25% | Drop@50% | rho | Max unrelated@25% |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Value | 1,100 | Value-nED | 9.0 | 21.5 | 41.7 | 1.00 | 0.0 |
| Hierarchy | 1,100 | Schema/TSR | 6.5/8.6 | 16.2/21.4 | 32.2/42.6 | 1.00/1.00 | 0.0 |
| Region | 1,100 | R-F1@0.5 | 10.0 | 24.9 | 49.9 | 1.00 | 0.0 |
| Line-item | 400 | LIG-F1 | 52.7 | 72.0 | 81.8 | 1.00 | 0.0 |
| Local-grid | 300 | LG-GriTS | 6.8 | 18.6 | 37.3 | 1.00 | 0.0 |
| Widget | 1,000 | WG-F1 | 9.8 | 24.9 | 49.8 | 1.00 | 0.0 |
| Relation | 1,100 | Rel-F1 | 10.1 | 25.1 | 50.1 | 1.00 | 0.0 |

Figure 7(a) 不是严格对角矩阵。25% 时以下 cross-response 是预期的：

- value corruption 使 `TSR-path` 下降 25.1 pp；
- region corruption 使 `LG-GriTS-Top` 下降 24.4 pp；
- region corruption 使 fixed-endpoint `Rel-F1` 下降 39.9 pp。

这些是路径、parent 和 endpoint matching 的层级依赖，不属于无关指标泄漏。

LIG-F1 的 drop 明显大于注入比例，是因为少量 membership 调换可能让重建后的
group envelope 越过 IoU=0.5 阈值。不要把“注入 25%”解释成“指标应该只下降
25 pp”，也不要为了压低 LIG 数值修改注入规则。

## 7. 只重建图表，不重跑扰动

如果 `paired_page_drops.csv` 已完整存在，而且你只修改了绘图、caption、bootstrap
或 LaTeX 格式，可以运行：

```bash
$PY -u -m formtsr_exp.controlled_diagnostic_validation \
  --report-only \
  --workers 1
```

这个命令会重新做 gold identity check，然后从已有 91,500 条 paired records 重建
CSV 汇总、Figure 7、Table 7、附录图和 metadata。它不会重新采样错误。

不要在 `paired_page_drops.csv` 不完整时使用 `--report-only`。

## 8. 论文文件

正文 Figure 7/Table 7 的 drop-in 片段：

```text
paper/controlled_diagnostic_validation.tex
```

附录实验协议和完整 severity curves 已写入：

```text
paper/appendix_experimental_details.tex
```

论文正文只报告 target response、最大无关响应和 monotonicity；完整曲线留在附录。

## 9. 常见问题

### 9.1 用成了 7,000 页索引

症状：脚本提示 formal scope 不是 1,100 pages/11 templates。

处理：使用默认命令，不要传 `outputs/main_exp/dataset_index.jsonl`。也不要加
`--allow-nonstandard-scope` 来绕过正式检查。

### 9.2 Gold identity check 失败

这表示 evaluator、GT reachability 或 annotation universe 有问题，不是正常实验
波动。停止运行并提交失败的 sample ID、metric、完整 traceback 和当前 git diff。

当前 relation audit 会排除 14 条 endpoint 不在任何冻结 prediction-side node
universe 中的 raw GT edge；各模板计数已写入 metadata。不要把这些不可达边重新
塞回 Rel-F1 分母。

### 9.3 进程中断

重新执行正式命令即可重建输出。脚本只写本实验目录，不会改模型预测。如果
`paired_page_drops.csv` 尚未写完，不要用 `--report-only` 冒充完整结果。

### 9.4 没生成 PDF/PNG

先检查：

```bash
$PY -c "import matplotlib; print(matplotlib.__version__)"
```

不要换系统 Python；使用仓库 `.venv`。依赖问题解决后可用 `--report-only` 重画。

### 9.5 实际注入率不是精确的 10/25/50

这是固定 seed 下对 eligible units 做确定性采样的正常波动。查看
`injection_rates.csv`；大样本指标应非常接近目标，只有 applicable unit 较少的
local-grid 波动更明显。不能按运行结果重新挑 seed。

### 9.6 输出没有出现在 `git status`

仓库默认忽略 `outputs/`，实验文件仍然真实存在。交付时打包整个
`outputs/aux_exp/controlled_diagnostic/`，或者按项目负责人的提交策略显式加入；
不要只交 PNG 而漏掉 CSV 和 metadata。

## 10. 交付清单

交付前逐项确认：

- [ ] test index 为 1,100 页、11 templates，SHA-256 正确；
- [ ] gold identity 八项均为 100；
- [ ] 105 个 conditions 全部完成；
- [ ] 三档 severity 和五个 fixed seeds 未修改；
- [ ] 所有 drop 使用 percentage points；
- [ ] applicable page 数与本手册一致；
- [ ] 95% CI 来自 10,000 次 template-clustered bootstrap；
- [ ] Figure 7(a)/(b) 的数字分别来自对应 CSV；
- [ ] Table 7 来自自动生成的 `.tex`；
- [ ] 附录 severity curves、paired source 和 metadata 一并交付；
- [ ] 没有把预期的 value/region 层级依赖写成 off-target leakage；
- [ ] 没有混入 7,000 页主实验或任意模型预测。
