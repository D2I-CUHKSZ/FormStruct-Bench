# FormTSR-Bench 当前指标与结果

本文档汇总当前 `formtsr_exp` 端到端表单理解主实验的指标、聚合规则和已有结果。这里的“当前”以仓库内现有产物为准：legacy 主结果生成于 2026-07-08，Page/document/corrected structure/final reporting 指标于 2026-07-15 重算，最新难度分层、评价组件消融、视觉退化及其 metadata component 切片于 2026-07-16 重算。

检索、GNN、RAG 和旧 PEFT 代码使用独立指标，不与本页的 FormTSR 主表混算，见[独立评测域](#独立评测域)。

## 一句话结论

- 当前正式主表使用 `Valid/Total / Page-EM / Schema-nTED / Value-nED / TSR-path / corrected R-F1@0.5 / R-F1@0.75 / corrected LIG-F1`。
- `VAcc / WAcc / CDS` 降为 legacy 或专项诊断；尤其 CDS 含未规范化的旧结构分数，不再用于最新模型排名。
- raw 结果中，Qwen3.6 的 `Schema-nTED=0.484589`、`Value-nED=0.669349`、`TSR-path=0.1791`、`R-F1@0.5=0.059498` 和 `LIG-F1=0.076500` 均最高。
- 多数模型的 `Value-nED` 明显高于 `TSR-path`，说明模型通常能读出部分值，但字段路径绑定和显式空间结构仍是主要瓶颈。
- `_aligned_metadata` 是后处理诊断结果，不是 raw 模型输出，不能与 raw 结果混在同一排名中。
- 当前主指标文件混有正式、smoke、失败和 aligned run。生成论文表时必须显式筛选 run，并同时报告 `Valid/7000`。

## 数据快照

当前主索引为 [`outputs/main_exp/dataset_index.jsonl`](outputs/main_exp/dataset_index.jsonl)：

| 项目 | 当前值 |
| --- | ---: |
| 模板数 | 70 |
| 实例数 | 7,000 |
| 可解析 `answer.json` | 7,000 |
| 缺失或无效图片 | 0 |
| 叶字段总数 | 256,806 |
| 空叶字段 | 0 |
| 难度模板分布 | L1=11, L2=24, L3=24, L4=11 |
| 难度实例分布 | L1=1,100, L2=2,400, L3=2,400, L4=1,100 |

数据质量和实例统计来自 [`outputs/instance_stats`](outputs/instance_stats)，模板结构统计来自 [`outputs/structure_stats`](outputs/structure_stats)。

## 当前主指标

### `TSR-path`

字段路径和值的严格正确率。将 GT 和预测递归展开为完整 leaf path，只有路径与值都相同才算命中：

```text
TSR-path = correct GT leaf fields / all GT leaf fields
```

字符串只做首尾去空白和连续空白合并，不做模糊匹配、大小写折叠或 Unicode NFKC。空 GT 字段也进入分母；额外预测字段不扣分。因此它衡量的是以 GT 为分母的严格字段召回，而不是同时惩罚多报字段的 precision-aware 指标。

### `VAcc`（legacy 诊断）

忽略字段路径，只比较非空叶值的规范化多重集：

```text
VAcc = matched non-empty GT values / all non-empty GT values
```

规范化包括 Unicode NFKC、转小写、合并空白，并移除数字之间的空格或逗号。重复值按出现次数匹配；额外预测值不扣分，因此 `VAcc` 更接近 path-independent value recall，不应解释为带 precision 惩罚的完整 accuracy。

### `R-F1`

corrected 区域检测 F1。旧预测分别使用 pixel、`0-1000` 或 `0-1` bbox；现在根据 [`configs/bbox_coordinate_spaces.json`](configs/bbox_coordinate_spaces.json) 的 run-level manifest 全部转换到 `[0,1]`。反向、零面积或 malformed bbox 不会交换端点修复，越界框会 clip 并写入审计计数。

GT 与预测 type 先映射到共同 ontology：`group / label / value / text / widget / table`。预测 region 与 GT region 必须同时满足：

1. canonical region type 兼容；
2. bbox IoU `>= 0.5`；
3. 在最大一对一匹配中配对。

```text
R-F1 = 2 * TP / (N_pred + N_gt)
```

主列是对完整 run scope 的逐页宏平均；缺失和无效页记 0。`R-F1@0.75` 使用相同口径和更严格的 IoU 阈值，作为精细定位诊断。报告还提供 micro `R-Precision/R-Recall/R-F1@0.5`，便于区分少报和误报。不同 adapter 的 region 输出上限仍不完全相同，因此该分数衡量当前完整系统输出，不应解释为脱离 prompt 的纯模型检测上限。

### `LIG-F1`

Line-item-group 定位 F1。预测和 GT bbox 使用与 R-F1 相同的坐标规范化，再在 IoU `>= 0.5` 时做最大一对一匹配。模型主分数对全部 GT-applicable 页面宏平均；适用页面上的缺失或无效预测记 0。没有 line-item-group GT 的页面不进入该指标分母。

该指标替代了旧版 `LG`。目前没有覆盖全数据且可靠的 cell row/column topology GT，因此主表不再将 cell topology 作为统一指标。

### `WAcc`（专项诊断）

只评估 metadata 标记为 checkbox/radio 等 selectable widget 的 answer path，并在相同路径上严格比较预测值与 GT 值：

```text
WAcc = correct selectable-widget answer fields / applicable widget fields
```

它不是 widget bbox 检测准确率。没有可映射 widget path 的样本为 `NA`；`WidgetBox-F1` 只在结构消融中作为诊断指标出现。

### `Page-EM`

Page Exact Match 将每页预测的顶层 `answer` 与该实例完整的 `answer.json` 比较。规范化后整棵树完全一致记 1，否则记 0：

```text
Page-EM = exact-match pages / all evaluated pages
```

dict 的 key 顺序被忽略，key 去除首尾空白；字符串去除首尾空白并合并连续空白；list 顺序保留，大小写和 Unicode 形式仍然敏感。缺失或无效预测记 0。`Page-EM-valid` 另以有效 JSON 为分母，仅作为覆盖率诊断。

当前实例 GT 只有 `answer.json`，因此 Page-EM 比较完整语义 answer tree，但不比较预测中的 `regions/widgets/local_grids/relations`。Page-EM 是独立报告项，当前不进入 CDS。

### `Schema-nTED`

Schema normalized Tree Edit Similarity 只比较 answer JSON 的 key、容器类型和层级，不比较字段值。schema tree 包含 root、key node 和有序 list-item node；node label 包含 key 名和 `object/array/scalar` 类型。

```text
Schema-nTED = 1 - APTED(T_pred, T_gt) / (|T_pred| + |T_gt|)
```

APTED 的 insert/delete cost 为 1，相同 label 的 rename cost 为 0，不同 label 为 2。dict key 先 strip 并排序，因此 JSON object 的原始键顺序不影响结果；list 顺序保留。分数范围 `[0,1]`，越高越好。

### `Value-nED`

Value normalized Edit Similarity 忽略字段路径，对预测和 GT 的非空叶值做最优一对一软匹配。单个值的相似度为：

```text
s(p, g) = 1 - Levenshtein(p, g) / max(len(p), len(g))

Value-nED = max_matching sum(s(p, g)) / max(|V_pred|, |V_gt|)
```

匹配使用 Hungarian assignment。值规范化与 VAcc 一致：Unicode NFKC、转小写、合并空白，并移除数字之间的空格或逗号。分母使用预测和 GT 值数量的较大者，所以漏报和多报都会扣分；这与只以 GT 为分母、不惩罚额外值的 VAcc 不同。

`Schema-nTED` 和 `Value-nED` 的主列都对完整 run scope 做逐页宏平均，缺失或无效页记 0；`*-valid` 只对有效 JSON 求平均，仅作覆盖率诊断。

### `CDS`（legacy）

CDS 是旧流程的逐样本复合分数。配置对五项 legacy 指标等权：

```text
w(TSR-path) = w(VAcc) = w(R-F1) = w(LIG-F1) = w(WAcc) = 0.20

CDS(sample) = sum(w_m * metric_m) / sum(w_m), m 为该样本可计算的指标
```

样本上的 `NA` 从分子和分母同时排除，剩余权重重新归一化。模型级 CDS 是先计算每个样本的 CDS，再对数值项做宏平均，因此不等于把主表中五个模型级均值再平均。

权重以 [`configs/main_experiment.yaml`](configs/main_experiment.yaml) 为准，实现以 [`formtsr_exp/metrics.py`](formtsr_exp/metrics.py) 为准。因为其中的 R-F1/LIG-F1 是未规范化的旧值，CDS 只用于解释历史辅助实验，不进入最新正式主表。

## 聚合与 `NA`

| 情况 | Page/Schema/Value/TSR | corrected R-F1 | corrected LIG-F1 |
| --- | --- | --- | --- |
| 预测有效且 GT 适用 | 正常计算 | 正常计算 | 正常计算 |
| GT 不包含该结构 | 正常计算 | 当前 70 个模板均有 region GT | 页面不进入 LIG 分母 |
| GT 有结构但预测为空 | 对应语义分数为 0 或按定义计算 | 0 | 0 |
| 缺预测、无效 JSON 或解析失败 | 0 | 0 | GT-applicable 页面记 0 |

所有正式主列都同时报告 `n_valid_json / n_total`。`*-valid` 排除缺失和无效页，只能作覆盖率诊断，不能替换主列。

`invalid_rate = 1 - n_valid_json / n_total`，范围也是 `[0, 1]`，但该项越低越好。

## 当前 raw 主结果

下表来自 [`outputs/main_exp/main_experiment_results.csv`](outputs/main_exp/main_experiment_results.csv)，只保留实际完成全部 7,000 次尝试的最佳 raw run，每个 `model_id` 一行。完整可读表和 LaTeX 表分别见 [`final_reporting_metrics.md`](outputs/main_exp/final_reporting_metrics.md) 和 [`final_reporting_metrics_table.tex`](outputs/main_exp/final_reporting_metrics_table.tex)。

| Model | Valid/7000 | Exact pages | Schema-nTED | Value-nED | TSR-path | R-F1@.5 | R-F1@.75 | LIG-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CapRL-InternVL3.5-8B | 6,933 | 0/7000 | 0.303448 | 0.570271 | 0.0001 | 0.010547 | 0.001591 | 0.000000 |
| DeepSeek-VL2 | 5,549 | 0/7000 | 0.058855 | 0.052943 | 0.0000 | 0.000000 | 0.000000 | 0.000104 |
| GLM-4.6V-Flash | 6,586 | 0/7000 | 0.304052 | 0.478803 | 0.0492 | 0.055899 | 0.009627 | 0.036596 |
| Kimi-VL-A3B-Instruct | 6,630 | 0/7000 | 0.160023 | 0.252678 | 0.0008 | 0.010906 | 0.000363 | 0.000000 |
| Qwen3.6-35B-A3B | 6,224 | 3/7000 | **0.484589** | **0.669349** | **0.1791** | **0.059498** | 0.014403 | **0.076500** |
| MinerU2.5-Pro | 7,000 | 0/7000 | 0.045259 | 0.109723 | 0.0000 | 0.000000 | 0.000000 | 0.000000 |
| PaddleOCR-VL-1.6 | 7,000 | 0/7000 | 0.029741 | 0.217337 | 0.0000 | 0.036788 | **0.016526** | 0.000000 |
| Qwen3.5-9B | 6,911 | 1/7000 | 0.455751 | 0.657279 | 0.1258 | 0.020258 | 0.002251 | 0.018393 |
| Step3-VL-10B | 6,841 | 0/7000 | 0.262000 | 0.594786 | 0.0004 | 0.005727 | 0.001785 | 0.010526 |

Qwen3.6 在语义、字段绑定、R-F1@0.5 和 LIG-F1 上均最高。PaddleOCR 的 R-F1@0.75 最高但 TSR-path 为 0，表明精细 OCR/layout 定位不能替代字段路径理解。Step3 的 Value-nED 达到 `0.594786`，但 TSR-path 只有 `0.0004`，主要瓶颈同样是 schema/path binding。

## 当前 Page-EM 结果

Page-EM 于 2026-07-15 从现有预测重新计算。主列分母使用 run 的全部评测页，缺失预测为 0；完整 24-run 表见 [`outputs/main_exp/page_em_results.csv`](outputs/main_exp/page_em_results.csv) 和 [`outputs/main_exp/page_em_results.md`](outputs/main_exp/page_em_results.md)。

覆盖至少 5,000 页的 raw run：

| Model | Valid/7000 | Exact pages | Page-EM | Page-EM-valid |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 6,224 | 3 | **0.000429** | 0.000482 |
| Qwen3.5-9B | 6,911 | 1 | **0.000143** | 0.000145 |
| CapRL-InternVL3.5-8B | 6,933 | 0 | 0.000000 | 0.000000 |
| DeepSeek-VL2 | 5,549 | 0 | 0.000000 | 0.000000 |
| GLM-4.6V-Flash | 6,586 | 0 | 0.000000 | 0.000000 |
| Kimi-VL-A3B-Instruct | 6,630 | 0 | 0.000000 | 0.000000 |
| MinerU2.5-Pro | 7,000 | 0 | 0.000000 | 0.000000 |
| PaddleOCR-VL-1.6 | 7,000 | 0 | 0.000000 | 0.000000 |
| Step3-VL-10B | 6,841 | 0 | 0.000000 | 0.000000 |

Metadata alignment 诊断 run：

| Model | Valid/7000 | Exact pages | Page-EM | Page-EM-valid |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B (aligned) | 6,911 | 59 | **0.008429** | 0.008537 |
| Qwen3.6-35B-A3B (aligned) | 6,224 | 48 | **0.006857** | 0.007712 |
| GLM-4.6V-Flash (aligned) | 6,586 | 26 | 0.003714 | 0.003948 |
| CapRL-InternVL3.5-8B (aligned) | 6,933 | 6 | 0.000857 | 0.000865 |

其余 11 个低覆盖、smoke 或失败 run 的 exact pages 均为 0：`caprl_internvl3_5_8b_sglang_vlm`、两个 CapRL smoke run、`deepseek_ocr2_sglang_vlm`、两个 Gemma run、`gpt_vlm`、`kimi_vl_a3b_sglang_vlm`、`mineru2_5_pro_hf_vlm`、`paddleocr_vl_1_6_hf_vlm` 和 `unlimited_ocr_hf_vlm`。

raw 模型合计只有 4 个整页命中，说明 Page-EM 比当前字段级指标严格得多。alignment 将四个模型的整页命中总数提高到 139，但最高 Page-EM 仍低于 1%，所以字段路径对齐只能解释部分整页误差。

## 当前 Schema-nTED / Value-nED 结果

两个指标于 2026-07-15 从现有预测重新计算。完整 24-run 结果见 [`outputs/main_exp/document_similarity_results.csv`](outputs/main_exp/document_similarity_results.csv) 和 [`outputs/main_exp/document_similarity_results.md`](outputs/main_exp/document_similarity_results.md)。

覆盖至少 5,000 页的可比 raw run，按 Schema-nTED 降序：

| Model | Valid/7000 | Schema-nTED | Schema-valid | Value-nED | Value-valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 6,224 | **0.484589** | 0.545006 | **0.669349** | 0.752802 |
| Qwen3.5-9B | 6,911 | **0.455751** | 0.461620 | **0.657279** | 0.665743 |
| GLM-4.6V-Flash | 6,586 | 0.304052 | 0.323165 | 0.478803 | 0.508901 |
| CapRL-InternVL3.5-8B | 6,933 | 0.303448 | 0.306381 | 0.570271 | 0.575782 |
| Step3-VL-10B | 6,841 | 0.262000 | 0.268090 | 0.594786 | 0.608610 |
| Kimi-VL-A3B-Instruct | 6,630 | 0.160023 | 0.168953 | 0.252678 | 0.266779 |
| DeepSeek-VL2 | 5,549 | 0.058855 | 0.074245 | 0.052943 | 0.066786 |
| MinerU2.5-Pro | 7,000 | 0.045259 | 0.045259 | 0.109723 | 0.109723 |
| PaddleOCR-VL-1.6 | 7,000 | 0.029741 | 0.029741 | 0.217337 | 0.217337 |

Qwen3.6 在两个 raw 软指标上均最高，Qwen3.5 接近。Step3 的 Schema-nTED 只有 `0.2620`，但 Value-nED 达到 `0.5948`，说明其主要问题是 schema/path，而不是所有内容都识别失败。PaddleOCR-VL 也呈现类似但更明显的 OCR-pipeline 特征：schema 很低，value 相对更高。

Metadata alignment 前后对比：

| Model | Raw Schema | Aligned Schema | Raw Value | Aligned Value |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 0.455751 | **0.736119** | 0.657279 | 0.586341 |
| CapRL-InternVL3.5-8B | 0.303448 | **0.689956** | 0.570271 | 0.568646 |
| Qwen3.6-35B-A3B | 0.484589 | **0.659673** | 0.669349 | 0.509338 |
| GLM-4.6V-Flash | 0.304052 | **0.612078** | 0.478803 | 0.463164 |

alignment 对四个模型的 schema 都有明显提升，但 Value-nED 没有同步上升。原因是 Value-nED 会对 alignment 后新增、重复或误放的非空值施加 precision penalty，而现有 VAcc 不惩罚额外预测值；因此两者回答的是不同问题。

## 指标相关性

相关性使用 9 个 `comparable_raw` run 的模型级汇总分数计算，不包含 aligned、smoke、failed 或低覆盖 run。完整 [Pearson 矩阵](outputs/main_exp/metric_correlations_model_level_pearson.csv)、[Spearman 矩阵](outputs/main_exp/metric_correlations_model_level_spearman.csv)和[全部指标对及 p-value](outputs/main_exp/metric_correlations_model_level_pairs.csv)均已保存。

| 指标对 | Pearson r | Spearman rho | 解释 |
| --- | ---: | ---: | --- |
| TSR-path / WAcc | 0.991 | 0.916 | WAcc 是 widget 字段上的严格 path/value 子集，明显重复 |
| VAcc / Value-nED | 0.974 | 0.967 | 都衡量 path-independent value，明显重复 |
| WAcc / Page-EM | 0.947 | 0.766 | Page-EM 稀疏，相关性不适合做稳定结论 |
| Schema-nTED / Value-nED | 0.933 | 0.867 | 好模型通常两项都好，但 alignment 实验表明两者可以反向变化 |
| WAcc / corrected LIG-F1 | 0.927 | 0.648 | 模型整体能力共同影响两项；排序一致性弱于线性相关 |
| TSR-path / Page-EM | 0.925 | 0.743 | Page-EM 只有两个 raw run 非零，相关性不稳定 |
| corrected R-F1@.5 / R-F1@.75 | 0.865 | 0.882 | 两个阈值测同一空间能力，但严格阈值仍会改变相对差距 |
| corrected R-F1 / LIG-F1 | 0.808 | 0.546 | 都依赖空间结构，但分别衡量通用 region 和 line-item grouping |
| TSR-path / Schema-nTED | 0.824 | 0.915 | 排名关系很强，但一个要求完整 path/value，一个只比较 schema |

corrected `R-F1@0.75` 与 Schema-nTED、Value-nED 的 Pearson 只有 `0.172/0.216`，是当前相对独立的精细定位维度。这里仅有 9 个模型，且 Page-EM/LIG-F1 很稀疏、模型覆盖率仍不完全一致；这些数字适合判断冗余，不应解释为样本级因果关系。

## Metadata alignment 诊断

alignment 使用模板 metadata 的 label/bbox 和预测文本重排字段路径，不使用 GT answer value 做匹配。它用于判断“识别出了值但字段路径不一致”造成了多少损失，不代表 raw 输出质量，也不能替代主结果。

| Model | Valid/7000 | Raw CDS | Aligned CDS | Aligned TSR-path | Aligned WAcc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 6,224 | 0.2462 | 0.4042 | 0.5481 | 0.5521 |
| Qwen3.5-9B | 6,911 | 0.2018 | 0.3547 | 0.4903 | 0.4031 |
| CapRL-InternVL3.5-8B | 6,933 | 0.1215 | 0.2738 | 0.3807 | 0.3054 |
| GLM-4.6V-Flash | 6,586 | 0.1270 | 0.2586 | 0.3466 | 0.3005 |

aligned CDS 的大幅上升说明路径对齐是核心误差来源之一，但 `R-F1` 和 `LIG-F1` 基本不变，显式空间结构生成仍然很弱。

## 辅助实验摘要

### 难度分层

最新难度表使用 9 个正式全量 raw run 和冻结的 70-template calibrated mapping。难度衡量模板级结构与上下文复杂度，不是实例视觉质量；L1/L2/L3/L4 分母固定为 1100/2400/2400/1100 页。下表是 9 个模型的 model-macro 均值，不是 pooled-page 分数：

| Metric | L1 | L2 | L3 | L4 | L1->L4 drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coverage | 0.962525 | 0.950278 | 0.950972 | 0.916970 | 0.045556 |
| Schema-nTED | 0.272491 | 0.213870 | 0.229137 | 0.248425 | 0.024065 |
| Value-nED | 0.492010 | 0.394304 | 0.379550 | 0.367276 | 0.124734 |
| TSR-path | 0.041379 | 0.040861 | 0.035967 | 0.042351 | -0.000972 |
| R-F1@0.5 | 0.024786 | 0.026277 | 0.019568 | 0.016336 | 0.008449 |
| R-F1@0.75 | 0.004837 | 0.007082 | 0.004499 | 0.002807 | 0.002030 |
| LIG-F1 | 0.022777 | 0.015691 | 0.021878 | 0.008033 | 0.014744 |

从 L1 到 L4，model-macro `Value-nED / R-F1@0.5 / R-F1@0.75 / LIG-F1` 分别下降约 25.4% / 34.1% / 42.0% / 64.7%。`Schema-nTED` 只下降 8.8%，`TSR-path` 基本持平且多模型呈非单调变化，因此不能声称所有能力随该难度等级单调下降。Page-EM 也过于稀疏：仅 Qwen3.6 在 L1 exact 3 页、Qwen3.5 在 L2 exact 1 页。

Qwen3.6 在四个难度级别的 Schema-nTED 和 LIG-F1 上均最高；其 L1 到 L4 的 Value-nED 从 0.738530 降至 0.583691，R-F1@0.5 从 0.070932 降至 0.040691，LIG-F1 从 0.119611 降至 0.037215。同时 coverage 从 0.987273 降至 0.766364，所以其 full-scope drop 同时包含能力退化和无效预测增加，报告时必须连同覆盖率解释。

### 约束切片与评价组件消融

约束切片定义为：

```text
Delta(c) = mean(CDS | constraint absent) - mean(CDS | constraint present)
```

正值表示约束存在时得分更低。对 aligned canonical run，region-local grids 通常是影响最大的约束之一：Qwen3.6 为 `0.1570`、Qwen3.5 为 `0.1091`、CapRL 为 `0.1023`；GLM 最大的是 dense key-field relations，为 `0.0911`。

最新版消融不是模型架构或训练模块实验，而是固定预测的评价组件敏感性分析。正式输入只包含主表选出的 9 个最佳全量 raw run，共 63,000 个模型-页面对；语义基线为 `Schema-nTED + Value-nED + TSR-path` 等权均值，空间项使用 corrected 逐页结果。GT scope/model 固定为 region 7,000、grid/LIG 2,400、widget 6,100、relation 7,000，缺失/无效预测在适用 scope 内按 0。

九模型宏平均结果如下。`Delta = score_with - score_without`，负值表示新增或更严格的组件暴露了更低得分；由于分数是等权组合，该值不能解释为模型组件的因果贡献。

| Comparison | Delta |
| --- | ---: |
| + corrected region R-F1@0.5 | -0.050588 |
| R-F1@0.75 替换 0.5 | -0.004252 |
| local-grid/LIG vs global-grid | -0.011305 |
| + corrected LIG-F1 | -0.032374 |
| + metadata widget answer | -0.023388 |
| + explicit relation | -0.030908 |
| full structural vs semantic | -0.105305 |

### 视觉退化鲁棒性

最新报告使用 68 个 clean 样本，以及每个样本的 5 类退化 x 3 个等级，共 1,020 张退化图。完成全部 1,020 次尝试的 7 个模型统一进入正式结果；汇总时假定 clean/degraded 运行之间的 backend 差异可忽略。

下表对每个模型先在 5 类退化上求均值，再对 7 个模型做等权 model-macro。drop 定义为 clean - degraded：

| Level | Clean coverage | Degraded coverage | Schema-nTED drop | Value-nED drop |
| --- | ---: | ---: | ---: | ---: |
| low | 0.955882 | 0.944958 | 0.032780 | 0.009242 |
| medium | 0.955882 | 0.948319 | 0.032216 | 0.013349 |
| high | 0.955882 | 0.950420 | 0.039435 | 0.017909 |

High 档 Value-nED 的绝对 drop 为 0.017909，约为 clean 宏平均的 3.8%。按退化类型看，high 档 Value-nED drop 最大的是 `blur_noise`（0.030640），其次是 `erode`（0.022537）；对应 Schema-nTED drop 为 0.045547 和 0.034169。Page-EM 在该 68 页子集的 clean/degraded 全部为 0，不能用于鲁棒性排序。

空间指标仅对 `blur_noise / erode / occlusion_stain` 有效；`dilate` 含局部 warp，`perspective_skew` 改变全页几何，但没有同步变换后的 bbox GT，因此两者的 `R-F1@0.5 / R-F1@0.75 / LIG-F1` 正式值为 `NA`。旧 CDS 已退出主指标，旧 failure boundary 也不再报告。

component-level 结果已改为按 clean instance 的 `template_name` 连接模板 metadata，绝不从 prediction、退化图或文件名反推标签。widget/relation 的 q75 阈值先在正式主索引的 70 个模板上冻结，再选 robustness 的 68 个模板。五个可重叠切片的 present 数分别为 local-grid 30、widget 32、dense relation 18、LIG 18、mixed-layout 10。

High severity 下，五个组件切片的 Value-nED excess drop 均为负，范围为 `-0.016032` 至 `-0.055383`；这表示当前切片中 component-present 页的 drop 小于 component-absent 页。由于样本较小且切片重叠，结果必须与 present/absent 分母一起报告，不能解释为组件的因果保护作用。

## 旧版到当前版

| 旧版 | 当前处理 |
| --- | --- |
| `TSR` | 拆为严格路径和值的 `TSR-path`，以及忽略路径的 `VAcc` |
| `R-F1` | 使用显式 run-level 坐标转换和 canonical type compatibility 重算；同时报告 0.5/0.75 |
| `LG` | line-item group 继续报告 `LIG-F1`；local grid topology 新增 `LG-GriTS-Top` |
| `WG-F1` | 使用 raw metadata group membership 和实例级 state 恢复，按两层 Hungarian matching 重算 |
| `Rel-F1` | 恢复为 page-macro 主指标；micro、per-type 和 matched-endpoint 结果作为附录诊断 |
| 四项/五项 CDS | 从正式主表删除，仅解释 legacy 辅助实验 |
| 无整页严格指标 | 新增独立 `Page-EM` 和 exact page count |
| 无整页软相似度 | 新增独立 `Schema-nTED / Value-nED` |

不要在同一张表中混用旧版表头和当前数值。引用旧结果时应明确标注 `legacy metric definition`。

## 当前产物与注意事项

主要文件：

| 文件 | 用途 |
| --- | --- |
| [`outputs/main_exp/main_experiment_results.csv`](outputs/main_exp/main_experiment_results.csv) | 最新正式 9-model 主表；每行 `n_attempted=7000`，学生整理结果时只读此文件 |
| [`outputs/main_exp/final_reporting_metrics_table.tex`](outputs/main_exp/final_reporting_metrics_table.tex) | 最新主表 LaTeX |
| [`outputs/main_exp/corrected_structure_metrics.csv`](outputs/main_exp/corrected_structure_metrics.csv) | corrected R/LIG 汇总、precision/recall 和 bbox 审计 |
| [`outputs/main_exp/corrected_structure_per_sample`](outputs/main_exp/corrected_structure_per_sample) | corrected R/LIG 逐页结果，共 161,000 行 |
| [`outputs/main_exp/hierarchical_structure_metrics.csv`](outputs/main_exp/hierarchical_structure_metrics.csv) | 24-run LG-GriTS-Top、WG-F1、page-macro/micro Rel-F1 汇总 |
| [`outputs/main_exp/hierarchical_relation_type_metrics.csv`](outputs/main_exp/hierarchical_relation_type_metrics.csv) | 各 run、各 relation type 的 corpus-level F1 附表 |
| [`outputs/main_exp/hierarchical_structure_metrics_metadata.json`](outputs/main_exp/hierarchical_structure_metrics_metadata.json) | GT provenance、R-F1/LIG 对齐、空集和排除规则审计 |
| [`outputs/main_exp/hierarchical_structure_per_sample`](outputs/main_exp/hierarchical_structure_per_sample) | hierarchical metrics 逐页诊断，共 161,000 条可恢复范围记录 |
| [`outputs/main_exp/main_results.csv`](outputs/main_exp/main_results.csv) | legacy 汇总；其中旧 R/LIG/CDS 不进入最新主表 |
| [`outputs/main_exp/per_sample_metrics.jsonl`](outputs/main_exp/per_sample_metrics.jsonl) | legacy 逐样本指标，共 161,003 行、24 个 run |
| [`outputs/main_exp/page_em_results.csv`](outputs/main_exp/page_em_results.csv) | 24 个 run 的 Page-EM、exact count 和覆盖率 |
| [`outputs/main_exp/page_em_exact_matches.jsonl`](outputs/main_exp/page_em_exact_matches.jsonl) | 143 条整页 exact-match 记录，含预测与 GT 路径 |
| [`outputs/main_exp/document_similarity_results.csv`](outputs/main_exp/document_similarity_results.csv) | 24 个 run 的 Schema-nTED、Value-nED 和覆盖率 |
| [`outputs/main_exp/document_similarity_results_metadata.json`](outputs/main_exp/document_similarity_results_metadata.json) | 两个软指标的公式、成本、依赖版本和聚合规则 |
| [`outputs/main_exp/metric_correlations_model_level.md`](outputs/main_exp/metric_correlations_model_level.md) | 9 个可比 raw run 的完整 Pearson/Spearman 相关矩阵 |
| [`outputs/main_exp/metric_correlations_model_level_pairs.csv`](outputs/main_exp/metric_correlations_model_level_pairs.csv) | 所有指标对的相关系数和 p-value |
| [`outputs/aux_exp/difficulty/difficulty_results.csv`](outputs/aux_exp/difficulty/difficulty_results.csv) | 9 个最佳全量 raw run × L1-L4 的 36 行最新难度结果 |
| [`outputs/aux_exp/difficulty/difficulty_diagnostic_summary.csv`](outputs/aux_exp/difficulty/difficulty_diagnostic_summary.csv) | 各模型各指标的 L1-L4 pivot、绝对与相对 drop |
| [`outputs/aux_exp/latex/auxiliary_experiments_tables.tex`](outputs/aux_exp/latex/auxiliary_experiments_tables.tex) | legacy 辅助表；其中消融部分仍混有 aligned/partial run，不作为正式结果 |
| [`outputs/aux_exp/structure_ablation/report_latest/ablation_targeted_deltas.csv`](outputs/aux_exp/structure_ablation/report_latest/ablation_targeted_deltas.csv) | 9 个最佳全量 raw run × 7 个 targeted comparison 的 63 行正式消融结果 |
| [`outputs/aux_exp/structure_ablation/report_latest/ablation_targeted_macro.csv`](outputs/aux_exp/structure_ablation/report_latest/ablation_targeted_macro.csv) | 7 行九模型消融宏平均 |
| [`outputs/aux_exp/structure_ablation/report_latest/ablation_results.md`](outputs/aux_exp/structure_ablation/report_latest/ablation_results.md) | 最新评价组件消融可读报告 |
| [`outputs/robustness_exp/report_latest/visual_degradation_results.csv`](outputs/robustness_exp/report_latest/visual_degradation_results.csv) | 7 个完整模型的 105 行统一视觉退化条件表 |
| [`outputs/robustness_exp/report_latest/visual_degradation_variant_severity.csv`](outputs/robustness_exp/report_latest/visual_degradation_variant_severity.csv) | 7 模型按退化类型和强度的等权宏平均表 |
| [`outputs/robustness_exp/report_latest/visual_degradation_component_membership.csv`](outputs/robustness_exp/report_latest/visual_degradation_component_membership.csv) | 68 个实例 × 5 个 metadata 组件标签及规则 |
| [`outputs/robustness_exp/report_latest/visual_degradation_by_component.csv`](outputs/robustness_exp/report_latest/visual_degradation_by_component.csv) | 7 模型的 1,050 行 component present/absent 条件明细 |
| [`outputs/robustness_exp/report_latest/visual_degradation_component_contrast.csv`](outputs/robustness_exp/report_latest/visual_degradation_component_contrast.csv) | 7 模型的 525 行 component present-vs-absent 对照 |
| [`outputs/robustness_exp/report_latest/visual_degradation_component_severity.csv`](outputs/robustness_exp/report_latest/visual_degradation_component_severity.csv) | 7 模型的 15 行 component × severity 宏平均 |
| [`outputs/robustness_exp/report_latest/visual_degradation_component_excess_drop_severity.csv`](outputs/robustness_exp/report_latest/visual_degradation_component_excess_drop_severity.csv) | component present-vs-absent 的 15 行 excess-drop 对照 |
| [`README_AUXILIARY_EXPERIMENTS.md`](README_AUXILIARY_EXPERIMENTS.md) | 辅助实验完整运行手册 |
| [`formtsr_exp/README_FormTSR_main_exp.md`](formtsr_exp/README_FormTSR_main_exp.md) | 主实验和 adapter 运行说明 |

当前 `per_sample_metrics.jsonl` 中有 74,109 行 `missing_prediction` 和 162 行 `invalid_json`。`outputs/main_exp` 下旧的 difficulty/constraint/ablation metadata 记录的是 154,004 行，与当前 161,003 行主 metrics 不同步。难度结果必须使用 `outputs/aux_exp/difficulty` 下的 9-model/36-row 最新表；消融必须使用 `outputs/aux_exp/structure_ablation/report_latest`。约束切片尚保留 legacy 口径，需按对应章节检查。

已知解释风险：

- `TSR-path` 和 `VAcc` 都不惩罚额外预测项，不能单独衡量 precision。
- `TSR-path` 对缺失 path 使用 `dict.get`；当 GT 真值恰好为 JSON `null` 时，缺失预测可能被当成命中。这是当前实现的边界问题。
- corrected R-F1 使用完整 run 分母；corrected LIG-F1 单列 `n_lig_applicable`，不能与 legacy `NA` 聚合混用。
- 在已有输出目录运行 `evaluate --models` 时，评测器会合并该目录 `per_model_metrics` 中的其他历史 run。需要干净快照时必须使用新的输出目录。

## 重评已有预测

下面命令不会重新调用模型。prediction 或 GT/layout metadata 更新后按顺序运行：

```bash
PY=.venv/bin/python
$PY -m formtsr_exp.page_em_report --workers 8
$PY -m formtsr_exp.document_similarity_report --workers 16
$PY -m formtsr_exp.structure_metrics_report --workers 4
$PY -m formtsr_exp.hierarchical_metrics_report --workers 4
$PY -m formtsr_exp.final_metrics_report
$PY -m formtsr_exp.metric_correlation_report
```

加入新 run 前，必须先在 `configs/bbox_coordinate_spaces.json` 中登记 bbox source space。最终合并脚本会校验四份来源的 run 集合和 `n_total/n_valid_json`；不一致时直接报错，不生成表。

## 独立评测域

以下指标属于仓库中的其他任务，不进入 FormTSR 主表：

- 图像检索：`Recall@K`、`mAP`、`nDCG`，实现见 [`src/eval.py`](src/eval.py)。
- OOD 检索诊断：MRR、mean rank 等，见 [`src/eval_ood.py`](src/eval_ood.py)。
- 旧 PEFT 抽取：JSON/field exact match，见 [`peft`](peft)。
- RAG 后处理：slot precision/recall/F1，见 [`src-Rag-postprocess`](src-Rag-postprocess)。
- GNN 分类：accuracy、macro-F1，见 [`gnn`](gnn)。

这些评测对象、数据划分和分母都不同，数值不能与 FormTSR 的 CDS 或字段指标直接比较。
