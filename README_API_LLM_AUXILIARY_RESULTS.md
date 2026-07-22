# API-based LLM 辅助实验结果整理说明

模型范围固定为以下六个，最终表格统一按此顺序排列：

1. Qwen3.7 Plus
2. GPT-5.5
3. Sonnet-5
4. Gemini 3.5 Flash
5. Seed 2.1 Pro
6. Kimi-K2.6

这里的 `Kimi-K2.6` 不等于本地模型 `Kimi-VL-A3B-Instruct`；`Qwen3.7 Plus` 也不等于 `Qwen-VL-Plus` 或 `Qwen3.7 Max`。这些结果不能互相替代。

## 1. 你的任务边界

预测和指标计算是解耦的：

- 你负责按约定 schema 生成六个 API 模型的 prediction，并保留模型版本、调用参数、raw response 和失败记录。
- 负责人将 prediction 导入当前评测仓库，用冻结的最新版代码统一计算指标。
- 负责人把 API 六模型对应的 metric CSV 交给你后，你再按本文档整理结果表。
- 你不需要自行实现 Page-EM、nTED、R-F1 或其他 evaluator，也不要用旧脚本重算或手工修正分数。

除主实验总表外，你当前需要整理四组结果：

| 实验 | 是否需要额外调用 API | 最终用途 |
|---|---:|---|
| 难度分层 | 否 | 比较 L1-L4 的能力变化 |
| 评价组件消融 | 否 | 分析不同评价组件对综合评分的影响 |
| 视觉退化鲁棒性 | 是 | 比较 clean 与 degraded 的性能下降 |
| 视觉退化 component slice | 否 | 判断哪些页面结构在退化下更脆弱 |

难度分层和评价组件消融都直接复用 7,000 页主实验 prediction。视觉退化只需额外生成 degraded prediction；component slice 复用同一批视觉退化结果，不需要第三次调用模型。

## 2. Prediction 交接要求

虽然两边的输出格式已经对齐，交接前仍需检查以下不变量，避免导入时因命名而丢样本。

### 2.1 文件组织

六个模型统一使用下列 `run_id`，不要把 smoke 目录名或供应商返回的长 model id 直接作为目录名：

| display_name | run_id |
|---|---|
| Qwen3.7 Plus | `qwen3_7_plus_api` |
| GPT-5.5 | `gpt_vlm` |
| Sonnet-5 | `claude_vlm` |
| Gemini 3.5 Flash | `gemini_vlm` |
| Seed 2.1 Pro | `seed_2_1_pro_api` |
| Kimi-K2.6 | `kimi_k2_6_api` |

文件目录为：

```text
pred/
  {run_id}/
    {sample_id}.json
```

- 文件名必须是 index 原始 `sample_id` 加 `.json`，不能自行编号或改大小写。
- 主实验每个模型有 7,000 个 attempted sample。
- 视觉退化每个模型有 1,020 个 attempted sample。
- 如果调用失败，保留对应 raw/error 记录，不要编造空 prediction 文件。
- `run_id`、论文显示名和供应商 model id 的映射单独写入 `model_name_map.csv`，交接后不要改目录名。

`model_name_map.csv` 至少包含：

```text
model,model_id,display_name,api_provider,api_model,run_date
```

其中 `model` 是 prediction 目录使用的 `run_id`，`display_name` 必须使用本文档开头的六个名称。

主实验输入清单以负责人提供的 `dataset_index.jsonl` 为准。正式版本必须恰好有 7,000 行，其 SHA-256 为：

```text
d7fcca1ea45453e8acf93476b6007cd175b8987cfae9bf91123ab1fae9247489
```

本地复制数据时可以调整 `image_path` 的路径前缀，但不得改 `sample_id`、`template_name` 或 `instance_id`。

### 2.2 单页 JSON

每个 prediction 文件是一个可被标准 JSON parser 直接读取的对象，不带 Markdown fence、解释文字或 thinking tag。文件内容直接是 prediction object，不要再外包成 `{"sample_id": ..., "prediction": ...}` 或 `{"model": ..., "result": ...}`。顶层保留以下七个 key：

```text
regions, widgets, local_grids, cells, line_item_groups, relations, answer
```

`answer` 必须是对象；其余结构没有预测时使用空数组。bbox 继续使用双方已经约定的 `0-1000` 归一化坐标 `[left, top, right, bottom]`。不要在结果整理阶段修改 prediction 内容。

### 2.3 推理设置

- 六个模型使用同一版 prompt、同一输入 index 和同一输出 schema。
- prompt 以负责人提供的当前版 `formtsr_exp/prompt.py` 中 `PROMPT` 为准，不使用旧输出目录中可能残留的 prompt 副本。
- 正式运行使用 `temperature=0`；记录实际 `max_tokens`、provider model id、运行日期和任何重试设置。
- 视觉退化必须与同一模型的 clean 主实验使用相同 provider、model id、prompt 和解码参数，只替换输入图片。
- API 若返回具体 revision/version，必须记录；无法获得时写 `NA`，不能猜测。
- API key 只通过环境变量或密钥管理器传入，不得写进脚本、配置、README 或结果包。

导入后，负责人会把这六个 run 的 bbox coordinate space 登记为 `normalized_1000`，再运行 corrected structure evaluator。学生端不要根据单页坐标最大值自行猜测或改写 bbox。

## 3. 统一结果口径

负责人交回的最新版结果会使用以下正式指标：

```text
Valid/Total | Page-EM | Schema-nTED | Value-nED | TSR-path | R-F1@0.5 | R-F1@0.75 | LIG-F1
```

整理所有 CSV 时遵循以下规则：

1. 只保留上述六个 API 模型，不混入本地模型、smoke run、aligned metadata run 或其他后端结果。
2. 主实验只有 `n_attempted=7000` 的模型能进正式表；视觉退化只有 `n_attempted=1020` 的模型能进正式表。
3. `n_valid_json` 可以小于 attempted 数。无效页保留在正式分母中并按评测规则记 0，不能只在 valid 页上取均值。
4. 每张模型表都保留 `n_total`、`n_valid_json` 和 `coverage`。Page-EM 同时保留 `n_exact_match`，正文优先写成 `exact/total`。
5. 分数统一保留 6 位小数，范围为 `[0,1]`，越高越好。
6. `NA` 表示该指标不适用，必须原样保留，不能改成空值或 0。
7. 不再报告旧 `VAcc`、旧 `R-F1`、旧 `LIG-F1`、`WAcc` 或 `CDS`，除非负责人明确要求做历史诊断。
8. 不用 Excel 公式重新计算均值或 drop。最终展示表只能由负责人交回的 CSV 筛选、转置或改显示名得到。

如果六个模型中有模型尚未跑完，在进度表中标为 `incomplete`，但不把 partial 数字放进正式比较表，也不以 0 补齐。

## 4. 难度分层结果

### 4.1 数据来源

难度分层不需要额外推理。负责人会从主实验 prediction 生成 API 六模型版的：

```text
difficulty_results.csv
difficulty_diagnostic_summary.csv
difficulty_results_metadata.json
```

正式 7,000 页按模板冻结为：

| Level | 名称 | 每模型页数 |
|---|---|---:|
| L1 | easy | 1,100 |
| L2 | medium | 2,400 |
| L3 | hard | 2,400 |
| L4 | expert | 1,100 |

### 4.2 需要交付的表

`api_llm_difficulty_results.csv` 使用 long format，每个模型四行，六个模型共 24 行：

```text
Model,difficulty_level,n_total,n_valid_json,coverage,n_exact_match,Page-EM,Schema-nTED,Value-nED,TSR-path,R-F1@0.5,R-F1@0.75,LIG-F1,n_lig_applicable
```

`api_llm_difficulty_drop.csv` 每个模型、每个指标一行，六模型乘七指标共 42 行：

```text
Model,Metric,L1,L2,L3,L4,L1_to_L4_drop
```

其中：

```text
L1_to_L4_drop = score(L1) - score(L4)
```

正值表示模型从 easy 到 expert 变差。L1-L4 页数不同，不能把四个 level 的分数简单等权平均来替代主实验 overall 分数。正文重点讨论 `Schema-nTED / Value-nED / TSR-path / R-F1@0.5 / LIG-F1` 的趋势；`Page-EM` 很稀疏时报告 exact page count，不夸大微小小数差异。

## 5. 评价组件消融

### 5.1 实验性质

这不是模型架构、训练模块或 prompt ablation。它固定同一批主实验 prediction，只改变评价组件的组合，正式名称写：

```text
evaluation-component ablation
```

或：

```text
metric sensitivity analysis
```

不要写成“加入某模型组件后性能下降”。

### 5.2 数据来源和交付表

负责人会提供：

```text
ablation_targeted_deltas.csv
ablation_targeted_macro.csv
ablation_variants.csv
ablation_results_metadata.json
```

正文使用 `api_llm_ablation_targeted.csv`，保留七组 paired comparison。每个模型七行，六个模型共 42 行：

```text
Model,comparison,n_scope,score_with,score_without,delta,relative_delta_pct
```

另交 `api_llm_ablation_macro.csv`，对六个 API 模型做 model-macro，共 7 行。七组 comparison 为：

1. 加入 `R-F1@0.5`。
2. `R-F1@0.75` 相对 `R-F1@0.5` 的严格定位影响。
3. region-local grid 相对 global grid。
4. 加入 `LIG-F1`。
5. 加入 widget answer component。
6. 加入 explicit relation component。
7. full structural 相对 semantic-only。

统一定义：

```text
delta = score_with - score_without
```

负值表示新增或更严格的评价组件暴露了更低的得分，不表示该组件导致模型能力下降。`Page-EM` 不进入这里的等权 variant score，因为它过于稀疏。`ablation_variants.csv` 放附录或留作核查，不用八种 variant 再制造一个新的模型总排名。

## 6. 视觉退化鲁棒性

### 6.1 需要额外跑的数据

视觉退化 clean baseline 复用主实验 prediction，不重新调用 API。每个模型只额外跑：

```text
68 clean templates x 5 degradation variants x 3 severity levels = 1,020 degraded samples
```

六个模型合计 6,120 个 attempted degraded samples。五种退化为：

```text
blur_noise, dilate, erode, occlusion_stain, perspective_skew
```

三个等级为：

```text
low, medium, high
```

额外推理只读取负责人提供的 `robustness_degraded_index.jsonl`。正式文件为 1,020 行，SHA-256 为：

```text
31d34edd790e065d800128a3d59ca50edb29b442a03391e84b3b16a78588ad59
```

与主实验相同，可以在本地调整 `image_path` 前缀，但所有 sample metadata 和文件名必须保持不变。

主实验有 70 个正式模板，但 robustness 只有 68 个。缺少的是 `ja_18` 和 `ja_20`：退化数据固定使用 instance `01`，而当前 clean 数据中这两个模板没有对应的 `01`，因此无法组成 clean/degraded pair。不要用其他 instance 顶替，也不要把分母改成 70。

### 6.2 数据来源和交付表

负责人统一配对和算分后会提供：

```text
visual_degradation_results.csv
visual_degradation_model_severity.csv
visual_degradation_variant_severity.csv
visual_degradation_by_difficulty.csv
visual_degradation_results_metadata.json
```

需要整理：

| 文件 | 行数，不含表头 | 用途 |
|---|---:|---|
| `api_llm_robustness_conditions.csv` | 90 | 6 模型 x 5 variant x 3 level |
| `api_llm_robustness_model_severity.csv` | 18 | 6 模型 x 3 level，跨 variant 汇总 |
| `api_llm_robustness_variant_macro.csv` | 15 | 5 variant x 3 level，六模型 macro |

condition 表至少保留：

```text
Model,degradation_variant,degradation_level,n_total,n_clean_valid_json,n_degraded_valid_json,clean_coverage,degraded_coverage,clean_{metric},degraded_{metric},{metric}_drop
```

每个指标的 drop 都定义为：

```text
drop = clean score - degraded score
```

正值表示退化造成损失，负值表示该条件下 degraded 分数偶然更高。不能把负值截成 0。

`Page-EM / Schema-nTED / Value-nED / TSR-path` 在全部 68 页上计算。空间指标 `R-F1@0.5 / R-F1@0.75 / LIG-F1` 只在几何保持的 `blur_noise / erode / occlusion_stain` 上报告；`dilate` 和 `perspective_skew` 没有可靠的变换后 bbox GT，因此空间指标必须为 `NA`。

六个 API 模型的 clean/degraded 应使用同一 provider 和同一模型版本，因此正式整理时六个都应属于 same-backend comparison。若某模型中途更换 API model id、版本、prompt 或解码设置，该模型只能单独标成 diagnostic，不能混入六模型 macro。

`visual_degradation_by_difficulty.csv` 可作为附录分析，不是主 robustness 表。使用时同时保留 level 的样本数，不把 L1-L4 等权平均。

## 7. 视觉退化 component-level 结果

### 7.1 标签来源

component 标签由负责人按 `template_name` 从模板 metadata 连接得到，不从 prediction、图片文件名或模型输出推断。五个 slice 为：

| Component | robustness 68 模板中的 present 数 |
|---|---:|
| Region-local grids | 30 |
| Widget grouping | 32 |
| Dense key-field relations | 18 |
| Line-item groups | 18 |
| Mixed layout | 10 |

这些 slice 会重叠，不是互斥分区，不能把 present 数相加当作 68。

### 7.2 需要交付的表

负责人会提供 component 版 latest CSV。正文整理两张表：

| 文件 | 行数，不含表头 | 解释 |
|---|---:|---|
| `api_llm_component_severity.csv` | 15 | 5 components x 3 severity，component-present 页上的六模型 macro drop |
| `api_llm_component_excess_drop.csv` | 15 | 5 components x 3 severity，present 相对 absent 的额外 drop |

统一定义：

```text
excess_drop = drop(component present) - drop(component absent)
```

正值表示具有该 component 的页面在视觉退化下更敏感。由于 component-present 和 component-absent 页本身难度分布不同，`excess_drop` 是切片对比，不要写成组件造成退化的因果结论。

以下文件保留作附录和复核：

```text
visual_degradation_component_condition_macro.csv
visual_degradation_component_contrast.csv
visual_degradation_component_membership.csv
visual_degradation_component_metadata.json
```

六模型版本中，condition macro 应有 75 行，即 5 components x 5 variants x 3 levels；逐模型 contrast 最多 450 行，即 6 models x 5 components x 5 variants x 3 levels。空间指标的 `NA` 规则与第 6 节完全相同。

## 8. 当前不需要整理的内容

除非负责人另外提供新版正式 CSV，下列内容不属于本次本科生交付：

- legacy constraint slice；当前旧报告仍混用 legacy 指标。
- metadata alignment 诊断结果。
- cross-backend robustness diagnostic。
- 指标相关性分析。
- 旧 `VAcc / WAcc / CDS` 表。
- 本地模型与 API 模型的混合排名。

## 9. 最终目录和检查清单

最终交付目录建议固定为：

```text
api_llm_results/
  model_name_map.csv
  01_main/
  02_difficulty/
    api_llm_difficulty_results.csv
    api_llm_difficulty_drop.csv
  03_evaluation_component_ablation/
    api_llm_ablation_targeted.csv
    api_llm_ablation_macro.csv
  04_visual_robustness/
    api_llm_robustness_conditions.csv
    api_llm_robustness_model_severity.csv
    api_llm_robustness_variant_macro.csv
  05_visual_robustness_components/
    api_llm_component_severity.csv
    api_llm_component_excess_drop.csv
  source_manifest.csv
  result_notes.md
```

`source_manifest.csv` 记录每张整理后表对应的负责人原始 CSV 文件名；`result_notes.md` 记录未完成模型、API 版本变化、异常重试和任何不能进入正式表的原因。

提交前逐项确认：

- [ ] 所有正式表只包含六个指定 API 模型，顺序一致。
- [ ] 主实验每个正式模型 attempted 为 7,000；视觉退化为 1,020。
- [ ] 难度 long table 为 24 行，七组 ablation targeted table 为 42 行。
- [ ] robustness condition/model severity/variant macro 分别为 90/18/15 行。
- [ ] component severity 和 excess-drop 表各 15 行。
- [ ] 所有表都保留 coverage；Page-EM 保留 exact count。
- [ ] `NA` 没有被改成 0，负 drop 没有被截断。
- [ ] 没有混入 partial、smoke、aligned metadata、本地模型或 cross-backend 诊断结果。
- [ ] 文字中没有把 metric sensitivity 或 component slice 写成因果 ablation。
- [ ] 表格数字来自负责人交回的最新版 CSV，没有手工重算或补值。
