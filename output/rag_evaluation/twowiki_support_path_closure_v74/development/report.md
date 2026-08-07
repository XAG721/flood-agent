# 2WikiMultiHopQA 支持路径闭包实验（v74 development）

- 案例：`800`
- 状态：`2WIKI_V74_SUPPORT_PATH_CLOSURE_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- Gate 2：`NO-GO/SHADOW`

## 主要结果

| 方法 | 证据 F1 | 证据召回 | 完整证据召回 | 平均 token |
|---|---:|---:|---:|---:|
| `bm25_topk` | 0.302909 | 0.477125 | 0.160000 | 113.902500 |
| `dense_topk` | 0.416413 | 0.660813 | 0.341250 | 115.090000 |
| `hybrid_topk` | 0.386214 | 0.613000 | 0.286250 | 114.961250 |
| `cross_encoder_topk` | 0.442893 | 0.701063 | 0.390000 | 123.723750 |
| `coverage_greedy_proxy` | 0.442536 | 0.700438 | 0.388750 | 123.675000 |
| `frc_select` | 0.442893 | 0.701063 | 0.390000 | 123.723750 |
| `title_link_only_control` | 0.387298 | 0.601688 | 0.341250 | 123.183750 |
| `cross_plus_title_link_without_source_diversity_control` | 0.494306 | 0.779375 | 0.553750 | 125.003750 |
| `hard_title_link_chain_control` | 0.496488 | 0.779375 | 0.562500 | 125.862500 |
| `alternating_anchor_link_control` | 0.517270 | 0.799437 | 0.551250 | 124.967500 |
| `soft_title_link_support_path_closure_v74` | 0.518988 | 0.809375 | 0.607500 | 126.108750 |

## 冻结门槛

候选证据 F1 为 `0.518988`；最强同资源对照为 `alternating_anchor_link_control`（`0.517270`）。
配对差值 `+0.001718`，95% CI [`-0.008036`, `+0.011401`]。

- PASS `exact_cases_equals_800`
- PASS `exact_question_type_balance`
- PASS `history_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- PASS `candidate_evidence_macro_f1_at_least_0_50`
- PASS `candidate_evidence_macro_recall_at_least_0_80`
- PASS `candidate_complete_evidence_recall_at_least_0_55`
- FAIL `candidate_minus_strongest_control_f1_at_least_0_01`
- FAIL `candidate_minus_strongest_control_ci_low_above_0`
- PASS `candidate_minus_frc_select_f1_at_least_0_03`
- PASS `candidate_minus_frc_select_ci_low_above_0`
- FAIL `every_question_type_delta_at_least_minus_0_02`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

该实验不是官方排行榜、独立数据集、自动分解、答案生成、洪水领域专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。
