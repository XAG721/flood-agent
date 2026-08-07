# MuSiQue 冻结 v76 路由器迁移实验（v77 development）

- 状态：`MUSIQUE_V77_FROZEN_V76_ROUTER_TRANSFER_ADVANTAGE_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`
- 候选证据 F1：`0.535779`
- 最强登记对照：`alternating_anchor_link_control` / `0.542004`
- 差值：`-0.006225`，95% CI [`-0.012733`, `+0.000164`]
- 软闭包覆盖率：`0.437500`
- 非劣包络：`False`
- Gate 2：`NO-GO/SHADOW`

## 严格迁移门槛

- PASS `exact_cases_equals_800`
- PASS `exact_hop_quota`
- PASS `prior_or_stage_overlap_equals_0`
- PASS `invalid_selector_output_rate_equals_0`
- FAIL `candidate_evidence_macro_f1_at_least_0_55`
- FAIL `candidate_complete_evidence_recall_at_least_0_55`
- FAIL `candidate_minus_strongest_control_f1_at_least_0_005`
- FAIL `candidate_minus_strongest_control_ci_low_above_0`
- PASS `candidate_minus_cross_encoder_f1_at_least_0_005`
- PASS `candidate_minus_cross_encoder_ci_low_above_0`
- FAIL `every_hop_delta_vs_best_control_at_least_minus_0_005`
- PASS `soft_override_rate_between_0_15_and_0_55`
- PASS `mean_selected_tokens_within_1_05_of_strongest_control`

该实验是完整冻结 v76 路由器的零调参迁移，不是 MuSiQue 官方排行榜、答案生成、真实 SetR、洪水领域专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。
