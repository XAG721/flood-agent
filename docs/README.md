# 文档索引

本目录只维护当前系统事实、现行升级合同和可复现验收证据。已完成的阶段性规划、重复导出件和过时接口说明不再保留在当前树中；需要追溯时使用 Git 历史。

## 现行合同

1. [`洪水预警响应系统_渐进式迭代开发与升级设计.md`](../洪水预警响应系统_渐进式迭代开发与升级设计.md)：当前迭代与门禁合同。
2. [`面向区县防办的洪水预警响应系统_升级设计说明V3.md`](../面向区县防办的洪水预警响应系统_升级设计说明V3.md)：产品定位和 V3 业务边界。
3. [`PRODUCT.md`](../PRODUCT.md)：面向区县防办的产品原则。

## 当前实现与验收

- [`releases/v0.3.2.md`](./releases/v0.3.2.md)：当前版本的 FRC-RAG 公开数据迭代、独立盲评基础设施、验证结果与放行边界报告。
- [`releases/v0.3.1.md`](./releases/v0.3.1.md)：三轮代码结构重构、运行时边界、性能与安全影响报告。
- [`releases/v0.3.0.md`](./releases/v0.3.0.md)：上一版本的外部接口仿真、场景目录、浏览器验收、性能和安全更新报告。
- [`progressive_upgrade/`](./progressive_upgrade/)：资产处置、合同、数据字典、运维、安全、评测和阶段验收。
- [`progressive_upgrade/frc_public_evaluation_protocol.md`](./progressive_upgrade/frc_public_evaluation_protocol.md)：FRC-RAG 公开数据真实模型参考审计的公平协议、难例覆盖和 Gate 2 判据。
- [`progressive_upgrade/hotpot_graph_router_protocol_v76.json`](./progressive_upgrade/hotpot_graph_router_protocol_v76.json)：HotpotQA 图与分数软闭包覆盖路由的目标前冻结模型、对照、门槛和双阶段规则。
- [`progressive_upgrade/hotpot_graph_router_synthesis_v76.md`](./progressive_upgrade/hotpot_graph_router_synthesis_v76.md)：800 例开发与 800 例确认均通过预注册门槛的 v76 结果、训练数据边界和禁止外推主张。
- [`../output/rag_evaluation/hotpot_graph_router_v76/confirmation/report.md`](../output/rag_evaluation/hotpot_graph_router_v76/confirmation/report.md)：确认阶段证据 F1、配对区间、分层非劣与 Gate 2 保持 `NO-GO/SHADOW` 的机器报告。
- [`progressive_upgrade/musique_graph_router_transfer_protocol_v77.json`](./progressive_upgrade/musique_graph_router_transfer_protocol_v77.json)：完整冻结 v76 路由器零调参迁移到未触碰 MuSiQue 案例的来源隔离、hop 配额、严格优势与次级非劣规则。
- [`progressive_upgrade/musique_graph_router_transfer_synthesis_v77.md`](./progressive_upgrade/musique_graph_router_transfer_synthesis_v77.md)：相对 cross-encoder 显著为正、但弱于最强交替锚点控制并停止在确认前的 v77 负结果与外推边界。
- [`../output/rag_evaluation/musique_graph_router_transfer_v77/development/report.md`](../output/rag_evaluation/musique_graph_router_transfer_v77/development/report.md)：v77 开发阶段七方法指标、配对区间、hop 分层和 `NO-GO/SHADOW` 机器报告。
- [`progressive_upgrade/musique_mean_calibrated_three_route_protocol_v78.json`](./progressive_upgrade/musique_mean_calibrated_three_route_protocol_v78.json)：把 v77 显式限定为总体均值校准集，并在新来源上验证三路路由的 600 例门禁协议。
- [`progressive_upgrade/musique_mean_calibrated_three_route_synthesis_v78.md`](./progressive_upgrade/musique_mean_calibrated_three_route_synthesis_v78.md)：历史双增益头、单一均值校准、非劣成立但严格优势失败及确认未打开的 v78 综合边界。
- [`../output/rag_evaluation/musique_mean_calibrated_three_route_v78/development/report.md`](../output/rag_evaluation/musique_mean_calibrated_three_route_v78/development/report.md)：v78 八方法指标、配对区间、路由分布、hop 分层和 `NO-GO/SHADOW` 机器报告。
- [`progressive_upgrade/conformal_score_stability_protocol.json`](./progressive_upgrade/conformal_score_stability_protocol.json)：四阶段检索分数稳定性 64 维目标域充分性头的冻结协议、采用门槛与失败停止规则。
- [`../output/rag_evaluation/conformal_score_stability/conformal_score_stability.md`](../output/rag_evaluation/conformal_score_stability/conformal_score_stability.md)：召回/AUC 改善但风险一致性失败的逐数据集结果与 `DO_NOT_ADOPT` 结论。
- [`progressive_upgrade/conformal_review_ranking_protocol.json`](./progressive_upgrade/conformal_review_ranking_protocol.json)：保持自动声明逐条不变的 5%/10%/20% 人工复核排序冻结协议与子组采用上限。
- [`../output/rag_evaluation/conformal_review_ranking/conformal_review_ranking.md`](../output/rag_evaluation/conformal_review_ranking/conformal_review_ranking.md)：总体复核捕获改善但 HotpotQA 最差子组超限的 `KEEP_BASE_REVIEW_ORDER` 结果。
- [`../output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.md`](../output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.md)：HousingQA 冻结真实模型字段/角色权重敏感性与限制。
- [`../output/rag_evaluation/conformal_sufficiency/conformal_sufficiency.md`](../output/rag_evaluation/conformal_sufficiency/conformal_sufficiency.md)：冻结真实模型分数上的 case 分组证据充分性校准、误放行风险和拒答代价。
- [`../output/rag_evaluation/conformal_robustness/conformal_robustness.md`](../output/rag_evaluation/conformal_robustness/conformal_robustness.md)：重复 case 分组的充分性风险—效用稳健性和人工复核分层。
- [`../output/rag_evaluation/conformal_model_selection/conformal_model_selection.md`](../output/rag_evaluation/conformal_model_selection/conformal_model_selection.md)：outer evaluation 不参与选型的充分性非线性特征/正则化嵌套实验。
- [`../output/rag_evaluation/conformal_cross_dataset/hotpotqa_confirmation.md`](../output/rag_evaluation/conformal_cross_dataset/hotpotqa_confirmation.md)：冻结安全协议在 HotpotQA 上的无选型跨数据集部分确认。
- [`../output/rag_evaluation/conformal_cross_dataset/multihoprag_confirmation.md`](../output/rag_evaluation/conformal_cross_dataset/multihoprag_confirmation.md)：同一冻结协议在 MultiHop-RAG 上的第二次独立完整确认。
- [`../output/rag_evaluation/conformal_cross_dataset/2wikimultihopqa_confirmation.md`](../output/rag_evaluation/conformal_cross_dataset/2wikimultihopqa_confirmation.md)：评分前冻结抽样、模型 revision 与门槛的 2WikiMultiHopQA 第三次独立完整确认。
- [`../output/rag_evaluation/conformal_cross_dataset/cross_dataset_confirmation_series.md`](../output/rag_evaluation/conformal_cross_dataset/cross_dataset_confirmation_series.md)：HotpotQA、MultiHop-RAG 与 2WikiMultiHopQA 的系列汇总及部分确认结论。
- [`progressive_upgrade/conformal_qasc_confirmation_protocol.json`](./progressive_upgrade/conformal_qasc_confirmation_protocol.json)：QASC 下载前冻结的数据版本、受控候选池、真实模型 revision、无 gold 泄漏和补充确认判据。
- [`../output/rag_evaluation/conformal_qasc_confirmation/conformal_qasc_confirmation.md`](../output/rag_evaluation/conformal_qasc_confirmation/conformal_qasc_confirmation.md)：QASC 严格 `NOT_CONFIRMED`、协议状态定义偏差及不改判审计。
- [`../output/rag_evaluation/conformal_qasc_subgroup_diagnostic/conformal_qasc_subgroup_diagnostic.md`](../output/rag_evaluation/conformal_qasc_subgroup_diagnostic/conformal_qasc_subgroup_diagnostic.md)：QASC 确认后的问题类型描述性异质性诊断，不用于选型或调参。
- [`progressive_upgrade/qasc_scoring_optimization_protocol.json`](./progressive_upgrade/qasc_scoring_optimization_protocol.json)：QASC 评分编排性能优化的冻结样本、实现哈希、等价容差与 1.10 倍采用门槛。
- [`../output/rag_evaluation/qasc_scoring_optimization/qasc_scoring_optimization.md`](../output/rag_evaluation/qasc_scoring_optimization/qasc_scoring_optimization.md)：32 例真实 GPU 基准的负结果；全局批大小 8 仅加速 1.030910 倍，故保留冻结逐 case 入口。
- [`progressive_upgrade/conflicts_expected_behavior_protocol.json`](./progressive_upgrade/conflicts_expected_behavior_protocol.json)：CONFLICTS 回答生成、逐例 A/B 盲化、双人独立评审与第三方裁决的结果前冻结协议。
- [`../output/rag_evaluation/conflicts_expected_behavior/PROTOCOL.md`](../output/rag_evaluation/conflicts_expected_behavior/PROTOCOL.md)：458 例、916 条待评回答的盲评说明；当前仅生成评审包，尚无人工 adherence 结果。
- [`progressive_upgrade/conflicts_annotation_operations_protocol.json`](./progressive_upgrade/conflicts_annotation_operations_protocol.json)：三名独立人员、每人 8 批、23 个隐藏复测及 0.80 组内一致性门槛的人工决定前冻结协议。
- [`../output/rag_evaluation/conflicts_annotation_operations/PROTOCOL.md`](../output/rag_evaluation/conflicts_annotation_operations/PROTOCOL.md)：24 个方法盲化批次的执行说明；公开作业无原条目 ID、重复标记和私有路由，空模板仍不是人工结果。
- [`progressive_upgrade/conflicts_annotation_workstation_contract.json`](./progressive_upgrade/conflicts_annotation_workstation_contract.json)：本机盲评工作台的冻结输入、哈希、会话安全、断点恢复、浏览器探针和“不构成人工结果”边界。
- [`progressive_upgrade/conflicts_annotation_collection_contract.json`](./progressive_upgrade/conflicts_annotation_collection_contract.json)：三角色 24 批完成回执、身份互异、私有路由承诺、0.80 组内质控和不解盲合并的 v31 失败关闭合同。
- [`progressive_upgrade/conflicts_selector_router_protocol.json`](./progressive_upgrade/conflicts_selector_router_protocol.json)：CONFLICTS 六选择器回顾性 case 五折路由、严格 no-FRC 消融、无 gold 特征白名单和不采用边界。
- [`../output/rag_evaluation/conflicts_selector_router/conflicts_selector_router.md`](../output/rag_evaluation/conflicts_selector_router/conflicts_selector_router.md)：总体 Accuracy 点增益存在但安全类 Recall、跨折稳定性与 FRC 边际贡献失败的 v32 结果。
- [`progressive_upgrade/whoqa_conflict_coverage_protocol.json`](./progressive_upgrade/whoqa_conflict_coverage_protocol.json)：WhoQA 下载前冻结的数据版本、六选择器、真实模型 revision、同预算指标、无 gold 泄漏与负结果边界。
- [`../output/rag_evaluation/whoqa_conflict_coverage/whoqa_conflict_coverage.md`](../output/rag_evaluation/whoqa_conflict_coverage/whoqa_conflict_coverage.md)：5,152 个未触碰问题上的独立冲突观点覆盖审计；FRC 未超过最强 BM25，Gate 2 保持 `NO-GO/SHADOW`。
- [`progressive_upgrade/whoqa_budget_stress_protocol.json`](./progressive_upgrade/whoqa_budget_stress_protocol.json)：复用 v33 盲评分、结果前冻结的六配置容量与 Token 压力规则。
- [`../output/rag_evaluation/whoqa_budget_stress/whoqa_budget_stress.md`](../output/rag_evaluation/whoqa_budget_stress/whoqa_budget_stress.md)：97 万次选择的 v34 负结果；紧预算下 FRC 未取得成本—观点覆盖优势。
- [`progressive_upgrade/rgb_cost_aware_frc_protocol.json`](./progressive_upgrade/rgb_cost_aware_frc_protocol.json) 与 [`progressive_upgrade/rgb_cost_aware_frc_execution.json`](./progressive_upgrade/rgb_cost_aware_frc_execution.json)：v35 在下载结果前冻结的 RGB 来源、盲适配、成本感知公式、九方法公平预算和统计判据。
- [`progressive_upgrade/rgb_cost_aware_frc_post_result_correction.json`](./progressive_upgrade/rgb_cost_aware_frc_post_result_correction.json)：保留初始负结果哈希，并把只存在于 `en_fact` 的错误正例率限定到正确统计总体。
- [`../output/rag_evaluation/rgb_cost_aware_frc/rgb_cost_aware_frc.md`](../output/rag_evaluation/rgb_cost_aware_frc/rgb_cost_aware_frc.md)：498 例真实模型盲评；成本感知 FRC 显著低于最强基线并触发安全回归，未采用。
- [`progressive_upgrade/musique_dual_resource_protocol.json`](./progressive_upgrade/musique_dual_resource_protocol.json) 与 [`progressive_upgrade/musique_dual_resource_execution.json`](./progressive_upgrade/musique_dual_resource_execution.json)：v36 在 MuSiQue 内容和神经评分前冻结的双资源公式、官方来源、盲适配、十方法公平预算与统计判据。
- [`../output/rag_evaluation/musique_dual_resource/musique_dual_resource.md`](../output/rag_evaluation/musique_dual_resource/musique_dual_resource.md)：2,417 例跨域多跳盲评；双资源 FRC 显著改善 v35，但仍显著低于最强非 FRC 基线，未采用。
- [`progressive_upgrade/musique_objective_gap_diagnostic_protocol.json`](./progressive_upgrade/musique_objective_gap_diagnostic_protocol.json) 与 [`../output/rag_evaluation/musique_objective_gap/musique_objective_gap.md`](../output/rag_evaluation/musique_objective_gap/musique_objective_gap.md)：v37 结果后精确集合枚举；v36 已接近冻结目标最优，精确解未显著改善证据选择。
- [`progressive_upgrade/musique_crossfit_support_protocol.json`](./progressive_upgrade/musique_crossfit_support_protocol.json) 与 [`../output/rag_evaluation/musique_crossfit_support/musique_crossfit_support.md`](../output/rag_evaluation/musique_crossfit_support/musique_crossfit_support.md)：v38 五折 case 隔离发现研究；通用校准小幅改善，但 FRC 角色增量区间跨零，停止该模型族。
- [`progressive_upgrade/hover_verification_roles_protocol.json`](./progressive_upgrade/hover_verification_roles_protocol.json)、[`progressive_upgrade/hover_verification_roles_execution.json`](./progressive_upgrade/hover_verification_roles_execution.json) 与 [`../output/rag_evaluation/hover_verification_roles/hover_verification_roles.md`](../output/rag_evaluation/hover_verification_roles/hover_verification_roles.md)：v39 在未接触 HoVer 内容前冻结的任务特定核验角色独立盲评；角色增量区间跨零，支持未建立。
- [`../output/rag_evaluation/hover_verification_roles/hover_role_mechanism_diagnostic.md`](../output/rag_evaluation/hover_verification_roles/hover_role_mechanism_diagnostic.md)：v39 结果后、无 gold 的静态角色信号坍缩诊断；角色排序近共线且选择高度相同，不作因果声称。
- [`progressive_upgrade/hover_dynamic_atomic_roles_v40_closure.json`](./progressive_upgrade/hover_dynamic_atomic_roles_v40_closure.json)：v40 在评分前因分区标签/跳数聚合意外暴露而关闭，2,512 个 ID 永久排除。
- [`progressive_upgrade/hover_dynamic_atomic_roles_protocol_v41.json`](./progressive_upgrade/hover_dynamic_atomic_roles_protocol_v41.json)、[`progressive_upgrade/hover_dynamic_atomic_roles_execution_v41.json`](./progressive_upgrade/hover_dynamic_atomic_roles_execution_v41.json)、[`progressive_upgrade/hover_dynamic_atomic_roles_result_v41.json`](./progressive_upgrade/hover_dynamic_atomic_roles_result_v41.json) 与 [`../output/rag_evaluation/hover_dynamic_atomic_roles/hover_dynamic_atomic_roles.md`](../output/rag_evaluation/hover_dynamic_atomic_roles/hover_dynamic_atomic_roles.md)：v41 动态原子角色先导机制通过；2,000 例同家族确认得到小幅显著正增益，但未达 +0.005/+0.010 实质门槛，支持未建立且 Gate 2 不变。
- [`progressive_upgrade/scifact_dynamic_atomic_roles_protocol_v42.json`](./progressive_upgrade/scifact_dynamic_atomic_roles_protocol_v42.json)、[`progressive_upgrade/scifact_dynamic_atomic_roles_implementation_v42.json`](./progressive_upgrade/scifact_dynamic_atomic_roles_implementation_v42.json)、[`progressive_upgrade/scifact_dynamic_atomic_roles_execution_v42.json`](./progressive_upgrade/scifact_dynamic_atomic_roles_execution_v42.json)、[`progressive_upgrade/scifact_dynamic_atomic_roles_result_v42.json`](./progressive_upgrade/scifact_dynamic_atomic_roles_result_v42.json) 与 [`../output/rag_evaluation/scifact_dynamic_atomic_roles/scifact_dynamic_atomic_roles.md`](../output/rag_evaluation/scifact_dynamic_atomic_roles/scifact_dynamic_atomic_roles.md)：v42 在未接触 SciFact 内容前冻结 v41 方法；外部盲评确认角色机制有区分度，但选择效用显著低于 Dense，禁止复用调参且 Gate 2 不变。
- [`../output/rag_evaluation/conformal_cross_dataset/source_preparation_benchmark.json`](../output/rag_evaluation/conformal_cross_dataset/source_preparation_benchmark.json)：重复解析与单次解析的全量性能和结果等价性证据。
- [`../output/rag_evaluation/conformal_subgroup_audit/conformal_subgroup_audit.md`](../output/rag_evaluation/conformal_subgroup_audit/conformal_subgroup_audit.md)：四数据集有限样本名义上界与可观察子群不稳定性审计。
- [`progressive_upgrade/conformal_subgroup_protocol.json`](./progressive_upgrade/conformal_subgroup_protocol.json)：子群维度、最低支持度、稳定性判据和理论声明边界。
- [`../output/rag_evaluation/conformal_mondrian/conformal_mondrian.md`](../output/rag_evaluation/conformal_mondrian/conformal_mondrian.md)：分层 Mondrian 校准相对全局阈值的事后方法开发、采用检查及负结果。
- [`../output/rag_evaluation/conformal_multi_axis_mondrian/conformal_multi_axis_mondrian.md`](../output/rag_evaluation/conformal_multi_axis_mondrian/conformal_multi_axis_mondrian.md)：候选规模感知八级 Mondrian 回退的四数据集事后开发、负结果及停止边界。
- [`../output/rag_evaluation/conformal_head_transfer/conformal_head_transfer.md`](../output/rag_evaluation/conformal_head_transfer/conformal_head_transfer.md)：禁用目标 train 标签与特征的四数据集 leave-one-dataset-out 充分性头迁移审计。
- [`../output/rag_evaluation/conformal_transfer_admission/conformal_transfer_admission.md`](../output/rag_evaluation/conformal_transfer_admission/conformal_transfer_admission.md)：calibration-only 跨域头准入证书的覆盖、误准入和失败关闭审计。
- [`progressive_upgrade/conformal_mondrian_protocol.json`](./progressive_upgrade/conformal_mondrian_protocol.json)：运行前冻结的分组、回退、支持度、采用门槛和非独立确认边界。
- [`../output/rag_evaluation/conformal_contextual/conformal_contextual.md`](../output/rag_evaluation/conformal_contextual/conformal_contextual.md)：train-only 上下文充分性头相对冻结全局头的事后开发及负采用结论。
- [`progressive_upgrade/conformal_contextual_protocol.json`](./progressive_upgrade/conformal_contextual_protocol.json)：上下文类别、训练支持度、全局校准和采用门槛的冻结协议。
- [`progressive_upgrade/conformal_cross_dataset_protocol.json`](./progressive_upgrade/conformal_cross_dataset_protocol.json)：第二确认集运行前冻结的输入、模型、阈值和系列判定规则。
- [`progressive_upgrade/conformal_third_confirmation_protocol.json`](./progressive_upgrade/conformal_third_confirmation_protocol.json)：2Wiki 原始文件哈希、1000 例抽样、模型 revision、确认门槛和子群规则的结果前登记。
- [`progressive_upgrade/tatqa_consensus_guarded_atomic_roles_result_v46.json`](./progressive_upgrade/tatqa_consensus_guarded_atomic_roles_result_v46.json) 与 [`../output/rag_evaluation/tatqa_consensus_guarded_atomic_roles/report.md`](../output/rag_evaluation/tatqa_consensus_guarded_atomic_roles/report.md)：TAT-QA 官方 dev 缺少精确 evidence mapping 后按协议在查询前关闭的 schema-inconclusive 记录；未创建伪 gold，也不形成方法性能结论。
- [`progressive_upgrade/fetaqa_consensus_guarded_atomic_roles_result_v47.json`](./progressive_upgrade/fetaqa_consensus_guarded_atomic_roles_result_v47.json) 与 [`../output/rag_evaluation/fetaqa_consensus_guarded_atomic_roles/report.md`](../output/rag_evaluation/fetaqa_consensus_guarded_atomic_roles/report.md)：600 例 FeTaQA supporting-cell 锁定负结果；共识候选未超过冻结 FRC 或最强非 FRC，召回与分层安全检查失败，不采用且 Gate 2 不变。
- [`progressive_upgrade/qasper_top2_proposal_guarded_atomic_roles_result_v48.json`](./progressive_upgrade/qasper_top2_proposal_guarded_atomic_roles_result_v48.json) 与 [`../output/rag_evaluation/qasper_top2_proposal_guarded_atomic_roles/report.md`](../output/rag_evaluation/qasper_top2_proposal_guarded_atomic_roles/report.md)：600 例 QASPER 全文证据选择前瞻负结果；Top-2 提案候选显著超过最强非 FRC，但低于最强冻结 FRC，召回和 includes-float 分层安全检查失败，不采用、不复用 v48 调参且 Gate 2 不变。
- [`progressive_upgrade/evidence_inference_low_core_divergence_atomic_roles_result_v49.json`](./progressive_upgrade/evidence_inference_low_core_divergence_atomic_roles_result_v49.json) 与 [`../output/rag_evaluation/evidence_inference_low_core_divergence_atomic_roles/report.md`](../output/rag_evaluation/evidence_inference_low_core_divergence_atomic_roles/report.md)：600 例 Evidence Inference 2.0 validation 全文句级证据选择前瞻负结果；低核心分歧候选显著超过最强非 FRC 且全部支持分层安全，但未超过最强冻结 FRC、召回增益未达预注册门槛，不采用、不复用 v49 调参且 Gate 2 不变。
- [`progressive_upgrade/contractnli_native_zero_consensus_abstention_result_v50.json`](./progressive_upgrade/contractnli_native_zero_consensus_abstention_result_v50.json) 与 [`../output/rag_evaluation/contractnli_native_zero_consensus_abstention/report.md`](../output/rag_evaluation/contractnli_native_zero_consensus_abstention/report.md)：600 例 ContractNLI 官方 test 均衡机制样本的完整合同跨度选择与缺失证据拒答负结果；透明 schema 勘误只剔除从未被 gold 引用的纯空白跨度，600/600 条均通过原生零共识门，拒答率为 0，故不采用、不复用 v50 拟合阈值且 Gate 2 不变。
- [`progressive_upgrade/contractnli_dev_calibrated_robust_consensus_development_result_v51.json`](./progressive_upgrade/contractnli_dev_calibrated_robust_consensus_development_result_v51.json)、[`progressive_upgrade/contractnli_dev_calibrated_robust_consensus_closure_v51.json`](./progressive_upgrade/contractnli_dev_calibrated_robust_consensus_closure_v51.json) 与 [`../output/rag_evaluation/contractnli_dev_calibrated_robust_consensus/development.md`](../output/rag_evaluation/contractnli_dev_calibrated_robust_consensus/development.md)：ContractNLI 官方 dev 的 240 例文档隔离 OOF 稳健共识阈值开发结果；utility 提升 +0.037341，但 evidence F1 损失 0.037738 超限且 NotMentioned 弃答准确率 0.1875 未达门槛，因此按协议停止在 train 之前，不采用且 Gate 2 不变。
- [`progressive_upgrade/contractnli_rank_concurrence_confirmation_result_v52.json`](./progressive_upgrade/contractnli_rank_concurrence_confirmation_result_v52.json)、[`progressive_upgrade/contractnli_rank_concurrence_confirmation_closure_v52.json`](./progressive_upgrade/contractnli_rank_concurrence_confirmation_closure_v52.json) 与 [`../output/rag_evaluation/contractnli_rank_concurrence_confirmation/report.md`](../output/rag_evaluation/contractnli_rank_concurrence_confirmation/report.md)：ContractNLI 官方 train 的 600 例一次性、角色反转无参数排名一致性确认；相对未门控 v49 显著改善，但未超过最强共享门非 FRC 且 evidence F1 损失超限，因此确认支持未建立、不得复用 train 调参、不采用且 Gate 2 不变。
- [`progressive_upgrade/cuad_top3_rank_concurrence_role_closure_development_result_v53.json`](./progressive_upgrade/cuad_top3_rank_concurrence_role_closure_development_result_v53.json)、[`progressive_upgrade/cuad_top3_rank_concurrence_role_closure_development_closure_v53.json`](./progressive_upgrade/cuad_top3_rank_concurrence_role_closure_development_closure_v53.json) 与 [`../output/rag_evaluation/cuad_top3_rank_concurrence_role_closure/development/report.md`](../output/rag_evaluation/cuad_top3_rank_concurrence_role_closure/development/report.md)：CUAD 官方 train 的 400 例、259 合同前瞻开发负结果；候选覆盖仅 0.635，Top-3 门通过 397/400，几乎不能识别无答案，且显著低于共享门最强 hybrid Top-K。按协议停止在 test 之前，不复用 train 调参、不采用且 Gate 2 不变。
- [`progressive_upgrade/twowiki_bridge_aware_precision_trim_protocol_v83.json`](./progressive_upgrade/twowiki_bridge_aware_precision_trim_protocol_v83.json)：2Wiki 桥接感知 4/5 条证据裁剪的目标前冻结模型、双阶段案例互斥、12 个对照和 18 项严格门。
- [`progressive_upgrade/twowiki_bridge_aware_precision_trim_synthesis_v83.md`](./progressive_upgrade/twowiki_bridge_aware_precision_trim_synthesis_v83.md)：开发全门通过、确认相对最强 v82 显著为正但绝对 F1 低 `0.000580` 而关闭的完整结果与禁止后验降门槛边界。
- [`../output/rag_evaluation/twowiki_precision_trim_v83/confirmation/report.md`](../output/rag_evaluation/twowiki_precision_trim_v83/confirmation/report.md)：v83 确认阶段 800 例指标、配对区间、题型非劣和唯一失败门的机器报告。
- [`progressive_upgrade/hotpot_question_type_cardinality_protocol_v84.json`](./progressive_upgrade/hotpot_question_type_cardinality_protocol_v84.json)：HotpotQA 问题类型感知 3/4 条证据基数控制的目标前冻结模型、两阶段 400/200 题型配额、九个控制和 18 项安全约束门。
- [`progressive_upgrade/hotpot_question_type_cardinality_synthesis_v84.md`](./progressive_upgrade/hotpot_question_type_cardinality_synthesis_v84.md)：开发与确认均通过全部严格门、同时完整披露原始 F1 最强但未满足完整证据召回安全线的固定前 3 条控制。
- [`../output/rag_evaluation/hotpot_cardinality_v84/confirmation/report.md`](../output/rag_evaluation/hotpot_cardinality_v84/confirmation/report.md)：v84 确认阶段 600 例的约束内优势、原始最强控制、配对区间与 Gate 2 边界机器报告。
- [`progressive_upgrade/completion_traceability_audit.md`](./progressive_upgrade/completion_traceability_audit.md)：最新设计合同的阶段、场景、切换清单、门禁和交付物逐项追溯结论。
- [`progressive_upgrade/musique_multisignal_chain_support_synthesis_v68.md`](./progressive_upgrade/musique_multisignal_chain_support_synthesis_v68.md)：MuSiQue oracle 计划九信号约束链支持门的校准、互斥开发、稳定正增益、失败门槛与确认前关闭边界。
- [`progressive_upgrade/musique_monotone_interaction_support_synthesis_v69.md`](./progressive_upgrade/musique_monotone_interaction_support_synthesis_v69.md)：MuSiQue oracle 计划单调分段与链一致性交互支持门的校准、实现勘误、互斥开发负结果和确认前关闭边界。
- [`../output/acceptance/design_contract_audit.md`](../output/acceptance/design_contract_audit.md)：最新设计 89 条显式条件的逐条证据与外部 No-Go 账本。
- [`../output/acceptance/legacy_baseline_manifest.md`](../output/acceptance/legacy_baseline_manifest.md)：升级前代码、SQLite 和 RAG 基线的离线可复现冻结报告。
- [`../output/acceptance/progressive_completion_audit.md`](../output/acceptance/progressive_completion_audit.md)：由 CI 确定性重建的机器可读完成性审计。
- [`V3_upgrade_acceptance_matrix.md`](./V3_upgrade_acceptance_matrix.md)：V3 功能证据与外部阻塞边界。
- [`openapi.json`](./openapi.json)：由运行中 FastAPI 应用导出的接口快照。
- [`../infra/postgis/README.md`](../infra/postgis/README.md)：PostGIS 影子迁移和对账说明。

## 维护规则

- 文档描述必须与当前代码、OpenAPI 或可复现报告一致。
- 生成型逐样本 JSON、DOCX 导出件、构建产物和临时研究笔记不进入版本库。
- `/response/*` 是正式响应状态唯一写入口；兼容 `/platform/*`、`/agent-twin/*` 不得绕过状态机。
- 本地模拟、自动测试和空白人工评测包不得表述为生产验收或专家结论。
