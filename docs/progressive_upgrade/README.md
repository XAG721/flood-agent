# 渐进式升级交付索引

本目录以仓库根目录《洪水预警响应系统_渐进式迭代开发与升级设计.md》为唯一升级合同，记录当前洪水预警响应任务闭环系统的实现、迁移和验收证据。

## 文档入口

- `../releases/v0.3.0.md`：当前外部接口仿真、场景目录、浏览器验收、性能、安全和兼容性更新报告。
- `asset_inventory.md`：原系统资产、真实能力、依赖、写状态风险和 REUSE/WRAP/REFACTOR/REBUILD/RETIRE 结论。
- `implementation_acceptance_matrix.md`：阶段 0—8、最终审查清单和 Go/No-Go 门禁状态。
- `completion_traceability_audit.md`：按最新设计合同汇总阶段、24 场景、第 18 节切换清单、Gate、交付物和外部 No-Go 的全目标追溯审计。
- `design_contract_evidence_policy.json`：把最新设计中的 89 条显式条件一一绑定到本地、受控、门禁或外部证据组的失败关闭策略。
- `../../output/acceptance/design_contract_audit.md`：从设计原文确定性提取并逐条核验的 89 条合同账本。
- `../../output/acceptance/legacy_baseline_manifest.md`：升级前代码、数据库重建输入、SQLite 与 RAG 索引的离线可复现冻结清单。
- `../../output/acceptance/progressive_completion_audit.md`：CI 可确定性重建的第 18—23 节机器可读审计，固定区分受控首期完成、Gate 2 No-Go 与生产外部 No-Go。
- `operations_and_rollback.md`：启动、健康检查、功能开关、备份、回滚和人工接管。
- `evaluation_and_gate_report.md`：真实运行的受控模拟指标、Gate 判定和外部 No-Go 边界。
- `frc_public_evaluation_protocol.md`：FRC-RAG 公开数据、真实模型、公平基线、配对统计和缺失/冲突/失效难例协议。
- `../../output/rag_evaluation/housing_weight_sensitivity/housing_weight_sensitivity.md`：HousingQA 冻结真实模型字段/角色权重扫描、参数辨识结果与 Gate 2 边界。
- `../../output/rag_evaluation/conformal_sufficiency/conformal_sufficiency.md`：ConditionalQA 冻结真实模型分数上的 case 分组 split-conformal 充分性拒答实验、误放行风险与效用代价。
- `../../output/rag_evaluation/conformal_robustness/conformal_robustness.md`：10 组预注册 case 分组的风险—效用稳健性、人工复核优先带与 Gate 2 边界。
- `../../output/rag_evaluation/conformal_model_selection/conformal_model_selection.md`：仅在 outer train 内进行的非线性特征/正则化嵌套选型、采用判据及负结果。
- `../../output/rag_evaluation/conformal_cross_dataset/hotpotqa_confirmation.md`：不进行确认集选型的 HotpotQA 跨数据集充分性验证及部分确认结论。
- `../../output/rag_evaluation/conformal_cross_dataset/multihoprag_confirmation.md`：同一冻结协议在 MultiHop-RAG 上的第二次独立完整确认。
- `../../output/rag_evaluation/conformal_cross_dataset/2wikimultihopqa_confirmation.md`：评分前冻结抽样与真实模型 revision 的 2WikiMultiHopQA 第三次独立完整确认。
- `../../output/rag_evaluation/conformal_cross_dataset/cross_dataset_confirmation_series.md`：三个确认集按预注册系列规则聚合后的部分确认。
- `conformal_qasc_confirmation_protocol.json`：QASC 下载前冻结的 validation 哈希、受控 40 候选池、BGE/reranker revision、无 gold 泄漏和补充确认边界。
- `../../output/rag_evaluation/conformal_qasc_confirmation/conformal_qasc_confirmation.md`：QASC 补充外部确认的严格 `NOT_CONFIRMED` 结果、协议定义偏差和不改判决策。
- `conformal_qasc_subgroup_diagnostic_protocol.json` 与 `../../output/rag_evaluation/conformal_qasc_subgroup_diagnostic/conformal_qasc_subgroup_diagnostic.md`：确认后仅按问题类型进行的描述性失效定位及重复一致性不足。
- `qasc_scoring_optimization_protocol.json` 与 `../../output/rag_evaluation/qasc_scoring_optimization/qasc_scoring_optimization.md`：冻结 32 例真实 GPU 全局批处理等价/性能基准；最高合格加速仅 1.030910 倍，结论为保留逐 case 评分入口。
- `conflicts_expected_behavior_protocol.json` 与 `../../output/rag_evaluation/conflicts_expected_behavior/PROTOCOL.md`：冻结本地生成器、无标签/方法泄漏提示、逐例 A/B 盲化、两名独立评审及独立裁决流程；当前 458 例、916 条回答只处于待人工评审状态。
- `conflicts_annotation_operations_protocol.json` 与 `../../output/rag_evaluation/conflicts_annotation_operations/PROTOCOL.md`：把三名独立人员各自的 458 个主任务和 23 个隐藏复测拆成 8 个方法盲化批次，并以 0.80 逐维精确一致率失败关闭；当前 24 个空作业批次已生成，仍无人工结果。
- `conflicts_annotation_workstation_contract.json`：冻结本机盲评 UI 的 48 个公开输入承诺、实现哈希、自动保存/刷新恢复、令牌与 Cookie 边界、`.cache` 私有草稿规则；当前只是 `LOCAL_REVIEW_UI_READY_AWAITING_HUMANS`。
- `conflicts_annotation_collection_contract.json`：冻结三角色完成回执、身份一致/互异、私有路由承诺、组内质控、旧状态失效和不解盲合并；当前真实收集状态为 `0/24`、`AWAITING_INDEPENDENT_HUMAN_BATCHES`。
- `conflicts_selector_router_protocol.json` 与 `../../output/rag_evaluation/conflicts_selector_router/conflicts_selector_router.md`：458 例回顾性 case 五折路由发现；总体 +0.050218 Accuracy，但 no-FRC 几乎相同且过时冲突 Recall 大幅退化，故不采用并保持 Gate 2。
- `whoqa_conflict_coverage_protocol.json` 与 `whoqa_conflict_coverage_execution*.json`：WhoQA 下载前方法协议、下载后结构登记、五次仅限运行时/结构修正及不可继续调参的冻结链。
- `whoqa_conflict_coverage_post_result_correction.json` 与 `../../output/rag_evaluation/whoqa_conflict_coverage/whoqa_conflict_coverage.md`：5,152 个未触碰问题、26,957 模板和六选择器的独立同预算比较；FRC 未超过 BM25 或覆盖贪心代理，结论为 `WHOQA_SELECTION_SUPPORT_NOT_ESTABLISHED`。
- `whoqa_budget_stress_protocol.json` 与 `../../output/rag_evaluation/whoqa_budget_stress/whoqa_budget_stress.md`：复用 v33 盲评分的六配置容量/Token 压力诊断；FRC family 均值显著低于 Dense，最差退化集中在 `K=4 / 256 tokens`，不在 WhoQA 上继续调参。
- `rgb_cost_aware_frc_protocol.json`、`rgb_cost_aware_frc_execution.json` 与 `../../output/rag_evaluation/rgb_cost_aware_frc/rgb_cost_aware_frc.md`：固定官方 RGB 提交上的 498 例独立盲评；成本感知 FRC 相对最强基线 -0.048719，并因最差数据集/预算退化 -0.127688 判为安全回归。
- `rgb_cost_aware_frc_post_result_correction.json`：在保留初始报告哈希后，仅修正 `positive_wrong` 次指标的统计总体，不重评分、不改主指标或安全状态。
- `musique_dual_resource_protocol.json`、`musique_dual_resource_execution.json` 与 `../../output/rag_evaluation/musique_dual_resource/musique_dual_resource.md`：固定官方 MuSiQue dev 上 2,417 例独立盲评；双资源 FRC 相对 v35 显著提高 0.004402，但相对最强非 FRC 基线为 -0.001854，故支持未建立且不采用。
- `musique_objective_gap_diagnostic_protocol.json` 与 `../../output/rag_evaluation/musique_objective_gap/musique_objective_gap.md`：v37 枚举冻结目标精确解；84.94% 配置已零 regret，精确解相对 v36 仅 +0.001399 且区间跨零。
- `musique_crossfit_support_protocol.json` 与 `../../output/rag_evaluation/musique_crossfit_support/musique_crossfit_support.md`：v38 五折 case 隔离支持校准；完整模型相对基础模型仅 +0.000548 且区间跨零，不建立 FRC 角色增量。
- `hover_verification_roles_protocol.json`、`hover_verification_roles_execution.json` 与 `../../output/rag_evaluation/hover_verification_roles/hover_verification_roles.md`：v39 HoVer 全量 4,000 例任务特定核验角色盲评；相对通用角色 -0.000035 且区间跨零，不建立任务特定角色增量。
- `../../output/rag_evaluation/hover_verification_roles/hover_role_mechanism_diagnostic.md`：v39 结果后无 gold 机制诊断，固定静态角色排序近共线与选择坍缩证据。
- `hover_dynamic_atomic_roles_v40_closure.json`：v40 在查询生成/评分前因标签与 hop 聚合意外暴露而程序性关闭；原分区不得使用。
- `hover_dynamic_atomic_roles_protocol_v41.json`、`hover_dynamic_atomic_roles_execution_v41.json`、`hover_dynamic_atomic_roles_result_v41.json` 与 `../../output/rag_evaluation/hover_dynamic_atomic_roles/hover_dynamic_atomic_roles.md`：v41 从未暴露 ID 恢复的动态原子角色实验；机制门槛通过，2,000 例确认相对静态/最强非 FRC 为 +0.001593/+0.001463，但未达实质门槛，不采用且 Gate 2 不变。
- `scifact_dynamic_atomic_roles_protocol_v42.json`、`scifact_dynamic_atomic_roles_implementation_v42.json`、`scifact_dynamic_atomic_roles_execution_v42.json`、`scifact_dynamic_atomic_roles_result_v42.json` 与 `../../output/rag_evaluation/scifact_dynamic_atomic_roles/scifact_dynamic_atomic_roles.md`：在下载 SciFact 前冻结 v41 方法的外部迁移验证；187 例完整盲评分显示动态角色机制仍有区分度，但动态选择 F1 0.403909，显著低于最强 `dense_topk` 0.460616，故支持未建立、禁止在 SciFact 上调参且 Gate 2 不变。
- `feverous_adaptive_atomic_roles_protocol_v43.json`、协议/实现/执行透明勘误、`feverous_adaptive_atomic_roles_result_v43.json` 与 `../../output/rag_evaluation/feverous_adaptive_atomic_roles/feverous_adaptive_atomic_roles.md`：v43 在官方 FEVEROUS dev challenge 六分层 240 例上完成页内候选盲评；候选完整证据覆盖率仅 0.4625，自适应证据数虽减少 2.227778，但相对动态 v41 / 最强非 FRC 分别退化 0.045874/0.036780，故状态为 `FEVEROUS_BOUNDED_POOL_INCONCLUSIVE`、禁止复用调参且 Gate 2 不变。
- `ottqa_guarded_adaptive_atomic_roles_protocol_v44.json`、协议/实现存储勘误、`ottqa_guarded_adaptive_atomic_roles_execution_v44.json`、`ottqa_guarded_adaptive_atomic_roles_result_v44.json` 与 `../../output/rag_evaluation/ottqa_guarded_adaptive_atomic_roles/ottqa_guarded_adaptive_atomic_roles.md`：v44 在接触 OTT-QA 内容前冻结一步多重性守卫，并在 oracle-table 链接证据边界内完成 240 例表格/段落平衡盲评。守卫相对动态 v41 提高 0.016093（95% CI [+0.001488,+0.029713]）且平均少选 1.125 个单元，但相对最强公平基线仍为 -0.015061，动态召回下降 0.066667，表格来源安全差值为 -0.158409；故状态为 `OTTQA_GUARDED_ADAPTIVE_SUPPORT_NOT_ESTABLISHED`、禁止复用 v44 调参且 Gate 2 不变。该结果不是 OTT-QA 官方榜单或开放域表格检索结论。
- `finqa_anchor_guarded_atomic_roles_protocol_v45.json`、路径勘误、`finqa_anchor_guarded_atomic_roles_implementation_v45.json`、`finqa_anchor_guarded_atomic_roles_execution_v45.json`、`finqa_anchor_guarded_atomic_roles_result_v45.json`、`finqa_anchor_guarded_atomic_roles_mandatory_strata_v45.json`、`finqa_anchor_equivalence_diagnostic_protocol_v45.json`、`finqa_anchor_equivalence_diagnostic_result_v45.json` 与 `../../output/rag_evaluation/finqa_anchor_guarded_atomic_roles/report.md`：v45 在接触 FinQA dev 前冻结相关性锚点守卫，并在完整记录上下文内完成表格/文本/混合各 120 例盲评。v45/v44 F1 同为 0.510429；v45 虽显著高于动态 v41 和最强非 FRC，却没有任何 v44 增量、召回损失 0.059937，且 v43 F1 更高。锁定结果后的无 gold 诊断进一步验证三档预算共 1,080 个配置的最终证据集合 100% 相同，仅选择顺序一致率为 0.866667，因而集合型指标必然相同；该诊断不授权一般等价性、调参或采用主张。状态为 `FINQA_ANCHOR_GUARDED_SUPPORT_NOT_ESTABLISHED`，不采用、不复用 v45 调参，Gate 2 不变；该结果不是 FinQA 官方端到端榜单或开放语料检索结论。
- `tatqa_consensus_guarded_atomic_roles_protocol_v46.json`、`tatqa_consensus_guarded_atomic_roles_implementation_v46.json`、`tatqa_consensus_guarded_atomic_roles_result_v46.json` 与 `../../output/rag_evaluation/tatqa_consensus_guarded_atomic_roles/report.md`：v46 只用合成反例冻结共识约束基数公式；官方 TAT-QA raw/TAGOP dev 均无精确表格单元/段落 mapping，故在查询生成前按协议关闭。未创建答案/推导伪 gold、未评分、未计算指标；`TATQA_FULL_CONTEXT_POOL_INCONCLUSIVE` 不能解释为方法正面或负面证据。
- `fetaqa_consensus_guarded_atomic_roles_protocol_v47.json`、`fetaqa_consensus_guarded_atomic_roles_implementation_v47.json`、`fetaqa_consensus_guarded_atomic_roles_execution_v47.json`、执行环境勘误、`fetaqa_consensus_guarded_atomic_roles_result_v47.json` 与 `../../output/rag_evaluation/fetaqa_consensus_guarded_atomic_roles/report.md`：v47 在 FeTaQA 官方 dev 的 600 条 SHA-256 固定样本上原样重放 v46。查询回退率 0.011667，gold 仅在完整查询/评分缓存后联结。候选 F1 为 0.358926，相对最强冻结 FRC 为 -0.016227（95% CI [-0.022628,-0.010334]），相对最强非 FRC 为 -0.007441；平均少选 1.843333 个单元但召回损失 0.085533，分层安全检查失败。状态为 `FETAQA_CONSENSUS_GUARDED_SUPPORT_NOT_ESTABLISHED`，不采用、不复用 v47 调参，Gate 2 不变；该结果不是官方答案生成榜单或开放域检索结论。
- `qasper_top2_proposal_guarded_atomic_roles_protocol_v48.json`、实现/执行登记、许可解析修订说明、执行环境勘误、`qasper_top2_proposal_guarded_atomic_roles_result_v48.json` 与 `../../output/rag_evaluation/qasper_top2_proposal_guarded_atomic_roles/report.md`：v48 只用合成不变量冻结四角色 Top-2 提案保留公式，然后在 QASPER 官方 dev 的 600 条 SHA-256 固定样本上前瞻验证。查询回退率 0.005，gold 仅在完整查询/评分缓存后联结。候选/最强冻结 FRC/最强非 FRC F1 为 0.296239/0.316885/0.247513；候选相对最强非 FRC 为 +0.048726（95% CI [+0.034631,+0.062649]），但相对最强 FRC 为 -0.020645，召回下降 0.043625，includes-float 分层退化 0.023157。状态为 `QASPER_TOP2_PROPOSAL_GUARDED_SUPPORT_NOT_ESTABLISHED`，不采用、不复用 v48 调参，Gate 2 不变；该结果不是官方答案生成榜单、开放语料检索或洪水领域结论。
- `evidence_inference_low_core_divergence_atomic_roles_protocol_v49.json`、实现/执行登记、执行超时勘误、`evidence_inference_low_core_divergence_atomic_roles_result_v49.json` 与 `../../output/rag_evaluation/evidence_inference_low_core_divergence_atomic_roles/report.md`：v49 只用理论反例和合成不变量冻结低首名核心—次名分歧守卫，然后在 Evidence Inference 2.0 官方 validation 的 600 条 SHA-256 固定样本上前瞻验证；train/test 永久排除。查询回退率 0.026667，gold 仅在完整查询/评分缓存后联结。候选/最强冻结 FRC/最强非 FRC F1 为 0.220684/0.223942/0.186549；候选相对最强非 FRC 为 +0.034135（95% CI [+0.021521,+0.046820]），所有支持分层安全检查通过，但相对最强 FRC 为 -0.003258 且召回增益 0.008710 未达到 0.01 门槛。状态为 `EVIDENCE_INFERENCE_LOW_CORE_DIVERGENCE_SUPPORT_NOT_ESTABLISHED`，不采用、不复用 v49 调参，Gate 2 不变；该结果不是 ERASER 官方结果、端到端推断榜单、开放语料检索或洪水领域结论。
- `contractnli_native_zero_consensus_abstention_protocol_v50.json`、schema/实现勘误、执行登记、`contractnli_native_zero_consensus_abstention_result_v50.json` 与 `../../output/rag_evaluation/contractnli_native_zero_consensus_abstention/report.md`：v50 冻结原始 reranker logit 严格零边界和同跨度共识门，在 ContractNLI 官方 test 的 Entailment/Contradiction/NotMentioned 各 200 条均衡样本上验证完整合同跨度选择与缺失证据拒答。首次准备在任何缓存/评分前发现 138 个不被 gold 引用的纯空白跨度；透明勘误只剔除它们并保留原编号，因此属于 schema-repair confirmation。600 条确定性查询无回退，gold 仅在完整评分缓存后联结；600/600 条全部通过门控，候选拒答率与 NotMentioned 拒答准确率均为 0。候选 utility F1 0.165173，相对最强共享门非 FRC +0.004430（95% CI [-0.015105,+0.023482]），相对最强冻结 FRC -0.001423。状态为 `CONTRACTNLI_NATIVE_ZERO_CONSENSUS_SUPPORT_NOT_ESTABLISHED`，不采用、不复用 v50 拟合阈值，Gate 2 不变；该结果不是官方 NLI 榜单、开放语料检索、真实 SetR 或洪水领域结论。
- `contractnli_dev_calibrated_robust_consensus_protocol_v51.json`、实现/顺序勘误、开发执行登记、`contractnli_dev_calibrated_robust_consensus_development_result_v51.json`、关闭记录与 `../../output/rag_evaluation/contractnli_dev_calibrated_robust_consensus/development.md`：v51 在官方 dev 的 240 条均衡样本、60 份合同和 5 个文档隔离折上开发稳健同跨度共识阈值。240 条确定性查询无回退，gold 在完整 GPU 分数缓存后连接；透明实现勘误只修正连接顺序，不改变公式或门槛。OOF utility F1 提升 +0.037341，但 evidence F1 损失 0.037738 超过 0.03，NotMentioned 弃答准确率 0.1875 未达到 0.20，故状态为 `CONTRACTNLI_V51_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TRAIN`；train 未打开、确认未启动、不采用，Gate 2 不变。
- `contractnli_rank_concurrence_confirmation_protocol_v52.json`、实现/执行登记、`contractnli_rank_concurrence_confirmation_result_v52.json`、关闭记录与 `../../output/rag_evaluation/contractnli_rank_concurrence_confirmation/report.md`：v52 在打开官方 train 前冻结无阈值同跨度排名一致性门，并把 train 作为一次性角色反转确认集。600 条/319 份合同、三标签各 200 条，查询无回退，gold 在完整评分后连接。候选 utility F1 0.250304，相对未门控 v49 +0.081783 且置信区间下界为正；但相对最强共享门非 FRC 仅 +0.009038、区间跨零，evidence F1 损失 0.082325 超限。状态为 `CONTRACTNLI_V52_RANK_CONCURRENCE_SUPPORT_NOT_ESTABLISHED`；train/v51 dev/v50 test 不得事后调参，选择器不采用，Gate 2 不变。
- `cuad_top3_rank_concurrence_role_closure_protocol_v53.json`、CUAD 来源/实现/开发执行登记、开发结果/关闭记录、结果后格式化勘误与 `../../output/rag_evaluation/cuad_top3_rank_concurrence_role_closure/development/report.md`：v53 在下载并打开 CUAD 数据前冻结 Top-3 角色交集门、角色闭包和共享门基线。官方 train 的 400 条均衡样本覆盖 259 份合同，答案/无答案各 200 条；完整盲评分后才连接 gold。答案候选完整覆盖仅 0.635，Top-3 门通过 397/400，候选无答案弃权准确率仅 0.005；候选相对共享门最强 hybrid Top-K 为 -0.023731（95% CI [-0.040711,-0.006903]），最长合同四分位退化 -0.050796。状态为 `CUAD_V53_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TEST`；test 未打开且不授权打开，train 不得复用调参，结果后格式化文件不得替换执行登记或重跑 v53，选择器不采用，Gate 2 不变。
- `doc2dial_document_contrastive_role_closure_protocol_v54.json`、来源/实现登记、`doc2dial_document_contrastive_role_closure_development_result_v54.json` 与 `../../output/rag_evaluation/doc2dial_document_contrastive_role_closure/development/report.md`：v54 在固定 Doc2Dial v1.0.1 来源上冻结答案/无答案各 200 条和 8 文档对照门。train 资格统计为 20,431 条答案样本、0 条无答案样本，故在盲缓存、查询、评分和指标前以 `DOC2DIAL_V54_SCHEMA_INCONCLUSIVE_STOP` 关闭；validation 未打开，schema 关闭不解释为方法负结果。
- `doc2dial_wood_document_contrastive_transfer_protocol_v55.json`、来源/实现登记、`doc2dial_wood_document_contrastive_transfer_development_result_v55.json` 与 `../../output/rag_evaluation/doc2dial_wood_document_contrastive_transfer/development/report.md`：v55 改用官方 v0.9 wOOD，开发成员中 3,459 段对话的 `turns` 为列表、12 段为键控对象；严格旧适配规则的 schema 排除率 0.018618 超过 0.01，故在盲评分前以 `DOC2DIAL_WOOD_V55_SCHEMA_INCONCLUSIVE_STOP` 关闭。wOOD dev 确认成员与 woOOD 兄弟成员均未打开。
- `doc2dial_wood_schema_corrected_transfer_protocol_v56.json`、来源/实现/执行登记、开发结果/关闭记录与 `../../output/rag_evaluation/doc2dial_wood_schema_corrected_transfer/development/report.md`：v56 披露 v55 train schema 访问，仅修正键控 `turns` 数值排序和官方空 reference 无答案语义，保持检索、门控、选择器、模型、预算与门槛不变。400 条均衡开发样本覆盖 386 段对话和 261 份文档，查询/评分回退率为 0；候选 utility F1 0.517417、无答案弃权准确率 0.875，但完整覆盖率仅 0.805，相对共享门最强 Dense Top-K 为 -0.008827（95% CI [-0.021324,+0.003586]），答案召回损失 0.049167，最差支持分层退化 -0.038932。状态为 `DOC2DIAL_WOOD_V56_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`；确认集未打开且不授权打开，开发样本不得复用调参，选择器不采用，Gate 2 不变。该结果不是官方共享任务、开放语料检索、真实 SetR 或洪水领域结论。
- `quac_anchor_safe_consensus_slot_protocol_v57.json`、`quac_source_registration_v57.json`、实现/执行登记、开发结果/关闭记录与 `../../output/rag_evaluation/quac_anchor_safe_consensus_slot/development/report.md`：v57 在任何 QuAC JSON 解析前冻结无参数单槽选择器，只允许在共享门控 Cross-Encoder Top-5 中保留前四项并替换第 5 项。官方 train/validation 先登记长度与哈希，随后仅打开 train；600 条均衡开发样本覆盖 578 段对话和 554 份文档，schema、查询、评分和对照文档回退率均为 0，候选覆盖上限为 1.0。候选 utility F1 0.114109、单槽触发率 0.085、无答案弃权准确率 0.066667；相对精确锚点和最强共享门控 Knapsack 分别仅 +0.000317 与 +0.000159，两个区间均跨 0。状态为 `QUAC_V57_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION`；validation 未解析且不授权打开，开发样本不得复用调参，选择器不采用，Gate 2 不变。该结果不是官方 QuAC 答案指标、开放语料检索、真实 SetR 或洪水领域结论。
- `squad2_generative_answerability_gate_protocol_v58.json` 至 `squad2_dual_support_union_protocol_v61.json`、对应实现/执行/结果/关闭记录、四份逐例证据与 `squad2_support_gate_iteration_synthesis_v61.md`：v58—v61 在 SQuAD2 train 上使用四组互斥的 600 条答案/无答案均衡样本，逐步比较二元支持门、严格抽取门、结构化 span 门和双门并集。v59 的严格解析造成 0.503333 无效输出率，v60 降至 0.016667；v61 平衡准确率 0.748333、答案通过率 0.723333、无答案拒绝率 0.773333。v60/v61 的 FRC 候选相对最强共享门非 FRC 分别显著提高 0.140103/0.143495，v61 相对最强同门 FRC 也提高 0.007778；但 v61 仍未达到 0.80 答案通过率、0.75 答案召回和 0.75 平衡准确率门槛。SQuAD2 dev 始终未解析，四轮均在确认前关闭；当前瓶颈是支持判断器，不授权采用选择器或提升 Gate 2。
- `quac_roberta_qa_support_transfer_protocol_v62.json`、模型/实现/执行登记、开发结果/关闭记录、逐例证据与 `quac_roberta_qa_support_transfer_synthesis_v62.md`：v62 固定公开 `deepset/roberta-base-squad2` revision，以模型原生 CLS 空答案分数和最优合法 span 分数形成无学习阈值的共享支持门，并迁移到与 v57 完全互斥的 600 条 QuAC train 均衡样本。支持判断平衡准确率/答案通过率/无答案拒绝率为 0.590000/0.686667/0.493333；候选 utility F1 为 0.308444，相对共享门最强非 FRC 为 -0.013077（95% CI [-0.027091,+0.001271]），相对最强同门 FRC 为 -0.013272（95% CI [-0.026309,-0.000087]）。跨域支持门和候选选择器均未建立支持；validation 未解析且不授权打开，不采用、不调参，Gate 2 保持 `NO-GO/SHADOW`。
- `quac_target_trained_qa_support_protocol_v63.json`、模型/实现/执行登记、支持结果/关闭记录、逐例盲证据与 `quac_target_trained_qa_support_synthesis_v63.md`：v63 在首次解析 QuAC validation 前固定公开目标域训练 SciBERT QA 模型和两级计算授权。600 条答案/无答案均衡 validation 样本覆盖 477 段对话和 477 份文档，QA 无效输出率和 schema 排除率均为 0；但支持判断平衡准确率/答案通过率/无答案拒绝率仅为 0.493333/0.486667/0.500000，未达到 0.75/0.75/0.65 门槛。查询生成和 BGE 检索评分按协议保持关闭，故不产生 FRC/非 FRC 效用比较。模型训练 split、checkpoint 选择和许可证未完整披露，本结果不是严格独立确认；validation 不得复用调参，不采用，Gate 2 保持 `NO-GO/SHADOW`。
- `../../output/rag_evaluation/conformal_cross_dataset/source_preparation_benchmark.json`：MultiHop-RAG 重复解析与单次解析的性能及规范哈希等价性。
- `../../output/rag_evaluation/conformal_subgroup_audit/conformal_subgroup_audit.md`：ConditionalQA、HotpotQA、MultiHop-RAG、2WikiMultiHopQA 的有限样本名义上界和可观察子群压力审计。
- `conformal_subgroup_protocol.json`：不读取 gold 的子群维度、支持度、重复一致性规则和全局/条件保证边界。
- `../../output/rag_evaluation/conformal_mondrian/conformal_mondrian.md`：分层 Mondrian 校准对全局方法的事后开发比较、采用检查和 `DO_NOT_ADOPT` 结论。
- `conformal_mondrian_protocol.json`：运行前冻结的联合分组、支持不足回退、采用门槛和独立确认边界。
- `../../output/rag_evaluation/conformal_multi_axis_mondrian/conformal_multi_axis_mondrian.md`：候选规模感知八级 Mondrian 回退在四个已揭示数据集上的事后开发负结果与停止规则。
- `conformal_multi_axis_mondrian_protocol.json`：运行前冻结的多轴层级、采用门槛，以及失败后不下载 MuSiQue 的决策边界。
- `../../output/rag_evaluation/conformal_head_transfer/conformal_head_transfer.md`：目标 train 标签隔离的四数据集 leave-one-dataset-out 参数迁移审计及 `DO_NOT_ADOPT` 结论。
- `conformal_cross_dataset_head_transfer_protocol.json`：三源集等权训练、目标 calibration-only 阈值、迁移采用门槛和失败停止规则。
- `../../output/rag_evaluation/conformal_transfer_admission/conformal_transfer_admission.md`：不读取 evaluation 的 calibration-only 跨域头准入守卫、3/3 误准入负结果和目标域拟合要求。
- `conformal_transfer_admission_protocol.json`：准入证书的 calibration 支持/AUC/召回门槛、误准入约束和失败停止规则。
- `../../output/rag_evaluation/conformal_contextual/conformal_contextual.md`：仅从 train 学习问题类型/角色数类别特征的充分性头实验及 `DO_NOT_ADOPT` 结论。
- `conformal_score_stability_protocol.json`：BM25、dense、hybrid、cross-encoder 四阶段分数与排名稳定性信号的运行前冻结协议。
- `../../output/rag_evaluation/conformal_score_stability/conformal_score_stability.md`：64 维目标域充分性头的四数据集比较、53,776 条逐 case 证据及安全门槛失败结论。
- `conformal_review_ranking_protocol.json`：不改变任何自动声明或阈值的固定工作量人工复核排序协议。
- `../../output/rag_evaluation/conformal_review_ranking/conformal_review_ranking.md`：总体 AP/捕获改善但 HotpotQA 最差子组超过退化上限的 `KEEP_BASE_REVIEW_ORDER` 审计。
- `conformal_contextual_protocol.json`：上下文 one-hot、最低训练支持、全局 calibration 阈值和采用检查。
- `conformal_cross_dataset_protocol.json`：MultiHop-RAG 运行前冻结的输入哈希、模型、alpha、分组和系列规则。
- `conformal_third_confirmation_protocol.json`：2Wiki 原始文件哈希、1000 例抽样、模型 revision、角色查询、确认门槛和子群边界的结果前登记。
- `../../infra/postgis/README.md`：SQLite 到 PostGIS 单向影子迁移、空间投影、隔离和对账演练。
- `../../output/performance/controlled_performance_report.md`：受控关键 API P50/P95、错误率和冻结预算报告。
- `dependency_security_audit.md`：Python、主前端和 Cesium 依赖漏洞清零及持续 CI 门禁。
- `api_compatibility_mapping.md`：设计概念、新 Core API 路径、旧接口边界和单写约束。

## 当前受控场景

- 区域：单一区县模拟区域；
- 预警：已有暴雨预警的模拟发布、更新与回放；
- 对象：下穿通道及可复用的重点对象登记台账；
- 下发：仅模拟 Outbox，不连接真实跨部门端点；
- 结论边界：系统辅助生成、人工审批、确定性执行、全过程可追溯，不进行洪水预测，也不替代指挥决策。

## 统一入口

- 新版核心业务 API：`/response/*`；兼容版本前缀：`/api/v1/*`。
- 旧 AgentTwin 展示入口：`/agent-twin/*`（内部兼容 `/v3/*`）。
- 旧平台入口：`/platform/*`（内部兼容 `/v2/*`）。
- 运维探针：`/health`、`/ready`、`/metrics`。

## v64 SQuAD 2.0 校准式 RoBERTa 组件验证

v64 以固定公开 RoBERTa QA revision 和单一 null-vs-span 阈值验证支持判断瓶颈。阈值只从与 v58—v61 承诺互斥的 2,000 条 train 校准样本学习，五折 OOF 平衡准确率 0.944；600 条首次打开的 dev 持出样本达到 0.873333。支持门通过后，冻结 FRC 候选相对同门精确锚点、最强同门非 FRC 和最强同门 FRC 的效用点差分别为 +0.183828、+0.171069 和 +0.006944，三者聚类 bootstrap 区间下界均高于 0。

完整协议、执行链、两次执行前勘误、结果标签勘误、逐例证据和边界说明见 [v64 综合报告](squad2_calibrated_roberta_support_synthesis_v64.md)。该结论限定为同域闭段落持出组件可行性，禁止复用确认样本调参，不授权选择器采用、CANARY、DEFAULT 或 Gate 2 提升。

## v65 MuSiQue-Full 跨域支持门压力确认

v65 把 v64 的精确阈值 `0.974609375` 零调参迁移到 600 条答案/无答案均衡的 MuSiQue-Full 多跳样本。初始运行在任何 gold 联结和指标计算前发现答案/无答案配对变体复用来源 ID，因违反“每个来源最多一条”而永久作废；修正运行以固定 SHA-256 奇偶分配答案状态，并排除无效运行全部 300 个来源承诺。最终 600 个唯一来源与 v36、SQuAD2 精确题面及无效运行来源均零重叠。

固定 RoBERTa 门的平衡准确率/答案通过率/无答案拒绝率为 `0.563333/0.630000/0.496667`；局部 span 陷阱拒绝率为 `0.449275`。支持门失败后 query 与神经检索评分未启动。完整登记链、程序性勘误、结果、逐例证据与解释边界见 [v65 综合报告](musique_full_roberta_transfer_synthesis_v65.md)。v65 不得用于后验调参或选型，不构成官方 MuSiQue、真实 SetR、独立模型训练、洪水领域或生产证据，Gate 2 保持 `NO-GO/SHADOW`。

## v66 MuSiQue 顺序链支持门

v66 排除 v65 全部 900 个来源承诺，在新的 600 条均衡样本上比较直接组合问题最大 margin 与 oracle 分解计划的顺序预测执行。链执行禁止分解答案和 supporting 标记，后续问题只使用前一跳模型预测 span；任一跳失败即整链失败关闭。顺序链将无答案拒绝率提高到 `0.816667`，但答案通过率降至 `0.390000`，平衡准确率仅 `0.603333`；相对直接门配对增益 `+0.035000`，95% CI `[-0.015000,+0.085000]`。

完整协议、来源/实现登记、导入路径勘误、执行、结果、关闭记录和逐例证据见 [v66 综合报告](musique_sequential_chain_support_synthesis_v66.md)。开发门失败后确认未打开；oracle 计划不是自动分解，不得在 v66 样本上放宽合取规则或阈值，不构成检索、真实 SetR、洪水领域或生产证据，Gate 2 保持 `NO-GO/SHADOW`。

## v67 MuSiQue 链级校准支持门

v67 在排除 v65/v66 共 1,500 个来源承诺后，用 1,000 条答案/无答案均衡新样本分别校准直接组合问题阈值和完整顺序链瓶颈阈值；链执行只在无有效 span 时失败关闭，不再应用中间 margin 门。五折 OOF 链门平衡准确率为 `0.662000`，高于直接门 `0.593000`，但没有同时满足 0.60 答案通过率和 0.80 无答案拒绝率。

冻结阈值原样应用到另 600 条互斥开发样本后，链门答案通过率/无答案拒绝率/平衡准确率为 `0.553333/0.700000/0.626667`；相对最强固定 v66 完整链基线仅 `+0.011667`，95% CI `[-0.018333,+0.041667]`。完整协议、校准/开发执行、结果、关闭记录和逐例证据见 [v67 综合报告](musique_calibrated_chain_support_synthesis_v67.md)。开发失败后确认未打开；不得在 v67 个案上新增特征或重选阈值，不构成自动分解、FRC 检索、真实 SetR、洪水领域或生产证据，Gate 2 保持 `NO-GO/SHADOW`。

## v68 MuSiQue 多信号链支持门

v68 排除 v65—v67 共 3,100 个来源承诺，以 1,200 条答案/无答案均衡新样本校准九维固定特征的投影逻辑回归。五折 OOF 候选平衡准确率为 `0.691667`，高于两信号控制 `0.663333` 和链瓶颈 `0.646667`；但答案通过率/无答案拒绝率 `0.596667/0.786667` 未同时达到安全目标。

冻结模型与阈值应用到另 600 条互斥开发样本后，候选答案通过率/无答案拒绝率/平衡准确率为 `0.573333/0.776667/0.675000`；相对最强校准链瓶颈基线 `+0.036667`，95% CI `[+0.010000,+0.063333]`。完整协议、来源/实现登记、校准/开发执行、结果、关闭记录和逐例证据见 [v68 综合报告](musique_multisignal_chain_support_synthesis_v68.md)。正增益稳定但未达预注册安全和实质门槛，确认未打开；不得复用 v68 个案选择非线性交互、特征或阈值，不构成自动分解、FRC 检索、真实 SetR、洪水领域或生产证据，Gate 2 保持 `NO-GO/SHADOW`。

## v69 MuSiQue 单调分段交互链支持门

v69 排除 v65—v68 共 4,900 个来源承诺，以 1,600 条答案/无答案均衡新样本冻结九信号线性、27 维加性分段控制和 32 维五交互候选。候选五折 OOF 平衡准确率为 `0.704375`，低于线性控制 `0.708125` 和加性控制 `0.706250`；校准只冻结参数，不授权采用。

冻结模型与阈值应用到另 800 条互斥开发样本后，交互候选答案通过率/无答案拒绝率/平衡准确率为 `0.620000/0.755000/0.687500`，弱于最强无交互加性基线 `0.690000`。候选相对最强基线及加性控制均为 `-0.002500`，95% CI `[-0.010000,+0.005000]`。完整协议、来源/实现登记及勘误、校准/开发执行、结果、关闭记录和逐例证据见 [v69 综合报告](musique_monotone_interaction_support_synthesis_v69.md)。交互增量未建立，确认未打开；不得复用 v69 个案选择交互、结点、正则或阈值，不构成自动分解、FRC 检索、真实 SetR、洪水领域或生产证据，Gate 2 保持 `NO-GO/SHADOW`。

## v76 HotpotQA 图与分数软闭包覆盖路由

v76 把 71 个无 gold 图结构、冻结分数、选择重叠和 v75 问题概率特征输入 25 棵深度 2 的梯度提升树，只在预测软标题闭包相对 cross-encoder 的证据 F1 增益大于 0 时覆盖。模型开发使用 1,000 条永久排除的 HotpotQA 历史案例并披露 1,089 个有限配置；目标开发和确认各 800 例，bridge/comparison 各 400 例，历史/开发/确认三者 ID 零重叠。

开发/确认候选 F1 为 `0.568959/0.570916`，相对各阶段最强登记对照提高 `+0.008119/+0.006460`，95% CI 为 `[+0.003859,+0.012623]` 与 `[+0.002201,+0.010864]`；完整证据召回为 `0.775000/0.787500`，13 项门槛连续两次全部通过。完整协议、来源、历史模型、实现锁、执行、结果和边界见 [v76 综合报告](hotpot_graph_router_synthesis_v76.md)。该结果独立于 v75 的 2Wiki 数据，但 v76 训练历史和目标仍同属 HotpotQA，只建立案例隔离重复支持；不构成独立训练数据、官方榜单、答案生成、真实 SetR、洪水专家或生产证据，选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。

## v77 MuSiQue 冻结 v76 路由器零调参迁移

v77 排除 v65—v73 的 16,900 个来源承诺，从 3,038 个未触碰 MuSiQue answerable 来源中按 2/3/4-hop=`400/250/150` 选择 800 例开发集。完整 v76 模型、段落适配、六个控制、严格优势门槛和非劣包络均在 ID 选择前锁定，目标训练或调参案例为 0。

候选 F1 `0.535779`，相对 cross-encoder 提高 `+0.007426`（95% CI `[+0.002535,+0.012416]`）；但最强交替锚点控制 F1 为 `0.542004`，候选相对它 `-0.006225`（95% CI `[-0.012733,+0.000164]`），3-hop/4-hop 分层退化 `-0.009000/-0.016296`。严格优势和次级非劣包络均失败，确认集未打开。完整协议、来源登记、实现锁、执行、结果、关闭和边界见 [v77 综合报告](musique_graph_router_transfer_synthesis_v77.md)。该结果限制 v76 的跨数据集外推，不能改写为普适迁移、真实 SetR、洪水专家或生产证据；选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。

## v78 MuSiQue 均值校准三路路由

v78 先在永久排除的 1,000 条 HotpotQA 历史上搜索 432 个双增益路由策略；最佳 OOF F1 `0.563008` 仅比 v76 高 `+0.000560`，且锚点路由为 0。随后把 v77 的 800 例明确作为目标域总体均值校准集，只登记一个无 hop、无个案条件的截距修正规则，并永久排除这些来源。

新的 600 例开发集按 2/3/4-hop=`300/200/100` 从剩余未触碰来源中锁定。候选 F1 `0.546382`，相对 v76 `+0.000496`（95% CI `[-0.006369,+0.007388]`），相对最强交替锚点 `-0.001700`（95% CI `[-0.006865,+0.003572]`）；全部非劣条件通过，但严格优势、F1 下限和 4-hop 安全门失败，确认未打开。完整边界见 [v78 综合报告](musique_mean_calibrated_three_route_synthesis_v78.md)。v78 不是零目标调参或独立训练数据确认，选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。

## v79 MuSiQue 目标域逐例三路路由

v79 只使用已冻结并永久排除的 v78 600 条证据训练两个目标域增益头。71 个运行时无 gold 特征、108 个模型配置和 4 个阈值形成 432 个有限策略；选中策略的五折 OOF F1 为 `0.551733`，相对 v78 为 `+0.005351`，但区间跨零且该分数参与选型，因此不是独立确认。

在模型、阈值、实现、470 例规模、2/3/4-hop=`250/150/70` 配额和严格门槛全部锁定后，另取与 v65—v78 零重叠的 470 条样本。候选 F1 `0.553723`、完整证据召回 `0.597872`，相对 v78 `+0.005961`（95% CI `[-0.000380,+0.012606]`），相对最强交替锚点 `+0.002871`（95% CI `[-0.002187,+0.007844]`）；三路覆盖、逐跳安全、成本和非劣包络通过，但严格优势区间门失败，确认未打开。完整边界见 [v79 综合报告](musique_target_three_route_synthesis_v79.md)。v79 是同数据集案例隔离的新样本验证，不是零目标调参、独立训练数据、官方榜单、真实 SetR 或防汛专家证据；选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。

## v80 MuSiQue 锚点默认三路由终局留出

v80 合并永久排除的 v78 600 条和 v79 470 条目标域证据，以交替锚点为默认路线，分别学习 cross-encoder 与软闭包相对锚点的增益。216 个模型配置、三种样本权重和 7 个阈值形成 1,512 个有限策略；选中策略的五折 OOF F1 为 `0.554843`，相对锚点 `+0.005544`，但它参与选型，只是开发诊断。

排除 v65—v79 共 18,770 个历史 ID 后，剩余 4-hop 只有 73 条，因此协议在任何 v80 目标 ID 前就冻结为一次性 600 例终局留出，2/3/4-hop=`400/150/50`，不再登记第二个同源确认阶段。终局集与全部历史暴露重叠为 0，600/600 条完成冻结 GPU 盲评分。候选 F1 `0.527817`、完整证据召回 `0.643333`，相对最强 v79 `+0.001561`（95% CI `[-0.002533,+0.005774]`）；三路覆盖和全部非劣条件通过，但绝对 F1、主增益/区间及 3-hop `-0.006667` 严格门失败。完整边界见 [v80 综合报告](musique_anchor_default_terminal_holdout_synthesis_v80.md)。v80 终局样本禁止回用于修模或改门槛，选择器/CANARY/DEFAULT 不授权，Gate 2 保持 `NO-GO/SHADOW`。

## v81 2Wiki 残差三路由前瞻开发

v81 以冻结 v75 路由为默认动作，只学习 cross 覆盖与软/锚翻转两个残差动作。模型开发仅使用永久排除的 v75 开发/确认 1,600 例；73 个运行时无 gold 特征、216 个模型配置和 7 个阈值形成 1,512 个有限策略。选中 OOF F1 为 `0.540164`，相对 v75 `+0.005913`（95% CI `[+0.004067,+0.007857]`），只作为模型选择诊断。

协议、源登记、模型、代码、门槛、盐值和测试在目标选择前完成八文件锁。开发集从剩余 2Wiki 容量中按四题型各取 200 例，与历史 1,000、v74 800、v75 开发 800、v75 确认 800 共 3,400 个 ID 零重叠；800/800 条完成冻结 GPU 盲评分。候选 F1 `0.547195`、完整证据召回 `0.658750`，相对最强 v75 `+0.004444`（95% CI `[+0.001706,+0.007381]`）。区间为正且非劣包络通过，但点增益未达到 `+0.005`，comparison 相对题型最佳对照为 `-0.008571`；严格门失败，确认阶段未打开。完整边界见 [v81 综合报告](twowiki_residual_three_route_synthesis_v81.md)。Gate 2 保持 `NO-GO/SHADOW`。

## v82 2Wiki 题型级联与安全残差路由前瞻开发

v82 将 v81 开发集明确转为训练/诊断历史；2400条永久排除样本中的 v81 增益预测逐例来自 held-out 折模型或未见该批次的完整模型。路由器先以问题文本识别直接 comparison 并走 cross，其余样本只允许 v81 的软/锚翻转。16个分类器配置与5×7个双阈值形成560个有限策略；选中样本外 F1 `0.543817`，相对 v75 `+0.006733`（95% CI `[+0.005079,+0.008505]`），三个训练批次均为正，但只作为模型选择诊断。

协议、源登记、模型、代码、门槛和测试在目标选择前完成9文件锁。开发集四题型各200例，与历史池、v74、v75两阶段和 v81 开发共4200个 ID 零重叠；25305个候选完成800/800条冻结 GPU 盲评分。候选 F1 `0.544792`、完整证据召回 `0.643750`，相对 v75 `+0.006508`（95% CI `[+0.003373,+0.009762]`），bridge comparison 与 comparison 均和各自最佳对照持平，分层安全全部通过；但最强登记对照为 v81，候选相对它只有 `+0.002143`（95% CI `[+0.000357,+0.004286]`），未达到 `+0.005`。16项严格门仅此项失败，确认阶段未打开。完整边界见 [v82 综合报告](twowiki_cascaded_style_residual_synthesis_v82.md)。Gate 2 保持 `NO-GO/SHADOW`。

## v83 2Wiki 桥接感知精度裁剪前瞻验证

v83 固定 v82 的路由和证据顺序，只用问题文本分类器识别 bridge-comparison；bridge 概率达到 `0.3` 时保留前 4 条，否则保留全部 5 条。模型开发使用永久排除的 v75 开发、v75 确认、v81 开发和 v82 开发共 3,200 例；16 个分类器配置和 112 个有限策略中选中维度 `64`、L2 `32.0`。五折样本外候选 F1 `0.554477`，相对冻结 v82 为 `+0.009652`（95% CI `[+0.007831,+0.011496]`），但该结果参与选型，不是确认。

九文件锁后，开发与确认各 800 例、四题型各 200 例，二者及 5,000 个历史 ID 零重叠。开发 F1 `0.556329`，相对最强 v81 为 `+0.011373`（95% CI `[+0.007802,+0.014985]`），18 项严格门全过并打开确认。确认 F1 `0.549420`，相对最强 v82 为 `+0.006434`（95% CI `[+0.002465,+0.010253]`），但未达到绝对 F1 下限 `0.55`；其余 17 项严格门和全部非劣门通过。最终状态为 `2WIKI_V83_PRECISION_TRIM_CONFIRMATION_ADVANTAGE_NOT_ESTABLISHED`，不得降低门槛或复用确认样本。完整证据见 [v83 综合报告](twowiki_bridge_aware_precision_trim_synthesis_v83.md)；选择器、CANARY、DEFAULT 不授权，Gate 2 保持 `NO-GO/SHADOW`。

## v84 HotpotQA 问题类型感知证据基数前瞻验证

v84 完全冻结 v76 的路由与证据顺序，只用问题文本判断 comparison 概率：概率不低于 `0.3` 时保留前 3 条，否则保留前 4 条。模型开发仅使用永久排除的 HotpotQA 历史 1,000 例及 v76 两阶段各 800 例；16 个分类器配置和 112 个有限策略选中维度 `512`、L2 `32.0`。交叉拟合候选 F1/完整证据召回 `0.657218/0.645385`，相对固定前 4 条 `+0.027613`（95% CI `[+0.024012,+0.031077]`），只作为选型证据。

九文件锁后，开发和确认各 600 例（bridge 400、comparison 200），二者及 2,600 个训练 ID 零重叠。开发候选 F1/完整证据召回 `0.644873/0.643333`，相对最强安全合格控制 `+0.029039`（95% CI `[+0.022617,+0.035716]`）；确认分别为 `0.639898/0.625000`，增益 `+0.026492`（95% CI `[+0.019841,+0.033051]`），两阶段全部 18 项严格门通过。固定前 3 条的原始 F1 更高，但两个阶段完整证据召回均为 `0.571667`，低于预注册 `0.62` 安全线，因此只支持约束内优势。完整边界见 [v84 综合报告](hotpot_question_type_cardinality_synthesis_v84.md)；选择器、CANARY、DEFAULT 不授权，Gate 2 保持 `NO-GO/SHADOW`。

## v85—v89 IIRC 证据基数路由与鲁棒复制

v85 先尝试把 BeerQA 作为独立迁移来源，但官方 dev 只含既有 SQuAD/HotpotQA 派生记录，没有可独立评测的 3+ hop 新标注子集，因此在评分前关闭。v86 随后登记 IIRC 原始来源并把冻结 HotpotQA 基数模型零调参迁移到 400 条新样本；候选 F1 `0.616381`，相对 `frc_fixed2` 为 `-0.004512`（95% CI `[-0.018155,+0.001346]`），否定了直接跨数据集基数迁移优势。

v87 只用已揭示的 v86 开发集训练 154 维 IIRC 分数差路由器，在另 400 条样本上取得 F1 `0.616637`，相对 `frc_fixed2` `+0.007831`，但区间跨 0，确认未打开。v88 汇聚 v86/v87 共 800 条永久排除训练历史，冻结 ExtraTrees 模型；新 800 条开发集 F1 `0.634559`，增益 `+0.028773`（95% CI `[+0.015490,+0.041769]`），全部门禁通过，但确认准备发现一条仅有 3 个真实候选源的样本，旧适配器要求至少 4 个候选，因而在写缓存和计算指标前失败关闭。

v89 不重训模型，只把适配器改为“候选不少于 4 时运行原 v88 模型，1—3 时选择所有真实源；控制前缀上限取实际可用源数”，且禁止丢样本、替换或合成填充。新的开发/确认集为 800/1200 条，与全部历史及彼此零重叠。开发候选 F1 `0.636527`，相对 `frc_fixed2` `+0.030402`（95% CI `[+0.017976,+0.042557]`）；确认 F1 `0.627097`，增益 `+0.028902`（95% CI `[+0.017896,+0.039656]`）。两个阶段全部严格门禁通过，且相对每个登记对照的点差均为正，建立了冻结 v88 模型在 IIRC 新案例上的同数据集复制支持。完整链路见 [v85—v89 综合报告](iirc_cardinality_transfer_iteration_synthesis_v89.md)。该结果不包含答案生成、独立训练数据迁移、真实 SetR 或防汛专家验证；选择器、CANARY、DEFAULT 不授权，Gate 2 继续 `NO-GO/SHADOW`。
