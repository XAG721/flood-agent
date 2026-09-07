# 基于多源时空语义关联与 FRC-RAG 的洪水预警响应系统

[![CI](https://github.com/XAG721/flood-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/XAG721/flood-agent/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.3.2-2563eb)
![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-149eca?logo=react&logoColor=white)

面向区（县）防汛抗旱指挥部办公室，将专业部门发布的预警与风险对象、时空位置、责任台账、预案条款和执行反馈进行关联，形成“对象可确认、任务有依据、过程可审批、结果可核验”的洪水预警响应闭环。

系统研究重点包括两部分：

- **多源时空语义关联**：把预警时空范围、对象坐标与属性、台账有效期、文档版本和业务语义统一到对象级响应事件中；
- **FRC-RAG**：以任务字段和功能角色覆盖为约束组织证据集合，显式处理证据冲突、缺失、失效与跨文档关系，为结构化任务草案提供可定位、可比较、可冻结的依据。

> [!IMPORTANT]
> 当前 `v0.3.2` 已完成单一区域、暴雨预警类型下的**受控模拟闭环**，并完成三轮模块边界重构；不进行洪水预测，也不替代防汛指挥决策。FRC-RAG 的工程链路与公开数据实验可复现，v89 已在 IIRC 案例互斥开发/确认集上建立源选择优势复制支持；但尚未完成独立训练数据迁移、端到端答案、真实 SetR 与防汛专家验证，Gate 2 保持 `NO-GO / SHADOW`。真实生产部署仍需权威数据、机构 UAT、安全与基础设施验收。

## 研究问题与系统目标

传统预警系统往往停留在信息展示或统一通知层面，难以回答四个直接影响执行的问题：

1. 预警具体影响哪些对象，候选结果为什么被选中？
2. 针对每个对象应由谁、在何时、执行什么动作？
3. 任务依据来自哪个版本、哪一条款，是否存在冲突或缺失？
4. 下发后是否接收、执行、反馈并经过独立核验？

本系统以响应事件为主线，将上述问题组织为确定性业务闭环：

```text
专业预警
  -> 响应事件与不可变预警快照
  -> 多源时空语义关联与候选对象筛查
  -> 人工确认并冻结对象清单
  -> FRC-RAG 九字段证据包
  -> 结构化任务草案与规则校验
  -> 授权人员审批
  -> 模拟下发、接收、执行与反馈
  -> 独立核验、事件关闭与复盘
```

AI 与检索模块只能生成候选对象、证据和草案，不能审批、下发或直接改变正式业务状态。智能服务不可用时，人工证据组装与确定性工作流仍可继续运行。

## 核心技术

### 1. 多源时空语义关联

系统关联预警、风险对象主数据、地理位置、责任关系、预案文档和事件反馈，并保留每一次关联所使用的数据版本。

- 预警保存区域、时间、等级、版本和可选 EPSG:4326 影响范围；
- 风险对象支持 API、UTF-8 CSV、受限 XLSX、JSON 与 Point GeoJSON 导入；
- 候选筛查综合空间、时效、属性、语义和数据质量五类特征，并保存逐候选解释快照；
- 缺少坐标的对象不会冒充空间命中，系统明确记录排除原因或降级为区域台账关联；
- 主数据发生实质变化后，相关候选运行与事件对象进入 `STALE`，重新筛查和人工核验前禁止成案；
- 已确认对象冻结为不可变 `CandidateObjectListVersion`，任务必须绑定清单 ID、版本与内容哈希。

这一机制把“模型给出的风险分”转化为可复核的数据关联过程，候选对象始终需要业务人员确认后才能成为正式处置对象。

### 2. FRC-RAG 证据组织

FRC-RAG（Functional Role Coverage-guided Retrieval-Augmented Generation，功能角色覆盖约束的检索增强生成）不是单一重排器，而是一条从查询分解到证据冻结的完整链路：

```text
任务 Schema
  -> 九字段查询分解
  -> BM25 / Dense / Hybrid 候选召回与重排
  -> 字段支持、功能角色、可信度、适用性与新颖性评分
  -> 冲突惩罚与证据预算约束
  -> SUPPORTED / CONFLICTED / MISSING 三态证据矩阵
  -> 人工裁决、版本比较与证据包冻结
```

证据包覆盖九个任务字段：

| 字段 | 业务含义 |
|---|---|
| `trigger_condition` | 触发条件、预警等级、阈值和适用范围 |
| `risk_object` | 风险对象、位置、脆弱性和影响范围 |
| `responsible_party` | 责任单位、责任岗位、权限和协同主体 |
| `action` | 处置动作、执行步骤、先后顺序和操作要求 |
| `deadline` | 接收、开始、完成、核验时限和数值阈值 |
| `resource_dependency` | 人员、车辆、设备、物资和资源依赖 |
| `feedback_requirement` | 反馈内容、附件、位置、时间和证据要求 |
| `escalation_condition` | 催办、升级、改派和人工接管条件 |
| `exception_condition` | 例外、受阻、终止、替代和特殊情况 |

每条证据保存来源、文档版本、条款、页码、章节路径、表格行和原文定位。核心字段缺失或存在未裁决冲突时阻断审批；资源依赖和例外条件可保持 `MISSING`，但必须给出明确原因，系统不会自动补造条款。

冲突由确定性规则、显式来源关系和可选 NLI 适配器产生候选，最终采用哪个来源必须由人工裁决。默认 NLI 状态为 `nli-unavailable`，不会把未部署的模型伪装成可用能力。

### 3. 确定性响应工作流

- 预警发布、更新、撤销与对象重新核验；
- 任务创建、提交、三档规则校验、职责分离审批与载荷哈希绑定；
- 接收、开始、完成、核验四类时限及催办、升级、改派和人工接管；
- 受阻、资源不足、部分完成、延期、撤回和核验退回整改分支；
- 正常、超时、拒收、部分成功、重复、乱序六类模拟下发场景；
- 回调只追加证据与审计记录，不能直接修改正式任务状态；
- 事件关闭、时间线完整性校验、审计归档和复盘草稿。

### 4. 安全、审计与恢复

- `/response/*` 接受可信身份网关签名的短时身份断言，高风险操作强制 AAL2；
- RBAC 与职责分离阻止越权操作和高风险任务自审；
- 响应域敏感载荷使用 Fernet 认证加密，密钥支持受控轮换；
- 事件台账形成 SHA-256 哈希链，关键事实由数据库触发器禁止更新或删除；
- Core API 写请求由身份绑定的通用幂等账本保护，处理并发、重放和未知写入结果；
- 备份清单使用独立 HMAC 签名，恢复前验证摘要、完整性、密钥标识和实际解密能力；
- 精确位置、联系方式等信息按岗位脱敏，外部模型调用经过个人信息出站过滤；
- 模拟网关只接受 `simulated://` 目标，拒绝连接真实外部下发端点。

## 系统架构

```text
┌──────────────── React 响应工作台 / Cesium 可视化 ────────────────┐
│  事件总览 · 对象核验 · 证据工作台 · 审批 · 执行 · 复盘          │
└──────────────────────────┬───────────────────────────────────────┘
                           │ /response/*  (/api/v1/* 兼容前缀)
┌──────────────────────────▼───────────────────────────────────────┐
│                         FastAPI Core API                         │
│ 身份与幂等边界 · 状态机 · 对象关联 · 文档治理 · FRC-RAG · 审计 │
└───────────────┬──────────────────┬──────────────────┬────────────┘
                │                  │                  │
      ┌─────────▼────────┐ ┌───────▼────────┐ ┌──────▼───────────┐
      │ 加密 SQLite 权威源 │ │ Worker / Outbox │ │ PostGIS 影子投影 │
      │ 版本、任务、证据   │ │ 模拟下发与巡检   │ │ 仅迁移对账，不切流 │
      └──────────────────┘ └────────────────┘ └──────────────────┘
```

接口边界：

- `/response/*`：新版响应域 Core API，也是正式响应状态的唯一写入口；
- `/api/v1/*`：Core API 的兼容版本前缀；
- `/platform/*`、`/agent-twin/*`：旧平台与数字孪生兼容展示层，不得绕过新版状态机写入响应域；
- `/health`、`/ready`、`/metrics`：存活、就绪与运行指标；
- SQLite 当前仍是权威状态源，PostGIS 只用于单向影子迁移和对账。

## 快速开始

### 方式一：Docker Compose 受控模拟环境

需要 Docker Desktop 或兼容的 Docker Compose。仓库根目录执行：

```powershell
docker compose up --build
```

启动后访问：

- 响应工作台：<http://127.0.0.1:8080/response>
- 后端 OpenAPI：<http://127.0.0.1:8000/docs>
- 就绪检查：<http://127.0.0.1:8000/ready>

Compose 会启动 `frontend`、`backend`、`worker` 和模拟专用 `postgis`。其中的身份、数据库和下发配置仅用于本地受控模拟，不能直接用于生产。

### 方式二：Windows 一键演示

需要 Python 3.12、Node.js 22，并先安装依赖：

```powershell
python -m pip install -e ".[test,postgres]"
npm.cmd ci --prefix frontend
```

然后运行：

```powershell
.\scripts\start-demo.ps1
```

脚本会重建并检查演示数据库，启动后端 `http://127.0.0.1:8000` 与前端 `http://127.0.0.1:5173`，并默认启用固定演示快照。常用参数：

```powershell
.\scripts\start-demo.ps1 -SkipRebuild   # 保留现有演示库
.\scripts\start-demo.ps1 -LiveData      # 关闭前端固定演示态
.\scripts\start-demo.ps1 -NoBrowser     # 不自动打开浏览器
```

### 本地开发

```powershell
python -m pip install -e ".[test,postgres]"
python scripts/rebuild_demo_db.py --force
$env:FLOOD_DB_PATH = "$PWD\data\flood_warning_system_demo.db"
python -m uvicorn flood_system.api:app --host 127.0.0.1 --port 8000
```

在另一个终端启动前端：

```powershell
npm.cmd ci --prefix frontend
$env:VITE_DEMO_MODE = "true"
npm.cmd run dev --prefix frontend
```

生产模式缺少数据加密密钥、身份断言密钥或备份清单密钥时会拒绝启动。完整的身份、密钥轮换、备份恢复和回滚要求见[运维与回滚手册](docs/progressive_upgrade/operations_and_rollback.md)及[安全与事件处置手册](docs/progressive_upgrade/security_and_incident_manual.md)。

## 复现实验与验证

在另一台电脑恢复代码、Git LFS 三维场景、Python/CUDA 环境和冻结模型 revision 时，先阅读[跨电脑续研说明](docs/experiment_resume.md)。密钥、CONFLICTS 解盲映射和人工评审草稿不进入公开仓库。

日常代码验证：

```powershell
python -m pytest -q --basetemp .pytest-tmp/readme
python -m ruff check flood_system scripts tests
npm.cmd test --prefix frontend -- --run
npm.cmd run build --prefix frontend
npm.cmd run build --prefix 3D_visual
```

重建受控验收证据：

```powershell
python scripts/generate_floodagent_bench.py
python scripts/run_candidate_evaluation.py
python scripts/run_rag_evaluation.py
python scripts/run_controlled_performance.py --db tmp/performance.db --output-dir output/performance
python scripts/run_design_contract_audit.py
python scripts/run_progressive_completion_audit.py
```

公开数据、真实神经评分器、消融、敏感性分析和人工标注流程见 [FRC-RAG 公开评测协议](docs/progressive_upgrade/frc_public_evaluation_protocol.md)。分层校准规则与负结果见 [Mondrian 协议](docs/progressive_upgrade/conformal_mondrian_protocol.json)和[报告](output/rag_evaluation/conformal_mondrian/conformal_mondrian.md)；train-only 上下文头的冻结规则与负结果见[上下文协议](docs/progressive_upgrade/conformal_contextual_protocol.json)和[报告](output/rag_evaluation/conformal_contextual/conformal_contextual.md)；多阶段检索分数稳定性信号见[冻结协议](docs/progressive_upgrade/conformal_score_stability_protocol.json)和[报告](output/rag_evaluation/conformal_score_stability/conformal_score_stability.md)；不改自动边界的人工复核排序审计见[复核协议](docs/progressive_upgrade/conformal_review_ranking_protocol.json)和[结果](output/rag_evaluation/conformal_review_ranking/conformal_review_ranking.md)。仓库不会把小型工程集、空白标注包、事后方法开发或跨领域实验表述为真实防汛领域专家结论。

最新 v58—v61 SQuAD2 支持门研究使用四组互斥的答案/无答案均衡样本。结构化 span 输出把无效输出率从 v59 的 50.33% 降至 v60 的 1.67%；v61 双支持并集下，FRC 候选相对最强共享门非 FRC 提高 14.35 个百分点（95% CI [+12.04,+16.70]），相对最强同门 FRC 提高 0.78 个百分点（95% CI [+0.40,+1.20]）。但支持判断器的平衡准确率 74.83%、答案通过率 72.33%和答案召回 65.50%仍低于预注册门槛，因此确认集未打开，Gate 2 仍为 `NO-GO/SHADOW`。完整迭代边界见 [v61 综合报告](docs/progressive_upgrade/squad2_support_gate_iteration_synthesis_v61.md)。

v62 随后把公开 `deepset/roberta-base-squad2` 的固定 revision 作为模型原生空答案/合法 span 支持门，零 QuAC 微调、零阈值拟合地迁移到与 v57 完全互斥的 600 条 QuAC train 样本。支持判断平衡准确率仅 59.00%，答案通过率 68.67%、无答案拒绝率 49.33%；FRC 候选 utility F1 为 0.308444，相对同门最强非 FRC 为 −0.013077（95% CI [−0.027091,+0.001271]），相对最强同门 FRC 为 −0.013272（95% CI [−0.026309,−0.000087]）。这表明 SQuAD2 QA 模型不能直接跨域充当 QuAC 支持门；validation 未打开，选择器与支持门均不采用，Gate 2 仍为 `NO-GO/SHADOW`。详见 [v62 综合报告](docs/progressive_upgrade/quac_roberta_qa_support_transfer_synthesis_v62.md)。

v63 再固定公开声明曾在 SQuAD2.0 和 QuAC 上训练的 `ixa-ehu/SciBERT-SQuAD-QuAC` revision，在首次解析 QuAC validation 前完成协议、代码和模型登记，并先运行独立于检索的 600 条均衡盲支持门。平衡准确率仅 49.33%，答案通过率 48.67%、无答案拒绝率 50.00%，未达到 75%/75%/65% 的预注册门槛；因此查询生成和 BGE 检索评分均未获授权，不能形成 v63 的 FRC/非 FRC 效用比较。该公开模型的精确训练 split、checkpoint 选择和许可证未完整披露，结果只作为来源受限的负向可行性证据；不调参、不采用，Gate 2 保持 `NO-GO/SHADOW`。详见 [v63 综合报告](docs/progressive_upgrade/quac_target_trained_qa_support_synthesis_v63.md)。

v64 在与 v58—v61 承诺互斥的 2,000 条 SQuAD2 train 样本上完成单阈值五折校准，并把精确阈值 `0.974609375` 原样应用到首次打开的 600 条 dev 持出样本；支持判断平衡准确率由 OOF 的 0.944000 到确认集的 0.873333。支持门通过后，冻结 FRC 候选效用 F1 为 0.807944，相对同门最强非 FRC 为 +0.171069（95% CI [+0.146246,+0.194769]）。该结果只建立同域闭段落组件可行性，不是独立模型训练确认，且不授权选择器或 Gate 2 提升。详见 [v64 综合报告](docs/progressive_upgrade/squad2_calibrated_roberta_support_synthesis_v64.md)。

v65 进一步把 v64 阈值零调参迁移到 MuSiQue-Full 的 600 条答案/无答案均衡多跳样本。修正后的 600 个唯一来源与无效初始运行、v36 和 SQuAD2 精确题面均零重叠；完整 QA 缓存后才连接 gold。平衡准确率仅 0.563333，答案通过率 0.630000、无答案拒绝率 0.496667；“存在局部 span 但链断裂”的 69 例拒绝率仅 0.449275。支持门失败后 query 与 BGE/reranker 评分未启动，不能形成 FRC/非 FRC 检索比较；该结果表明高置信局部 span 不能替代多跳链完整性判断。详见 [v65 综合报告](docs/progressive_upgrade/musique_full_roberta_transfer_synthesis_v65.md)。

v66 随后在与 v65 的 900 个来源承诺完全互斥的新样本上验证顺序链支持门：官方分解问题只作为 oracle 计划，执行器禁止读取分解答案和 supporting 标记，后续 `#k` 只能替换为模型前一跳预测 span，并要求所有跳严格超过同一固定阈值。无答案拒绝率从直接门的 0.526667 提高到 0.816667，直接门误放行案例的 69.72% 被链门拒绝；但答案通过率从 0.61 降至 0.39，平衡准确率仅从 0.568333 升到 0.603333，配对增益 +0.035（95% CI [-0.015,+0.085]）。严格全跳合取过度保守，开发门失败并停止在确认前；不调参、不采用，Gate 2 保持 `NO-GO/SHADOW`。详见 [v66 综合报告](docs/progressive_upgrade/musique_sequential_chain_support_synthesis_v66.md)。

v67 在继续排除 v65/v66 共 1,500 个来源承诺后，用 1,000 条新样本只校准完整预测链的单一瓶颈阈值，再原样应用到另 600 条互斥开发样本。链执行不再因中间 margin 偏低提前终止，答案通过率由固定 v66 完整链的 0.393333 提高到 0.553333，但无答案拒绝率由 0.836667 降到 0.700000；平衡准确率仅由 0.615000 升到 0.626667，相对最强公平基线 +0.011667（95% CI [-0.018333,+0.041667]）。单一链分数无法区分完整链中的跨域低置信 hop 与真正缺失 hop，开发门失败且确认未打开；不在 v67 个案上追加拟合、不采用，Gate 2 保持 `NO-GO/SHADOW`。详见 [v67 综合报告](docs/progressive_upgrade/musique_calibrated_chain_support_synthesis_v67.md)。

v68 永久排除 v65—v67 共 3,100 个来源承诺，在 1,200 条新校准样本上冻结九维链特征、非负证据信号约束逻辑回归及同优化器两信号控制，再原样应用到另 600 条互斥开发样本。候选在开发集达到 `0.573333/0.776667/0.675000` 的答案通过率、无答案拒绝率和平衡准确率；相对最强校准链瓶颈基线提高 `+0.036667`，95% CI `[+0.010000,+0.063333]`，说明多信号机制有稳定正增益，但仍未达到 0.72/0.60/0.80 和 +0.05 的预注册门槛。确认未打开，v68 个案不得用于后验交互、特征或阈值选择，Gate 2 继续 `NO-GO/SHADOW`。详见 [v68 综合报告](docs/progressive_upgrade/musique_multisignal_chain_support_synthesis_v68.md)。

v69 永久排除 v65—v68 共 4,900 个来源承诺，在 1,600 条新校准样本上冻结 18 个单调分段基、5 个链一致性交互及同优化器线性/无交互加性控制，再原样应用到另 800 条互斥开发样本。交互候选的答案通过率、无答案拒绝率和平衡准确率为 `0.620000/0.755000/0.687500`，弱于最强无交互加性基线的 `0.615000/0.765000/0.690000`；两项同例增益均为 `-0.002500`，95% CI `[-0.010000,+0.005000]`。这否定了本次五个固定乘性交互的增量假设，确认未打开，v69 的 2,400 个校准/开发案例不得用于后验交互、结点、正则或阈值选择，Gate 2 继续 `NO-GO/SHADOW`。详见 [v69 综合报告](docs/progressive_upgrade/musique_monotone_interaction_support_synthesis_v69.md)。

v74 将研究切换到 2WikiMultiHopQA，并永久排除此前已揭示的 1,000 个案例。在四类各 200 例的新开发集上，软标题链接路径闭包的证据 F1 为 `0.518988`，相对现有 FRC-Select 提高 `+0.076095`（95% CI `[+0.063476,+0.088913]`）；但相对最强同资源交替锚点对照仅 `+0.001718`，区间跨 0，且两类 comparison 问题退化约 4.4 个百分点。确认集因此未打开，结论限定为“标题图显著改善旧 FRC，但软闭包不优于最强图对照”。详见 [v74 综合报告](docs/progressive_upgrade/twowiki_support_path_closure_synthesis_v74.md)。

v75 随后把 v74 的类型互补性冻结为“仅问题文本”的路由假设：历史 800 例只用于 27 配置五折开发，运行时不读取官方类型、答案或 gold；另取互不重叠的 800 例开发集和 800 例确认集。学习路由在两阶段的证据 F1 为 `0.537489/0.531013`，相对各阶段最强静态对照提高 `+0.019575/+0.013571`，95% CI 分别为 `[+0.013056,+0.026464]` 与 `[+0.008214,+0.018929]`，全部预注册门槛通过。但它相对简单词法路由在开发集仅 `+0.001071` 且区间跨 0，在确认集点估计为 0；因此证据支持“问题表述路由有效”，不支持“学习器显著优于词法规则”。这仍是同一公开数据集内的案例隔离证据，不是独立数据集、官方榜单、答案生成或洪水专家验证；选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。详见 [v75 综合报告](docs/progressive_upgrade/twowiki_question_router_synthesis_v75.md)。

v76 进一步把运行时图结构、冻结检索分数、选择器重叠和 v75 问题概率组合成 71 个无 gold 特征，只在预测软标题闭包相对 cross-encoder 有正收益时覆盖。模型开发完整披露 1,000 条永久排除 HotpotQA 历史上的 1,089 个有限配置；另取与历史及彼此均零重叠的 800 例开发集和 800 例确认集，bridge/comparison 各 400 例。候选在两阶段的证据 F1 为 `0.568959/0.570916`，相对全部登记对照中的最强 cross-encoder 提高 `+0.008119/+0.006460`，95% CI 为 `[+0.003859,+0.012623]` 与 `[+0.002201,+0.010864]`，13 项门槛连续两次全部通过；完整证据召回也达到 `0.775000/0.787500`。这提供了独立于 v75 2Wiki 的公共数据集机制证据，但 v76 的训练历史与目标仍同属 HotpotQA，故不是独立训练数据确认；comparison 点估计仍小幅退化，且不是官方榜单、答案生成、真实 SetR 或洪水专家验证。选择器、CANARY、DEFAULT 均未授权，Gate 2 保持 `NO-GO/SHADOW`。详见 [v76 综合报告](docs/progressive_upgrade/hotpot_graph_router_synthesis_v76.md)。

v77 随后把完整冻结的 v76 路由器零调参迁移到 MuSiQue。v65—v73 的 16,900 个来源承诺全部永久排除，开发集从剩余 3,038 个未触碰 answerable 来源中按 2/3/4-hop=`400/250/150` 选择 800 例。候选 F1 `0.535779`，相对 cross-encoder 显著提高 `+0.007426`（95% CI `[+0.002535,+0.012416]`），但 MuSiQue 上最强控制变为交替锚点链接，F1 `0.542004`；候选相对它为 `-0.006225`（95% CI `[-0.012733,+0.000164]`），3-hop/4-hop 分层退化 `-0.009000/-0.016296`。严格优势和次级非劣包络均未建立，确认集按预注册规则未打开。该负结果限制了 v76 的跨数据集主张，也表明二选一“cross-encoder/软闭包”目标没有覆盖 MuSiQue 更强的锚点策略；不得用这 800 例回调 v76 后仍声称独立确认。详见 [v77 综合报告](docs/progressive_upgrade/musique_graph_router_transfer_synthesis_v77.md)。

v78 把 v77 明确降格为目标域均值校准集：HotpotQA 历史上的双增益头先在 432 个有限策略中选型，但 OOF F1 `0.563008` 仅比 v76 高 `+0.000560`，且锚点路由为 0；随后只用 v77 三种方法的总体均值形成一个无 hop、无个案条件的截距校准，并在另取的 600 个零重叠 MuSiQue 来源上前瞻验证。候选 F1 `0.546382`，相对 v76 `+0.000496`（95% CI `[-0.006369,+0.007388]`），相对最强交替锚点 `-0.001700`（95% CI `[-0.006865,+0.003572]`）。全部非劣条件通过，但严格优势、F1 下限和 4-hop 安全条件失败，确认仍未打开。这说明总体均值校准消除了明显退化，却没有证明逐例三路路由优于直接锚点；详见 [v78 综合报告](docs/progressive_upgrade/musique_mean_calibrated_three_route_synthesis_v78.md)。

v79 随后只用冻结 v78 的 600 条证据做目标域逐例路由开发：71 个运行时无 gold 特征、两个相对 cross-encoder 增益头、108 个模型配置与 4 个阈值形成 432 个有限策略。选中策略的 v78 OOF F1 为 `0.551733`，但该值参与选型，不能视作独立确认。模型、阈值、470 例规模、2/3/4-hop=`250/150/70` 配额、八个控制和门槛锁定后，另取与 v65—v78 零重叠的 470 条 MuSiQue 样本完成盲评。候选 F1 `0.553723`、完整证据召回 `0.597872`，相对 v78 `+0.005961`（95% CI `[-0.000380,+0.012606]`），相对最强交替锚点 `+0.002871`（95% CI `[-0.002187,+0.007844]`）；三条路线均实质使用，逐跳安全和非劣包络通过，但严格优势及两项区间门失败，因此确认未打开。它支持目标域逐例模型优于总体均值校准的点估计趋势，但尚未建立相对强锚点的显著优势；详见 [v79 综合报告](docs/progressive_upgrade/musique_target_three_route_synthesis_v79.md)。

v80 在不复用终局样本调参的前提下，把锚点改为默认路线，并用永久排除的 v78+v79 共 1,070 条证据学习 cross-encoder/软闭包相对锚点的两个增益头。216 个模型配置 × 7 个阈值形成 1,512 个有限策略；选中策略的 OOF F1 `0.554843` 仍只是模型选择诊断。由于排除 18,770 个历史 ID 后 4-hop 只剩 73 条，协议在目标 ID 前锁定一次性 600 例终局留出，2/3/4-hop=`400/150/50`，不再预留伪确认集。终局候选 F1 `0.527817`、完整证据召回 `0.643333`，相对最强 v79 仅 `+0.001561`（95% CI `[-0.002533,+0.005774]`）；三路均实质使用，非劣包络通过，但绝对 F1、主增益/区间和 3-hop `-0.006667` 严格门失败。状态为 `ADVANTAGE_NOT_ESTABLISHED`，这 600 例不得回用于修模，Gate 2 继续 `NO-GO/SHADOW`；详见 [v80 综合报告](docs/progressive_upgrade/musique_anchor_default_terminal_holdout_synthesis_v80.md)。

v81 回到仍有充足未触碰容量的 2WikiMultiHopQA，把冻结 v75 作为默认动作，仅学习 `cross_override` 与软/锚翻转两个高置信残差覆盖。模型只使用永久排除的 v75 开发与确认 1,600 例；73 个运行时无 gold 特征、216 个模型配置与 7 个阈值形成 1,512 个有限策略，选中 OOF F1 `0.540164`、相对 v75 `+0.005913`（95% CI `[+0.004067,+0.007857]`），但仍只是模型选择证据。协议、源登记、模型、代码、门槛和测试在目标选择前完成八文件锁；新开发集四题型各 200 例，与 3,400 个既往 2Wiki ID 零重叠。候选 F1 `0.547195`、完整证据召回 `0.658750`，相对最强 v75 为 `+0.004444`（95% CI `[+0.001706,+0.007381]`）：区间为正但未达到预注册的 `+0.005`，comparison 相对该题型最佳对照为 `-0.008571`。严格门失败、非劣包络通过，确认阶段未打开，Gate 2 继续 `NO-GO/SHADOW`；详见 [v81 综合报告](docs/progressive_upgrade/twowiki_residual_three_route_synthesis_v81.md)。

v82 把 v81 开发集明确纳入永久排除的训练历史，用问题文本先识别直接 comparison 并路由到 cross，其余样本只允许 v81 的软/锚安全翻转。2400条历史样本的严格样本外诊断相对 v75 为 `+0.006733`（95% CI `[+0.005079,+0.008505]`）；9文件锁后另取四题型各200例，与4200个既往 ID 零重叠。盲开发候选 F1 `0.544792`、完整证据召回 `0.643750`，相对 v75 为 `+0.006508`，并把 bridge comparison/comparison 相对各自最佳对照修复到 `0`；但本批最强对照变为 v81，候选相对 v81 仅 `+0.002143`（95% CI `[+0.000357,+0.004286]`），未达到 `+0.005` 实际优势线。16项严格门仅此1项失败，确认阶段未打开，Gate 2 继续 `NO-GO/SHADOW`；详见 [v82 综合报告](docs/progressive_upgrade/twowiki_cascaded_style_residual_synthesis_v82.md)。

## 当前验证状态

以下为截至 `2026-08-04` 的仓库验证基线；Python、Ruff 与审计项已于当日重跑，其余项目继承此前通过记录：

| 验证项 | 结果 |
|---|---|
| Python 全量测试 | 789 / 789 通过；Ruff 0 问题；编译通过 |
| 前端单元测试 | 19 / 19 通过 |
| Chromium E2E | 2 / 2 通过 |
| 生产构建 | 主前端与 Cesium 独立构建均通过 |
| 依赖安全审计 | Python 隔离依赖、两套 Node 工程均无已知漏洞 |
| 受控性能预算 | 4 / 4 端点通过，160 次请求错误率为 0 |
| 正常与故障场景 | 24 / 24，`PASS` |
| 设计合同审计 | 89 / 89 已归属：87 条本地/受控证据，2 条外部 No-Go |
| 完成性审计 | 本地要求 63 / 63，`CONTROLLED_SCOPE_COMPLETE_PRODUCTION_NO_GO` |
| GitHub Actions | 后端、前端、安全、Compose/Chromium/PostGIS 4 项门禁通过 |

CI 会重新生成关键评测、OpenAPI 和完成性审计并逐字节比较，防止文档结论与代码行为漂移。

## Gate 与研究结论边界

| 门禁 | 当前状态 | 结论 |
|---|---|---|
| Gate 0：模拟数据治理 | `GO`（受控模拟） | 数据来源、版本、种子和清单可复现 |
| Gate 1：候选对象质量 | `GO`（受控模拟） | 达到冻结阈值，仍强制人工确认 |
| Gate 2：FRC-RAG | `NO-GO / SHADOW` | v84 在 HotpotQA 安全约束内两阶段通过；v86 否定零调参 HotpotQA→IIRC 基数迁移；v89 在 IIRC 800/1200 例案例互斥开发/确认集上相对 `frc_fixed2` 提升 `+0.030402/+0.028902` 且区间下界均为正，但仍缺独立训练数据迁移、真实 SetR、公平答案质量与防汛专家证据 |
| Gate 3：安全不变量 | `GO`（自动化） | 越权、未审批、重复写、冲突和非法状态迁移失败关闭 |
| Gate 4：生产验收 | `CONDITIONAL NO-GO` | 缺真实 UAT、生产安全、容量和异地恢复证据 |
| Migration M7：旧链路退役 | `NO-GO` | 需真实旧流量归零、在途事件清空、归档恢复和账号撤权 |

FRC-RAG 当前可证明的是：多字段、功能角色、适用性和冲突约束能够形成可运行、可审计、可复现的证据组织流程；v75/v76 分别在 2WikiMultiHopQA 与 HotpotQA 建立案例隔离机制支持，v77—v83 划清跨域迁移、目标域路由和确认门边界，v84 在 HotpotQA 安全约束内连续通过开发与确认。最新 v85—v89 又完整记录了 BeerQA 来源不合格、HotpotQA→IIRC 零调参迁移失败、IIRC 目标域分数路由开发，以及冻结模型在新 IIRC 开发/确认集上的复制成功。现有证据仍不能写成独立训练数据迁移、答案生成质量、真实 SetR 或真实防汛专家有效性证明，因此正式任务继续保留 Baseline 降级策略，只有补齐 Gate 2 全部条件后才考虑 `CANARY` 或 `DEFAULT`。

最新充分性校准实验在冻结 ConditionalQA 真实模型分数上按 case 隔离训练、校准和评估：主分析将不完整证据 case 家族的误放行率从 0.810811 降至 0.040541，但拒答率为 0.946667、完整变体放行召回率仅 0.295455。它说明保守 abstention 机制具备理论可行性，也清楚显示当前效用仍不足；该跨域合成缺失实验不改变 Gate 2，也不替代真实防汛双专家评判。

10 组预注册重复 case 分组进一步确认：alpha=0.10 的 case 家族误放行率均值从启发式的 0.847827 降至 0.080422，10/10 组均改善，但平均拒答率仍为 0.930263。系统不把 alpha=0.20 当作自动放行阈值，只将两阈值之间标记为人工 `REVIEW_PRIORITY`；该复核带平均额外找回 0.319242 的完整变体。所有候选仍保持 `SHADOW` 或人工复核。

为降低拒答率进行的 12 配置、三段内层隔离嵌套选型未达到采用标准：完整召回提高 0.065759，但拒答只下降 0.022537，case 风险增加 0.040930。outer evaluation 未参与选型，负结果被保留，固定线性 L2=4 不变。

未参与选型的 HotpotQA 跨数据集确认中，校准后 case 风险均值为 0.096653，安全改善在 10/10 分组复现，但逐组不高于 alpha 的结果只有 6/10，未达到预注册 7/10，状态为 `PARTIAL_CONFIRMATION`。平均安全信号得到复现，逐分组一致性尚未确认。

随后按同一冻结协议在 2,556 例 MultiHop-RAG 真实 BGE/reranker 分数上进行第二次独立确认：case 风险均值为 0.087890，10/10 组降低误放行，8/10 组不高于 alpha，单数据集达到 `FULL_CONFIRMATION`。第三次确认在查看结果前冻结 2WikiMultiHopQA dev 的 1000 例 SHA-256 抽样、模型 revision、角色查询、L2=4、alpha=0.10 和相同 10 个分组；真实评分源 SHA-256 为 `7add5a5ec5206a33848de0eecc76a8697dc321e03946fcfefee20df0525a6104`。2Wiki case 风险均值为 0.090575，10/10 组降低误放行、7/10 组不高于 alpha，单集也为 `FULL_CONFIRMATION`，但拒答率 0.927512、完整召回仅 0.162305。预注册系列规则要求所有确认集均完整通过，因此三确认集系列仍为 `PARTIAL_CONFIRMATION`（完整 2/3、部分 1/3）；该结果增强了保守安全协议跨公开多跳数据的可行性证据，但不改变 Gate 2、不能证明 FRC 检索优势，也不能替代真实防汛双专家评测。

重复分组执行已改为单次解析 JSONL、复用与 split 无关的特征，再按冻结版本在内存中重分组。MultiHop-RAG 全量等价性基准中，旧路径 65.687275 秒、新路径 7.297463 秒，提升 9.001385 倍；两条路径的 10 组结果规范 SHA-256 完全一致。该数字是本机实验编排耗时，不代表生产 SLA。

进一步的可观察子群压力审计表明：四数据集 40 个全局有限样本名义上界均不高于 alpha=0.1，但这只是交换性假设下的边际保证，不是条件子群保证。22 个满足预注册支持度的子群中，HotpotQA `comparison` 问题风险均值为 0.179924、仅 1/10 组不高于 alpha；MultiHop-RAG `required_role_count=2` 为 0.134452、仅 2/10；2Wiki 候选数 `<20` 子群为 0.189738、仅 3/10，`bridge_comparison` 为 0.138887、仅 3/10。总体状态仍为 `SUBGROUP_INSTABILITY_DETECTED`，因此单集完整确认不能外推到所有问题类型、角色复杂度或候选规模。

在看到上述不稳定性后，仓库另行冻结并运行了 `question_type × required_role_count` 的分层 Mondrian conformal 事后开发方案；校准支持不足时依次回退到问题类型、角色数和全局阈值。它把 ConditionalQA、HotpotQA、MultiHop-RAG 的总体 case 风险均值分别从 0.080422/0.096653/0.087890 降至 0.075427/0.093480/0.074545，最坏可评子群风险平均改善 0.047331；但 MultiHop-RAG 完整召回下降 0.066904，超过冻结上限 0.05，且 ConditionalQA、HotpotQA 仍未满足子群重复一致性要求。因此预注册状态为 `DO_NOT_ADOPT`：不替换当前全局方法，也不构成独立确认或 Gate 2 放行证据。

第三确认又揭示候选规模与问题类型的残余失稳后，仓库在运行结果前另行冻结了包含 `question_type`、`candidate_count_bucket`、`required_role_count` 的八级多轴 Mondrian 回退规则，并在四个已揭示数据集上作为事后方法开发运行。它将 2WikiMultiHopQA、ConditionalQA、HotpotQA、MultiHop-RAG 的总体 case 风险均值从 0.090575/0.080422/0.096653/0.087890 调整为 0.070206/0.072862/0.099623/0.074545，最坏可评子群风险平均改善 0.055714；但 MultiHop-RAG 完整召回仍下降 0.066904，HotpotQA `40_to_59` 候选子群风险均值仍为 0.119606，且 2Wiki 与 HotpotQA 多个子群未达到至少 7/10 次一致性。因此八项采用检查中三项失败，状态为 `DO_NOT_ADOPT`。协议据此停止本轮方法扩张，不下载或查看 MuSiQue，不替换全局方法。

为区分“跨数据集重复校准”与“模型参数迁移”，仓库又在结果前冻结四数据集 leave-one-dataset-out 审计：每个目标只使用另外三集 train case，三集等权、集内正负类等权，目标 train 标签和特征完全禁用；目标 calibration 仅设置安全阈值。反事实单元测试会改写目标集全部标签与特征，并要求迁移头参数逐元素不变。实际迁移头相对目标拟合头的跨数据集平均 AUC 下降 0.115104；ConditionalQA/MultiHop-RAG 完整召回分别下降 0.202343/0.351185，2Wiki/HotpotQA 平均 case 风险为 0.104410/0.102824，只有 MultiHop-RAG 达到至少 7/10 次 alpha 控制。因此五项性能采用检查全部失败，状态为 `DO_NOT_ADOPT`，下一步固定为 `STOP_TRANSFER_METHOD_SELECTION_WITHOUT_NEW_DATA`。这说明现有 37 维充分性头需要目标域拟合，不能表述为可跨域迁移或零样本方法。

随后冻结了一个不改模型、不改阈值的 calibration-only 跨域头准入守卫，尝试在 evaluation 前失败关闭不可靠迁移。证书要求 calibration 至少 30 个不完整 case、30 个完整变体、AUC≥0.75 且冻结阈值下完整召回≥0.15。40 个重复中仅 3 个获得证书，低于最低覆盖 8 个；更重要的是这 3 个在 evaluation 上全部超过 alpha，case 风险为 0.112500、0.153846、0.132616，安全且有效证书精度为 0。因此守卫同样为 `DO_NOT_ADOPT`，不能用同一 calibration 的 AUC/召回替代独立验证；系统失败关闭策略明确为 `REQUIRE_TARGET_FITTED_HEAD`。

随后又冻结了只从 outer train 学习 `question_type`、`required_role_count` 及联合类别 one-hot 的上下文充分性头，仍使用 calibration-only 全局阈值。它在三集上的平均完整召回增益为 0.011962，未达到 0.02；ConditionalQA case 风险由 0.080422 增至 0.091715（+0.011292），超过风险增量上限 0.01，最坏子群风险平均变化为 -0.006680，即总体恶化。HotpotQA comparison 与 MultiHop-RAG temporal/comparison 子群仍不稳定，因此同样判定 `DO_NOT_ADOPT`。这说明简单上下文 one-hot 不能安全解决当前子群异质性，不修改冻结全局方法。

为引入真正不同于上下文类别的新可观察信号，仓库又在运行前冻结 BM25、dense、hybrid、cross-encoder 四阶段的分数间隔、选择排名稳定性、六组 Top-K Jaccard 和首位共识，共追加 27 维；每个目标数据集仍只用 train 拟合 64 维头、calibration 设置全局阈值。四集平均完整召回提高 0.090658，AUC 均提高且平均增益 0.014709；但 2Wiki/ConditionalQA/HotpotQA case 风险升至 0.104750/0.110814/0.106772，四集分别只有 6/3/6/6 次风险不高于 alpha，ConditionalQA 最坏子群风险增加 0.029127。因此安全采用检查失败，状态为 `DO_NOT_ADOPT`，下一步固定为 `STOP_SCORE_STABILITY_EXPANSION_ON_REVEALED_EVALUATIONS`。这些信号可供后续冻结的人工复核排序研究，但不能改变自动完整声明或 Gate 2。

在单独冻结的复核排序协议下，自动候选集合、37 维分数和 alpha=0.10 阈值保持逐条不变，只对未自动声明的队列比较原 37 维分数与 64 维分数的审核次序。64 维排序的跨集平均 AP 提高 0.053574，在 5%/10%/20% 固定审核预算下完整变体捕获率平均提高 0.036405/0.049725/0.026372；但 HotpotQA 最差合格子组在 10% 预算下降 0.055752，超过预注册 0.05 上限。因此最终状态为 `KEEP_BASE_REVIEW_ORDER`，不采用新排序，并停止在已揭示 evaluation 上继续开发；Gate 2 仍为 `NO-GO/SHADOW`。

为增加一套未参与上述方法选择的公开补充证据，仓库在下载 QASC validation 前冻结了数据提交、文件 SHA-256、926 例全量样本、每例 2 条官方事实加 38 条确定性 BM25 干扰事实的受控候选池，以及 BGE-large-en-v1.5/bge-reranker-large revision。评分包装器在神经模型调用前移除答案、金证据 ID、`gold` 与 `gold_roles`，真实评分源 SHA-256 为 `6419428623c1c4ed1a5935acf1402331c166e9afa9a80f45441e9e7540c759c0`。冻结 37 维目标拟合充分性协议将 case 风险均值从 0.873854 降至 0.101786，10/10 次降低误放行，但只有 4/10 次不高于 alpha，平均拒答率 0.885408、完整召回 0.282873，因此按更严格的冻结评估器保留 `NOT_CONFIRMED`。登记文字原本会将该组合归为部分确认，而既有评估器要求均值风险也通过；该定义差异已作为 `PARTIAL_STATUS_DEFINITION_MISMATCH` 审计记录，不事后改代码、阈值或结论。

QASC 总体结果揭示后，仅按事前限定的 `question_type` 做描述性诊断。`qasc_what` 与 `qasc_other` 的平均风险为 0.097125/0.095696，但分别只有 6/10、5/10 次不高于 alpha，其余类型未达到每次 30 个不完整 case 的支持度，故状态为 `DESCRIPTIVE_HETEROGENEITY_DETECTED`。该诊断不是条件 conformal 保证，不运行失败的 64 维复核排序，不纳入原三确认系列，也不改变 `NOT_CONFIRMED`、Gate 2 或 MuSiQue 的未触碰状态。

在不使用 QASC 标签调参、也不改写冻结评分源的前提下，仓库进一步预注册了 32 例、1,280 候选的真实 GPU 评分编排基准，用于评估跨 case 全局批处理能否等价替代逐 case 评分。批大小 8 的全部分数、九组排序与 FRC 选择在 `1e-6` 容差内等价，但仅加速 1.030910 倍，低于 1.10 倍采用门槛；批大小 16/32 虽然排序和选择仍完全一致，最大分数漂移约为 `2.40e-5`/`2.33e-5`，超过容差。结论为 `KEEP_FROZEN_PER_CASE_SCORING_ENTRYPOINT`：新批处理实现保留为可复现实验代码，不进入后续正式评分入口，不改变 QASC `NOT_CONFIRMED`、Gate 2 或 MuSiQue 边界。协议与结果见 [qasc_scoring_optimization_protocol.json](docs/progressive_upgrade/qasc_scoring_optimization_protocol.json) 和 [qasc_scoring_optimization.md](output/rag_evaluation/qasc_scoring_optimization/qasc_scoring_optimization.md)。

CONFLICTS 的检索选择和冲突类型结果揭示后，仓库另行冻结了 expected-behavior 回答评审协议：同一本地 Qwen 模型、提示和预算分别读取覆盖贪心代理与 FRC 的已冻结证据；生成器看不到冲突标签、官方答案、方法身份、检索分数或角色分数。458 例共形成 916 条逐例随机 A/B 盲化回答（19 例两方法选择完全相同，只生成一次并复用；共 897 个唯一提示），方法映射仅保存在 Git 忽略缓存，发布清单只提交其 SHA-256。当前状态为 `GENERATED_AWAITING_INDEPENDENT_HUMAN_ANNOTATION`：两名独立评审员和一名独立裁决员尚未完成表单，因此不存在 expected-behavior adherence、事实依据、引用正确性或答案正确率结果，Gate 2 继续保持 `NO-GO/SHADOW`。协议与评审说明见 [conflicts_expected_behavior_protocol.json](docs/progressive_upgrade/conflicts_expected_behavior_protocol.json) 和 [PROTOCOL.md](output/rag_evaluation/conflicts_expected_behavior/PROTOCOL.md)。

为让三名独立人员可以在不解盲的前提下实际执行评审，仓库又在任何人工决定产生前登记了 [conflicts_annotation_operations_protocol.json](docs/progressive_upgrade/conflicts_annotation_operations_protocol.json)，把每个角色的 458 个主任务和 23 个跨批隐藏复测拆成 8 个可续作批次（每角色 481 项，共 24 个批次）。公开批次不含原条目 ID、重复标记、方法身份或私有路由；路由只保存在 Git 忽略缓存，清单提交其 SHA-256。合并器恢复冻结的 458 例顺序，并在任一评分维度或偏好精确一致率低于 0.80 时失败关闭为 `REVIEW_REQUIRED`。当前仍只是 `PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW`，空模板不是人工结果。

为降低真实人员处理 JSON 表单的操作风险，v30 增加了独立于洪水响应产品界面的本机盲评工作台。它只监听 `127.0.0.1`，启动链接和 API 双重校验会话令牌，逐批验证公开文件哈希，将原子草稿、导出件和完成回执限定在 Git 忽略的 `.cache`，且从不加载私有路由或 A/B 方法映射。工作台支持自动保存、重启/刷新恢复、完整性校验、完成后锁定与显式重开；[工作台合同](docs/progressive_upgrade/conflicts_annotation_workstation_contract.json)固定状态为 `LOCAL_REVIEW_UI_READY_AWAITING_HUMANS`，不能视为人工评测已经开始或完成。

三名人员必须分别使用不同计算机账户或相互隔离的本机环境，并各自连续完成 1—8 批；例如第一批：

```powershell
python scripts/run_conflicts_annotation_workstation.py --reviewer-slot reviewer_1 --batch-index 1 --open-browser
python scripts/run_conflicts_annotation_workstation.py --reviewer-slot reviewer_2 --batch-index 1 --open-browser
python scripts/run_conflicts_annotation_workstation.py --reviewer-slot adjudicator --batch-index 1 --open-browser
```

后续只替换 `--batch-index`。同一角色八批必须使用同一个私有评审员标识，不同角色不得由同一人承担；不要查看其他角色草稿、私有路由、实现代码或解盲映射。

v31 增加三角色私有批次收集与合并前检查，避免人工枚举 24 个文件或绕过完成回执。以下命令只读取状态，不生成结果；当前真实状态是 `0/24`、`AWAITING_INDEPENDENT_HUMAN_BATCHES`：

```powershell
python scripts/run_conflicts_annotation_collection.py status
```

只有状态达到 `READY_FOR_PRIVATE_MERGE` 时才能运行私有合并：

```powershell
python scripts/run_conflicts_annotation_collection.py merge
```

编排器要求三角色分别完成 8 个有效回执、同一角色身份一致且三角色身份互异；随后验证私有路由承诺并执行每角色隐藏复测质控。任一维度低于 0.80 时输出 `REVIEW_REQUIRED`，不会生成跨评审员比较；全部通过也只进入 `READY_FOR_BLIND_MAPPING_VERIFICATION_AND_FINALIZATION`，仍未解盲、未完成最终裁决、不会改变 Gate 2。详细边界见 [私有收集合同](docs/progressive_upgrade/conflicts_annotation_collection_contract.json)。

v32 进一步审计“现有六种选择器是否存在可学习互补性”。这是结果已可见后的回顾性发现实验，不是预注册确认：47 个白名单特征只读取方法身份、冻结分类标签、检索分数统计、角色分数、成本/域/词汇多样性和选择集合重叠；gold 只作为训练目标与报告标签，按 case 哈希五折隔离。跨折路由 Accuracy 为 0.395197，相对最强静态覆盖贪心代理的 0.344978 提高 0.050218，95% 配对区间为 [+0.008734,+0.091703]。但严格移除 FRC 的路由仍达 0.393013，FRC 边际仅 +0.002183（区间 [-0.006550,+0.010917]），oracle 也只有 2 个 FRC 独有正确例；同时过时信息冲突 Recall 从 0.693548 降至 0.258065，且仅 3/5 折不劣于静态基线。最终状态为 `DISCOVERY_ROUTING_SIGNAL_NOT_ADOPTED_FRC_CONTRIBUTION_NOT_ESTABLISHED`：路由和 FRC 均不替换，Gate 2 仍为 `NO-GO/SHADOW`。协议与逐例可重算结果见 [路由协议](docs/progressive_upgrade/conflicts_selector_router_protocol.json) 和 [路由报告](output/rag_evaluation/conflicts_selector_router/conflicts_selector_router.md)。

```powershell
python scripts/run_conflicts_selector_router.py
```

v33 又在此前未参与方法开发的 WhoQA 上完成独立公开数据选择实验。下载前冻结 5,152 个问题、六种选择器、Top-K=4、1,500 Token、真实 BGE/reranker revision、逐问题 bootstrap 和无 gold 泄漏边界；26,957 个问题模板形成 161,742 次选择。最终 BM25/FRC 的归一化不同观点覆盖为 0.995203/0.994919，差值 -0.000283，同时区间 [-0.001100,+0.000039]；相对覆盖贪心代理的区间也跨 0。该候选池接近饱和且不代表时效失效、错误文件或防汛规则例外，故结论严格限定为 `WHOQA_SELECTION_SUPPORT_NOT_ESTABLISHED`，不据此否定其他 FRC 机制，也不改变 Gate 2。完整登记链与结果见 [WhoQA 协议](docs/progressive_upgrade/whoqa_conflict_coverage_protocol.json) 和 [WhoQA 报告](output/rag_evaluation/whoqa_conflict_coverage/whoqa_conflict_coverage.md)。

```powershell
python scripts/run_whoqa_conflict_coverage.py --evaluate-only
```

v34 在不修改选择器、模型分数或 v33 证据的前提下，进一步冻结 `K=2/3/4` 与 `256/512/1024/1500` Token 的六配置压力诊断。主指标不再按“预算实际能选出的条数”放宽分母，而是固定按 `min(观点数, K)` 计算，因此会同时惩罚重复观点和 Token 短缺。5,152 个问题、26,957 个模板共完成 970,452 次选择；六配置等权家族均值中 Dense/FRC 为 0.890719/0.887793，FRC 差值 -0.002926，同时 95% 区间 [-0.003858,-0.002025]。最大配置退化出现在 `K=4 / 256 tokens`，为 -0.016657；观点数超过 4 的 1,119 例差值为 -0.011928。预注册 -0.02 安全线未触发，但全部支持条件失败，状态为 `WHOQA_STRESS_SUPPORT_NOT_ESTABLISHED`。该结果明确暴露当前 FRC 在紧预算下没有成本效率优势；按停止规则不在 WhoQA 上回头调权重或算法，也不改变 Gate 2。协议与结果见 [预算压力协议](docs/progressive_upgrade/whoqa_budget_stress_protocol.json) 和 [预算压力报告](output/rag_evaluation/whoqa_budget_stress/whoqa_budget_stress.md)。

```powershell
python scripts/run_whoqa_budget_stress.py
```

v35 针对 v34 暴露的紧预算弱点，在接触 RGB 数据内容前冻结了成本感知子模选择器、9 种同预算方法、512/1024/1500 Token、真实 BGE 模型 revision 和 10,000 次分层配对 bootstrap。官方固定提交的 500 行中，2 行因正确证据与错误证据文本冲突而按协议排除；498 个盲评案例、14,580 个候选分块共执行 13,446 次选择。结果没有修复弱点：成本感知 FRC 的等权 Evidence F1 为 0.507421，低于旧 FRC 的 0.551199 和最强覆盖贪心代理的 0.556141；相对最强基线差值 -0.048719，同时 95% CI [-0.066307,-0.032655]，最差 `en_refine/1500` 退化 -0.127688，触发 `RGB_COST_AWARE_FRC_SAFETY_REGRESSION`。失败机制是按 Token 密度排序在 Top-K 槽位同时受限时偏向短片段：新方法通常占满 5 个槽位，却明显少用预算并损失精度/召回。反事实错误证据选择率相对旧 FRC 下降 0.022449，不足以抵消主指标退化。选择器未替换，禁止在已揭示 RGB 上调整公式，Gate 2 继续 `NO-GO/SHADOW`。协议、运行登记、结果后指标范围修正和报告见 [RGB 协议](docs/progressive_upgrade/rgb_cost_aware_frc_protocol.json)、[执行登记](docs/progressive_upgrade/rgb_cost_aware_frc_execution.json)、[修正登记](docs/progressive_upgrade/rgb_cost_aware_frc_post_result_correction.json) 与 [结果报告](output/rag_evaluation/rgb_cost_aware_frc/rgb_cost_aware_frc.md)。

```powershell
python scripts/run_rgb_cost_aware_frc.py --evaluate-only
```

v36 没有在已揭示的 RGB 上继续调参，而是在接触 MuSiQue 数据内容前冻结了无拟合权重的双资源代价 `token_count / token_budget + 1 / top_k`，用于同时表达 Token 与 Top-K 槽位稀缺性。官方 MuSiQue-Answerable dev 的 2,417 例、48,656 个候选分块完成 72,510 次同预算选择；准备与评分缓存无金标，逐例证据不导出问题、答案或候选原文。双资源 FRC 的等权 support evidence F1 为 0.528545，相对失败的 v35 提高 0.004402（95% CI [+0.002244,+0.006613]），但低于旧 FRC 的 0.530801，也低于最强非 FRC 基线 cross-encoder top-k 的 0.530399；相对最强基线为 -0.001854，同时 95% CI [-0.003784,-0.000020]。最差预算/跳数差值为 -0.004570/-0.004806，未触发安全回归线，但预注册优势条件失败，状态为 `MUSIQUE_DUAL_RESOURCE_FRC_SUPPORT_NOT_ESTABLISHED`。选择器未替换，禁止在 MuSiQue 上回调公式，Gate 2 保持 `NO-GO/SHADOW`。协议、执行登记和结果见 [MuSiQue 协议](docs/progressive_upgrade/musique_dual_resource_protocol.json)、[执行登记](docs/progressive_upgrade/musique_dual_resource_execution.json) 与 [结果报告](output/rag_evaluation/musique_dual_resource/musique_dual_resource.md)。

```powershell
python scripts/run_musique_dual_resource.py --evaluate-only
```

v37 按 v36 的冻结集合目标枚举每例所有满足 Top-K/Token 双约束的子集，用精确最优解区分“贪心优化误差”与“评分目标对齐不足”。84.94% 配置的 v36 目标 regret 不超过 `1e-9`，平均归一化 regret 仅 0.000809；精确解 Evidence F1 为 0.529945，相对 v36 仅 +0.001399（95% CI [-0.000208,+0.003008]），相对 cross-encoder top-k 为 -0.000455。状态为 `MIXED_OPTIMIZATION_AND_ALIGNMENT_DIAGNOSTIC`，说明少量尾部优化误差不足以解释整体差距，精确解不采用。v38 随后冻结五折 case 隔离、固定 L2=4 的支持概率校准：不含角色的基础模型和加入六项 FRC 角色特征的完整模型 F1 为 0.537234/0.537782。完整模型相对 cross-encoder 提高 +0.007383（区间 [+0.003425,+0.011259]），但未达到预注册 +0.01；FRC 角色相对基础模型仅 +0.000548（区间 [-0.000744,+0.001834]）。因此状态为 `CROSSFIT_SUPPORT_SIGNAL_NOT_ESTABLISHED`，停止在 MuSiQue 上继续调整此模型族，不能把通用校准小幅收益归因于 FRC。详见 [v37 诊断](output/rag_evaluation/musique_objective_gap/musique_objective_gap.md) 与 [v38 诊断](output/rag_evaluation/musique_crossfit_support/musique_crossfit_support.md)。

```powershell
python scripts/run_musique_objective_gap.py
python scripts/run_musique_crossfit_support.py
```

v39 按 v38 的停止规则离开已揭示 MuSiQue，在读取 HoVer 内容前预注册支持、反驳、实体桥接和跨文档链四类任务特定核验角色，并固定官方 dev、官方 top-20 TF-IDF 候选和官方 Wikipedia 数据库。全量 4,000 例、80,000 个候选文档、81,224 个分块完成 84,000 次同预算选择；候选池对支持文档的平均上限只有 0.608500，完整覆盖 1,060 例，因此该实验只检验固定候选内选择增量。任务特定角色 FRC、通用角色 FRC 和最强非 FRC `cross_encoder_topk` 的 document evidence F1 分别为 0.433286/0.433321/0.433019。任务特定角色相对通用角色为 -0.000035（95% CI [-0.000187,+0.000092]），相对最强非 FRC 仅 +0.000267，虽未触发任何预算或预注册分层的 -0.02 安全线，但均未达到 +0.01 支持门槛，状态为 `HOVER_VERIFICATION_ROLE_SUPPORT_NOT_ESTABLISHED`。4,000 行逐例证据无 claim、标题、文章或 supporting facts，evaluate-only 复跑三类产物字节一致；不采用新角色族、不回调 HoVer 参数，也不改变 Gate 2。详见 [v39 协议](docs/progressive_upgrade/hover_verification_roles_protocol.json)、[执行登记](docs/progressive_upgrade/hover_verification_roles_execution.json) 与 [结果报告](output/rag_evaluation/hover_verification_roles/hover_verification_roles.md)。

```powershell
python scripts/run_hover_verification_roles.py --evaluate-only
```

v39 结果后的无 gold 机制审计进一步定位到静态角色信号坍缩：通用/核验角色内部平均 Spearman 分别为 0.947723/0.946191，四个角色共享同一首选候选的比例为 0.8580/0.8425，通用与核验角色在三档预算下的有序选择完全相同率均超过 0.96。该诊断不作因果声称，只支持在新数据上前瞻测试 claim-conditioned 原子查询。最初的 v40 协议在任何查询生成或神经评分前，因一次结构普查意外打印了冻结分区的标签/跳数聚合而被关闭；2,512 个 v40 ID 永久排除，状态固定为 `REGISTRATION_BOUNDARY_VIOLATED_BEFORE_SCORING`，不得用于任何支持性结论。

v41 原样继承 v40 算法并从剩余 15,659 个未暴露 ID 重新抽取 512 例机制先导和 2,000 例确认集。先导集不连接 gold：动态角色内部平均 Spearman 从 0.945471 降至 0.521314，不同角色首选数从 1.154297 增至 2.230469，静态/动态有序选择相同率降至 0.432292，五项机制门槛全部通过。确认集随后按预注册顺序完成 40,000 个候选文档、40,616 个分块的 claim-only 查询生成和盲评分；29/2,000 条使用固定回退模板，回退率 1.45%，生成器和评分器均未看到 gold。动态/静态秩覆盖/最强非 FRC `cross_encoder_topk` 的 document evidence F1 为 0.428851/0.427258/0.427388。动态方法相对静态方法为 +0.001593（95% CI [+0.000453,+0.002762]），相对最强非 FRC 为 +0.001463（95% CI [+0.000308,+0.002688]）；方向显著为正且全部预算/分层均通过 -0.02 安全线，但分别未达到预注册 +0.005/+0.010 实质增益门槛。因此状态为 `DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED`：机制改善得到同 HoVer 家族确认，但方法支持未建立，不采用选择器、不继续在已揭示 HoVer 上调参，Gate 2 仍为 `NO-GO/SHADOW`。详见 [v40 关闭记录](docs/progressive_upgrade/hover_dynamic_atomic_roles_v40_closure.json)、[v41 协议](docs/progressive_upgrade/hover_dynamic_atomic_roles_protocol_v41.json)、[执行登记](docs/progressive_upgrade/hover_dynamic_atomic_roles_execution_v41.json)、[结果记录](docs/progressive_upgrade/hover_dynamic_atomic_roles_result_v41.json) 与 [确认报告](output/rag_evaluation/hover_dynamic_atomic_roles/hover_dynamic_atomic_roles.md)。

v42 将同一冻结方法迁移到此前未接触的 SciFact，并在下载前登记协议、实现和测试哈希。300 条 dev claim 中 187 条满足主评测资格；语料级 BM25 top-20 不注入 gold，Qwen 查询与 BGE 评分缓存均在 gold 联结前完成。动态角色信号仍显著去相关，但动态/静态/最强 `dense_topk` 的 document evidence F1 为 0.403909/0.401088/0.460616；动态相对静态 +0.002821、区间跨零，相对最强非 FRC -0.056707（95% CI [-0.078099,-0.036516]）。因此外部结果为 `SCIFACT_DYNAMIC_ATOMIC_ROLE_SUPPORT_NOT_ESTABLISHED`。该结果区分了“角色机制产生不同选择”和“选择真正提升效用”，并固定禁止在 SciFact 上调参、重复选型或改写 Gate 2。详见 [v42 协议](docs/progressive_upgrade/scifact_dynamic_atomic_roles_protocol_v42.json)、[实现登记](docs/progressive_upgrade/scifact_dynamic_atomic_roles_implementation_v42.json)、[执行登记](docs/progressive_upgrade/scifact_dynamic_atomic_roles_execution_v42.json)、[结果记录](docs/progressive_upgrade/scifact_dynamic_atomic_roles_result_v42.json) 与 [报告](output/rag_evaluation/scifact_dynamic_atomic_roles/scifact_dynamic_atomic_roles.md)。

v43 在访问 FEVEROUS 内容前冻结六类 challenge 各 40 例、官方基线预测页内候选、四角色 argmax 去重后的自适应证据数、三档预算和 10,000 次配对 bootstrap。240 条查询和评分缓存均在 gold 联结前完成，生成回退率为 2.9167%。候选池完整覆盖金证据组的比例仅 46.25%，低于预注册 60% 信息充分线；自适应方法把平均证据数从 4.998611 降至 2.770833、精度从 0.229444 提高到 0.301389，但完整证据召回下降 0.0875，Evidence F1 为 0.172505，显著低于动态 v41 的 0.218379（差值 -0.045874，95% CI [-0.079872,-0.016021]）和最强非 FRC `cross_encoder_topk` 的 0.209285（差值 -0.036780，95% CI [-0.073102,-0.005646]）。状态固定为 `FEVEROUS_BOUNDED_POOL_INCONCLUSIVE`：不采用自适应选择器、不在这 240 例上调参，且该页内候选实验不是官方 FEVEROUS 榜单或全库检索结果。详见 [v43 协议](docs/progressive_upgrade/feverous_adaptive_atomic_roles_protocol_v43.json)、[容量勘误](docs/progressive_upgrade/feverous_adaptive_atomic_roles_protocol_erratum_v43.json)、[执行登记](docs/progressive_upgrade/feverous_adaptive_atomic_roles_execution_v43.json)、[结果记录](docs/progressive_upgrade/feverous_adaptive_atomic_roles_result_v43.json) 与 [报告](output/rag_evaluation/feverous_adaptive_atomic_roles/feverous_adaptive_atomic_roles.md)。Gate 2 继续为 `NO-GO/SHADOW`。

v44 没有在已揭示的 FEVEROUS 上回调，而是在访问 OTT-QA 内容前冻结一步多重性守卫和全部统计门槛，并在 oracle-table 链接证据边界内平衡盲评 120 个表格来源与 120 个段落来源案例。守卫相对动态 v41 的 answer-evidence macro F1 提高 0.016093（95% CI [+0.001488,+0.029713]），平均少选 1.125 个证据单元，并修复了 v43 的部分过压缩；但它仍比最强公平非 FRC `official_anchor_topk` 低 0.015061，相对动态召回下降 0.066667，表格来源安全差值达到 -0.158409。因此状态为 `OTTQA_GUARDED_ADAPTIVE_SUPPORT_NOT_ESTABLISHED`，不采用、不在 v44 案例上继续调参，Gate 2 保持 `NO-GO/SHADOW`。该实验不是 OTT-QA 官方榜单或开放域表格检索结果。详见 [v44 协议](docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_protocol_v44.json)、[结果登记](docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_result_v44.json)与[完整报告](output/rag_evaluation/ottqa_guarded_adaptive_atomic_roles/ottqa_guarded_adaptive_atomic_roles.md)。

v45 离开已揭示的 OTT-QA，在下载 FinQA 前冻结“最高交叉编码器相关性锚点 + v44 守卫基数 + 动态角色覆盖填充”、实现/测试哈希、公平基线和统计门槛。官方 dev 的完整记录上下文中，表格、文本、混合金证据各抽取 120 例；360 条查询与神经评分在 gold 联结前完成，候选上限为 1.0，查询回退率为 3.3333%。v45/v44 supporting-fact macro F1 均为 0.510429，配对差值严格为 0；虽然 v45 相对动态 v41 和最强非 FRC `cross_encoder_knapsack` 分别提高 0.098177（95% CI [+0.084628,+0.111273]）和 0.100927（95% CI [+0.085851,+0.113962]），并平均少选 1.762037 个单元，但召回下降 0.059937，且更简单的 v43 F1 达到 0.623946。单/多金证据分层也均显示 v45 与 v44 完全相同。锁定结果后的无 gold 等价性诊断覆盖三档预算、1,080 个案例—预算配置：v45 与 v44 的目标证据数和最终证据集合均 100% 相同，仅 13.3333% 的配置选择顺序不同，v44 对可行最高相关性锚点的最终保留率为 100%。这解释了集合型 F1 的零差值，但不证明跨数据集的一般等价性。因此状态为 `FINQA_ANCHOR_GUARDED_SUPPORT_NOT_ESTABLISHED`：锚点约束没有提供相对 v44 的增量，不采用、不复用 v45 调参，Gate 2 保持 `NO-GO/SHADOW`。该实验只评估记录内 supporting-fact 选择，不是 FinQA 官方答案/程序榜单或开放语料检索。详见 [v45 协议](docs/progressive_upgrade/finqa_anchor_guarded_atomic_roles_protocol_v45.json)、[锁定结果](docs/progressive_upgrade/finqa_anchor_guarded_atomic_roles_result_v45.json)、[强制分层补充](docs/progressive_upgrade/finqa_anchor_guarded_atomic_roles_mandatory_strata_v45.json)、[等价性诊断协议](docs/progressive_upgrade/finqa_anchor_equivalence_diagnostic_protocol_v45.json)、[诊断结果](docs/progressive_upgrade/finqa_anchor_equivalence_diagnostic_result_v45.json)与[报告](output/rag_evaluation/finqa_anchor_guarded_atomic_roles/report.md)。

v46 没有在 FinQA 上继续调参，而是仅依据秩不变性和合成反例冻结“角色首选去重数 + 非首选 runner-up 双票时扩展 1 个”的共识约束基数公式。随后按固定提交获取 TAT-QA；官方 raw dev 的 1,668 条问题和 TAGOP dev 均不提供协议要求的精确表格单元/段落 evidence mapping。实验因此在结构普查后、查询生成前按停止规则关闭，没有用答案、推导或 `rel_paragraphs` 构造伪 gold，也没有生成神经分数或指标。状态为 `TATQA_FULL_CONTEXT_POOL_INCONCLUSIVE`：它既不支持也不反对 v46 方法。详见 [v46 协议](docs/progressive_upgrade/tatqa_consensus_guarded_atomic_roles_protocol_v46.json)、[关闭记录](docs/progressive_upgrade/tatqa_consensus_guarded_atomic_roles_result_v46.json)与[报告](output/rag_evaluation/tatqa_consensus_guarded_atomic_roles/report.md)。

v47 在未触碰的 FeTaQA 官方 dev 上原样重放 v46 公式。1,001 条合格样本按冻结 SHA-256 顺序选择前 600 条；完整表格候选对 supporting cells 的上限为 1.0，600 条查询和 GPU 分数均在 gold 联结前完成，查询回退 7 条（1.1667%）。候选方法、最强冻结 FRC 控制 `guarded_adaptive_cardinality_frc_v44` 和最强非 FRC `cross_encoder_knapsack` 的 supporting-cell macro F1 分别为 0.358926、0.375153 和 0.366367。候选相对最强 FRC 为 -0.016227（95% CI [-0.022628,-0.010334]），相对最强非 FRC 为 -0.007441（95% CI [-0.025087,+0.000670]）；虽然平均比动态 v41 少选 1.843333 个单元，召回却下降 0.085533，且候选池 Q1/Q3 分层分别退化 0.031307/0.027512。状态为 `FETAQA_CONSENSUS_GUARDED_SUPPORT_NOT_ESTABLISHED`：不采用、不复用 v47 案例调参，Gate 2 保持 `NO-GO/SHADOW`。该结果是附表内 supporting-cell 选择实验，不是 FeTaQA 官方答案生成榜单、表格—文本路由或开放语料检索结论。详见 [v47 协议](docs/progressive_upgrade/fetaqa_consensus_guarded_atomic_roles_protocol_v47.json)、[执行勘误](docs/progressive_upgrade/fetaqa_consensus_guarded_atomic_roles_execution_erratum_v47.json)、[锁定结果](docs/progressive_upgrade/fetaqa_consensus_guarded_atomic_roles_result_v47.json)与[报告](output/rag_evaluation/fetaqa_consensus_guarded_atomic_roles/report.md)。

v48 没有复用 v47 个案调参，而是只根据“保留四个动态角色各自 Top-2 提案并在提案集合内执行冻结秩覆盖”的合成不变量冻结新公式，随后在此前未使用的 QASPER 官方 dev 上前瞻验证。1,005 条问题中 802 条满足无歧义证据映射，按冻结 SHA-256 顺序选择 600 条；段落/图表标题完整候选池对替代证据参考的上限为 1.0，查询和 30,821 个候选单元的 GPU 分数均在 gold 联结前完成，查询回退 3 条（0.5%）。v48、最强冻结 FRC `adaptive_argmax_cardinality_frc_v43` 和最强非 FRC `cross_encoder_topk` 的 evidence macro F1 分别为 0.296239、0.316885 和 0.247513。v48 相对最强非 FRC 提高 0.048726（95% CI [+0.034631,+0.062649]），但相对最强 FRC 下降 0.020645（95% CI [-0.034871,-0.006266]）；平均比动态 v41 少选 1.263333 个单元，却带来 0.043625 的召回下降，includes-float 分层相对最强非 FRC 退化 0.023157。状态为 `QASPER_TOP2_PROPOSAL_GUARDED_SUPPORT_NOT_ESTABLISHED`：显著优于非 FRC 的局部信号不足以超过冻结 FRC 和召回/分层安全门槛，不采用、不复用 v48 案例调参，Gate 2 保持 `NO-GO/SHADOW`。该结果是附带全文的段落/图表标题证据选择实验，不是 QASPER 官方答案生成榜单、开放语料检索或洪水领域结论。详见 [v48 协议](docs/progressive_upgrade/qasper_top2_proposal_guarded_atomic_roles_protocol_v48.json)、[实现登记](docs/progressive_upgrade/qasper_top2_proposal_guarded_atomic_roles_implementation_v48.json)、[执行勘误](docs/progressive_upgrade/qasper_top2_proposal_guarded_atomic_roles_execution_erratum_v48.json)、[锁定结果](docs/progressive_upgrade/qasper_top2_proposal_guarded_atomic_roles_result_v48.json)与[报告](output/rag_evaluation/qasper_top2_proposal_guarded_atomic_roles/report.md)。

v49 没有读取或复用 v48 逐例结果，而是只依据“首名角色赢家高度集中、但次名提案存在广泛分歧时最多增加一个证据单元”的理论反例与合成不变量冻结 `low_core_divergence_guarded_frc_v49`，随后在 Evidence Inference 2.0 官方 validation 上前瞻验证。443 篇验证文章产生 1,201 条合格提示，按冻结 SHA-256 顺序选择 600 条；train/test ID 与文章正文始终未打开，查询和完整候选评分在 gold 联结前完成，查询回退 16 条（2.6667%）。v49、最强冻结 FRC `adaptive_argmax_cardinality_frc_v43` 和最强非 FRC `cross_encoder_topk` 的 evidence macro F1 分别为 0.220684、0.223942 和 0.186549。v49 相对最强非 FRC 提高 0.034135（95% CI [+0.021521,+0.046820]），三档预算及全部受支持分层安全检查均通过；但相对最强 FRC 为 -0.003258（95% CI [-0.008086,+0.001747]），召回增益 0.008710 也未达到预注册的 0.01。状态为 `EVIDENCE_INFERENCE_LOW_CORE_DIVERGENCE_SUPPORT_NOT_ESTABLISHED`：选择器不采用、不复用 v49 案例调参，Gate 2 保持 `NO-GO/SHADOW`。该实验是完整已给定临床论文内的证据句选择，不是 ERASER 官方结果、Evidence Inference 端到端推断榜单、开放语料检索或洪水领域结论。详见 [v49 协议](docs/progressive_upgrade/evidence_inference_low_core_divergence_atomic_roles_protocol_v49.json)、[实现登记](docs/progressive_upgrade/evidence_inference_low_core_divergence_atomic_roles_implementation_v49.json)、[执行超时勘误](docs/progressive_upgrade/evidence_inference_low_core_divergence_atomic_roles_execution_erratum_v49.json)、[锁定结果](docs/progressive_upgrade/evidence_inference_low_core_divergence_atomic_roles_result_v49.json)与[报告](output/rag_evaluation/evidence_inference_low_core_divergence_atomic_roles/report.md)。

v50 面向“缺失证据时应返回空集合”这一尚未解决的问题，在下载 ContractNLI 前冻结了 BGE reranker 原始 logit 的模型原生零边界：只有同一官方跨度的 anchor logit 大于 0，且 support 或 contradiction logit 也大于 0，才允许沿用 v43 选择器；否则拒答。官方 test 的 2,091 条固定假设记录按标签各取 200 条并限制每合同最多 6 条，形成 600 条均衡机制样本。首次结构准备发现 138 个从未被 gold 引用的纯空白官方跨度，实验在任何盲缓存或神经评分前失败关闭，并以透明 schema 勘误固定“只跳过空白跨度且保留原 `span_NNNN` 编号”；因此 v50 是 schema-repair confirmation，不表述为完全未触碰数据。随后完整候选池对 400 条证据样本的上限为 1.0，200 条 NotMentioned 的官方证据均为空，600 条确定性查询无回退，全部 GPU 分数在 gold 联结前完成。结果显示 anchor 与共识门在 600/600 条上全部通过，候选拒答率和 NotMentioned 拒答准确率均为 0；候选 utility F1 为 0.165173，相对最强共享门非 FRC 仅 +0.004430（95% CI [-0.015105,+0.023482]），相对最强冻结 FRC v49 为 -0.001423（95% CI [-0.008765,+0.005032]），Contradiction 分层差值为 -0.056798。状态为 `CONTRACTNLI_NATIVE_ZERO_CONSENSUS_SUPPORT_NOT_ESTABLISHED`：原始零边界对该 reranker 不具备缺失证据判别力，门控不采用，v50 案例不得用于阈值拟合或方法选型，Gate 2 保持 `NO-GO/SHADOW`。该实验是给定完整 NDA 内的跨度证据选择与拒答机制测试，不是 ContractNLI 官方 NLI 榜单、开放语料检索、真实 SetR 复现或洪水领域结论。详见 [v50 协议](docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_protocol_v50.json)、[schema 勘误](docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_protocol_erratum_v50.json)、[实现勘误](docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_implementation_erratum_v50.json)、[执行登记](docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_execution_v50.json)、[锁定结果](docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_result_v50.json)与[报告](output/rag_evaluation/contractnli_native_zero_consensus_abstention/report.md)。

v51 未在 v50 test 上追认阈值，而是在打开 ContractNLI dev 前冻结了“每个角色按候选池中位数/IQR 稳健标准化、同跨度 anchor 与 support/contradiction 取共识、五折文档隔离 OOF 拟合阈值”的两阶段协议。dev 按三类各 80 条、每合同最多 6 条形成 240 条/60 份合同；240 条确定性查询无回退，完整 GPU 评分 SHA-256 为 `e802cf…edbc6`，gold 只在评分完成后连接。执行器曾在盲样本准备后、查询/评分/指标前发现“覆盖率连接顺序”与协议冲突，已用透明实现勘误把 gold 覆盖检查移到完整评分后，公式、门槛、样本、模型和阈值规则均未改变。五折 OOF 的 utility F1 从 0.154504 提升到 0.191845（+0.037341），弃答率 0.141667；但 evidence F1 从 0.231756 降到 0.194018（损失 0.037738，超过 0.03 上限），NotMentioned 弃答准确率 0.1875 也未达到 0.20。状态因此锁定为 `CONTRACTNLI_V51_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TRAIN`：官方 train 未打开，确认实验未启动，阈值与选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。详见 [v51 协议](docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_protocol_v51.json)、[实现勘误](docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_implementation_erratum_v51.json)、[开发结果](docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_development_result_v51.json)、[关闭记录](docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_closure_v51.json)与[报告](output/rag_evaluation/contractnli_dev_calibrated_robust_consensus/development.md)。

v52 随后冻结了不含阈值、温度、边距或候选数特征的排名一致性门：anchor 第一名必须与 support 或 contradiction 第一名是同一跨度，门通过后才原样运行 v49 选择器。官方 train 仅作为一次性、角色反转的独立确认集；按三类各 200 条、每合同最多 6 条形成 600 条/319 份合同，查询无回退，完整 GPU 分数 SHA-256 为 `120055…353175`，gold 仍只在 600 条评分完整后连接。候选 utility F1 为 0.250304、弃答率 0.373333、NotMentioned 弃答准确率 0.410000；相对未门控 v49 和原生零控制均为 +0.081783（95% CI [+0.047778,+0.117474]），但相对同一门控下最强非 FRC hybrid 仅 +0.009038（95% CI [-0.006814,+0.024925]），且 evidence F1 相对未门控 v49 下降 0.082325，超过 0.03 上限。状态锁定为 `CONTRACTNLI_V52_RANK_CONCURRENCE_SUPPORT_NOT_ESTABLISHED`；train、v51 dev 与 v50 test 都不得事后调参或选型，选择器不采用，Gate 2 继续 `NO-GO/SHADOW`。详见 [v52 协议](docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_protocol_v52.json)、[结果](docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_result_v52.json)、[关闭记录](docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_closure_v52.json)与[报告](output/rag_evaluation/contractnli_rank_concurrence_confirmation/report.md)。

v53 随后换用独立 CUAD 合同审查来源，并在下载/打开数据前冻结 Top-3 角色交集门、角色闭包、BM25+dense Top-48 候选并集和共享门公平基线。官方 train 按答案/无答案各 200 条、每合同最多 4 条形成 400 条/259 份合同，查询无回退，完整盲评分 SHA-256 为 `1109b2…aeb4`，gold 只在 400 条评分完成后连接。检索池对答案例的完整覆盖仅 0.635；Top-3 门通过 397/400，候选弃权率 0.0075、无答案弃权准确率 0.005，几乎没有缺失证据判别。候选 utility F1 0.046032，相对共享门最强 `hybrid_topk` 为 -0.023731（95% CI [-0.040711,-0.006903]），最长合同四分位差值 -0.050796。状态锁定为 `CUAD_V53_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TEST`；CUAD test 未打开且不授权打开，train 不得复用调参/选型，选择器不采用，Gate 2 继续 `NO-GO/SHADOW`。详见 [v53 协议](docs/progressive_upgrade/cuad_top3_rank_concurrence_role_closure_protocol_v53.json)、[开发结果](docs/progressive_upgrade/cuad_top3_rank_concurrence_role_closure_development_result_v53.json)、[关闭记录](docs/progressive_upgrade/cuad_top3_rank_concurrence_role_closure_development_closure_v53.json)与[报告](output/rag_evaluation/cuad_top3_rank_concurrence_role_closure/development/report.md)。

v54 转向 [Doc2Dial 官方文档对话数据](https://doc2dial.github.io/data.html)，在固定官方仓库 revision 的 v1.0.1 train 上尝试验证“关联文档在 7 个同域对照文档中应排名第一”的无拟合门控。协议要求答案/无答案各 200 条，但结构资格统计得到 20,431 条答案样本、0 条空引用样本；因此在盲缓存、查询、神经评分和指标之前以 `DOC2DIAL_V54_SCHEMA_INCONCLUSIVE_STOP` 关闭，validation 内容未打开。该关闭只说明所固定版本不满足预注册的双状态配额，不是方法负结果。详见 [v54 协议](docs/progressive_upgrade/doc2dial_document_contrastive_role_closure_protocol_v54.json)、[开发关闭结果](docs/progressive_upgrade/doc2dial_document_contrastive_role_closure_development_result_v54.json)与[报告](output/rag_evaluation/doc2dial_document_contrastive_role_closure/development/report.md)。

v55 随后登记 Doc2Dial v0.9 的 wOOD 分支，因为[官方 README](https://doc2dial.github.io/README.html)明确说明无关轮次使用空 reference。冻结适配器在开发成员中发现 3,459 段对话的 `turns` 为列表、12 段为键控对象；若只接受列表并按旧规则解释空引用，schema 排除率为 0.018618，超过 0.01 上限。实验据此在任何盲缓存、查询、评分或指标前以 `DOC2DIAL_WOOD_V55_SCHEMA_INCONCLUSIVE_STOP` 关闭；wOOD dev 确认成员和 woOOD 兄弟成员均未打开。详见 [v55 协议](docs/progressive_upgrade/doc2dial_wood_document_contrastive_transfer_protocol_v55.json)、[schema 关闭结果](docs/progressive_upgrade/doc2dial_wood_document_contrastive_transfer_development_result_v55.json)与[报告](output/rag_evaluation/doc2dial_wood_document_contrastive_transfer/development/report.md)。

v56 在完整披露 v55 的 train schema 访问后，仅修正两项数据解释：`turns` 允许列表或整数键对象并按数值顺序读取；所有空 reference 都按官方语义记为无答案，邻接 OOD act 只作为描述性子类型。模型、查询、BM25/dense 候选池、8 文档对照门、角色闭包选择器、128/256/512 Token 预算和支持门槛均保持不变。开发盲样本为 400 条（答案/无答案各 200）、386 段对话和 261 份文档，schema、查询和评分回退率均为 0；完整 GPU 分数 SHA-256 为 `d1861b…e200`，gold 仅在缓存完整后连接。候选 utility F1 为 0.517417，无答案弃权准确率为 0.875，但答案 F1/Recall 仅 0.159833/0.123750，候选完整覆盖率 0.805 未达 0.99；相对共享门最强 Dense Top-K 为 -0.008827（95% CI [-0.021324,+0.003586]），相对同门 FRC 消融为 -0.000381，答案召回损失 0.049167，最差预算或支持分层差值 -0.038932。状态锁定为 `DOC2DIAL_WOOD_V56_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION`：确认集未打开且不授权打开，开发样本不得复用调参或选型，选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。该平衡机制实验不是 Doc2Dial 官方共享任务结果、真实 SetR 复现、开放语料检索或洪水领域结论。详见 [v56 协议](docs/progressive_upgrade/doc2dial_wood_schema_corrected_transfer_protocol_v56.json)、[开发结果](docs/progressive_upgrade/doc2dial_wood_schema_corrected_transfer_development_result_v56.json)、[关闭记录](docs/progressive_upgrade/doc2dial_wood_schema_corrected_transfer_development_closure_v56.json)与[报告](output/rag_evaluation/doc2dial_wood_schema_corrected_transfer/development/report.md)。

v57 转向此前未使用的 QuAC v0.2，并在任何 JSON 解析前冻结“共享门控 Cross-Encoder Top-5 前四项不变、至多替换第 5 项”的无参数单槽共识选择器。官方 train/validation 先按字节下载并登记长度与 SHA-256；首次 train 下载截断被审计记录并替换，随后只打开 train，validation 始终未解析。开发样本为 600 条（答案/无答案各 300）、578 段对话和 554 份文档，schema、查询、评分与 7 文档对照回退率均为 0，候选覆盖上限为 1.0；完整 GPU 分数 SHA-256 为 `031e669a…beedf`。候选 utility F1 为 0.114109、答案 F1/Recall 为 0.161552/0.424537，但无答案弃权准确率仅 0.066667、总体弃权率 0.075，单槽触发率 0.085 未达下限。相对精确锚点基线仅 +0.000317（95% CI [-0.001667,+0.002381]），相对最强共享门控 Knapsack 仅 +0.000159（95% CI [-0.002336,+0.002682]）；相对最强同门 FRC 为 +0.014943，但区间仍跨 0。状态锁定为 `QUAC_V57_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_VALIDATION`：validation 未打开且不授权打开，开发样本不得复用调参或选型，选择器不采用，Gate 2 保持 `NO-GO/SHADOW`。该实验是平衡的证据单元选择机制评测，不是 QuAC 官方答案抽取结果、真实 SetR 复现、开放语料检索或洪水领域结论。详见 [v57 协议](docs/progressive_upgrade/quac_anchor_safe_consensus_slot_protocol_v57.json)、[开发结果](docs/progressive_upgrade/quac_anchor_safe_consensus_slot_development_result_v57.json)、[关闭记录](docs/progressive_upgrade/quac_anchor_safe_consensus_slot_development_closure_v57.json)与[报告](output/rag_evaluation/quac_anchor_safe_consensus_slot/development/report.md)。

```powershell
python scripts/run_hover_role_mechanism_diagnostic.py
python scripts/run_hover_dynamic_atomic_roles.py evaluate
```

## 目录结构

```text
flood_system/                         FastAPI 运行时
  api.py                              应用工厂与兼容路径装配
  http/                               运维、Core、旧平台和 AgentTwin 路由
  response_workflow/                 响应域服务、端口、状态机与证据治理
  storage/                            分域仓储、聚合指标与有序迁移
  compat/                             M7 前保留的旧平台/AgentTwin 兼容实现
frontend/src/features/response/      React 响应工作台面板、文件和 API 分域
3D_visual/src/scene*.ts               Cesium 场景配置、类型和模型放置
research/frc_rag/                    不进入运行时发行包的 FRC-RAG 实验
tests/support/                        跨领域测试构造器与公共夹具
scripts/                              迁移、演示、评测、审计与 Worker 入口
infra/                                PostGIS 影子迁移与基础设施合同
benchmarks/                           冻结预算与受控基准配置
output/                               可复现评测与验收摘要
docs/                                 当前设计、运维、安全和验收文档
```

## 文档导航

- [产品定义与设计原则](PRODUCT.md)
- [渐进式迭代开发与升级设计](洪水预警响应系统_渐进式迭代开发与升级设计.md)
- [V3 产品与业务设计说明](面向区县防办的洪水预警响应系统_升级设计说明V3.md)
- [当前文档索引](docs/README.md)
- [渐进式升级交付索引](docs/progressive_upgrade/README.md)
- [接口、状态与数据合同](docs/progressive_upgrade/contracts_and_data_dictionary.md)
- [用户手册](docs/progressive_upgrade/user_manual.md)
- [评测与 Gate 报告](docs/progressive_upgrade/evaluation_and_gate_report.md)
- [全目标完成度追溯审计](docs/progressive_upgrade/completion_traceability_audit.md)
- [v0.3.2 FRC-RAG 实验与盲评基础设施更新报告](docs/releases/v0.3.2.md)
- [v0.3.1 三轮结构重构报告](docs/releases/v0.3.1.md)
- [v0.3.0 功能更新报告](docs/releases/v0.3.0.md)
- [OpenAPI 快照](docs/openapi.json)

## 项目定位

本项目的研究价值不在于让大模型自主发出防汛指令，而在于探索如何把多源、异构、带版本和有效期的业务信息，转换为对象级、证据约束、人工可控的响应任务。系统坚持“事实与建议分层、关键动作人工审批、证据和责任链全程可追溯”的设计原则。

## 最新研究进展：MuSiQue 迁移与 2Wiki 残差路由 v77—v81

v76 在目标样本选择前冻结 71 个无 gold 运行时特征、25 棵深度 2 的 Huber 梯度提升树、六个公平对照和 13 项门槛。1,000 条历史只用于模型开发并永久排除；800 条开发和 800 条确认样本均按 bridge/comparison 各 400 条平衡，历史/开发/确认三者 ID 完全互斥。候选证据 F1 在两阶段达到 `0.568959/0.570916`，相对最强登记对照提高 `+0.008119/+0.006460`，两个 95% CI 下界均高于 0；完整证据召回达到 `0.775000/0.787500`，平均 token 基本不增加。

该结果为独立于 v75 2Wiki 的公共数据集机制证据，但模型开发历史和目标阶段仍来自同一 HotpotQA 配置，只能声称案例隔离重复支持，不能声称独立训练数据确认。comparison 分层仍有约 0.1—0.2 个百分点小幅退化；实验也不是官方 HotpotQA 榜单、答案生成、真实 SetR、洪水领域验证或生产证据。详见 [v76 综合报告](docs/progressive_upgrade/hotpot_graph_router_synthesis_v76.md)。

v77 在任何新目标 ID 选择前锁定完整 v76 模型、MuSiQue 段落适配、六个公平控制、2/3/4-hop 配额、严格优势门槛和次级非劣包络。排除 v65—v73 的 16,900 个来源承诺后，800 例开发集与全部历史暴露重叠为 0。冻结路由器 F1 `0.535779`，相对 cross-encoder 提高 `+0.007426` 且区间下界为正；但最强交替锚点控制为 `0.542004`，候选相对它 `-0.006225`，95% CI `[-0.012733,+0.000164]`。3-hop/4-hop 分层分别退化 `-0.009000/-0.016296`，严格优势与非劣包络均失败，确认集未打开。

v77 证明“优于 cross-encoder”不能等价为“优于最强同资源方法”，也限制了 v76 的跨分布外推。v77 个案不得用于回调 v76 后仍声称独立确认；选择器/CANARY/DEFAULT 不授权，Gate 2 仍为 `NO-GO/SHADOW`。详见 [v77 综合报告](docs/progressive_upgrade/musique_graph_router_transfer_synthesis_v77.md)。

v78 只用 v77 已登记的总体路由均值校准 HotpotQA 双增益头，并将 v77 的 800 例永久排除。新的 600 例开发集按 2/3/4-hop=`300/200/100` 锁定，与 v65—v77 全部来源零重叠。候选 F1 `0.546382`、完整证据召回 `0.563333`，相对 v76 为 `+0.000496`、相对最强锚点为 `-0.001700`；全部非劣条件通过，但严格优势和 4-hop 安全门失败。确认未打开，且 v77 是显式目标域校准而不是独立确认，不能声称零目标调参。详见 [v78 综合报告](docs/progressive_upgrade/musique_mean_calibrated_three_route_synthesis_v78.md)。

v79 把 v78 的 600 条证据明确作为目标域训练/模型选择集，只在 432 个有限策略中选择 71 特征双增益路由器；其 OOF 结果参与选型，不能算独立确认。另取与 v65—v78 零重叠的 470 条样本后，候选 F1 `0.553723`、完整证据召回 `0.597872`，相对 v78 `+0.005961`，相对最强锚点 `+0.002871`；三路覆盖和 2/3/4-hop 安全均通过，但两个关键置信区间跨零，严格优势未建立，确认仍未打开。详见 [v79 综合报告](docs/progressive_upgrade/musique_target_three_route_synthesis_v79.md)。

v80 合并永久排除的 v78+v79 共 1,070 条目标域证据，把锚点设为默认路由并学习两个覆盖增益头；216 个模型配置与 7 个阈值形成 1,512 个策略。模型、阈值、九个控制、600 例终局规模和 2/3/4-hop=`400/150/50` 配额在读取目标 ID 前锁定，样本与 v65—v79 的 18,770 个历史 ID 零重叠。候选 F1 `0.527817`、完整证据召回 `0.643333`，相对最强 v79 为 `+0.001561`（95% CI `[-0.002533,+0.005774]`）；非劣包络成立，但严格优势、F1 下限和 3-hop 安全门失败。由于剩余 4-hop 容量不足以支持第二个可比阶段，v80 是终局留出且禁止同源再确认或回调。详见 [v80 综合报告](docs/progressive_upgrade/musique_anchor_default_terminal_holdout_synthesis_v80.md)。

v81 在读取目标 ID 前冻结“v75 默认 + cross/软锚翻转残差覆盖”的 73 特征模型和两阶段协议。模型开发只用随后永久排除的 v75 开发/确认 1,600 例；八文件实现锁后，新开发集从剩余 2Wiki 容量按四题型各取 200 例，与 3,400 个既往 ID 零重叠。候选 F1 `0.547195`、完整证据召回 `0.658750`，相对最强 v75 为 `+0.004444`（95% CI `[+0.001706,+0.007381]`）；非劣包络和 12/15 严格检查通过，但点增益未达到 `+0.005`，comparison 分层为 `-0.008571`。确认阶段按协议关闭，开发集不得作为 v81 确认复用。详见 [v81 综合报告](docs/progressive_upgrade/twowiki_residual_three_route_synthesis_v81.md)。

v82 使用永久排除的 v75 两阶段与 v81 开发集共 2,400 例冻结题型级联安全残差路由。新开发集四题型各 200 例，与 4,200 个既往 ID 零重叠；候选 F1 `0.544792`，相对 v75 提高 `+0.006508`，但相对最强 v81 仅提高 `+0.002143`，未达到预注册 `+0.005` 实际优势线，因此确认未打开。详见 [v82 综合报告](docs/progressive_upgrade/twowiki_cascaded_style_residual_synthesis_v82.md)。

v83 不再修改 v82 路由或排序，只用问题文本识别 bridge-comparison，并在该类案例上把冻结 Top-5 裁剪为前 4 条。模型开发使用 3,200 条永久排除历史；九文件锁后，开发与确认各取四题型各 200 例，二者及 5,000 个历史 ID 完全互斥。开发候选 F1 `0.556329`，18 项严格门全过并打开确认；确认候选 F1 `0.549420`，相对最强 v82 提高 `+0.006434`（95% CI `[+0.002465,+0.010253]`），但比绝对门槛 `0.55` 低 `0.000580`。该预注册失败不能后验降门槛，选择器/CANARY/DEFAULT 不授权，Gate 2 保持 `NO-GO/SHADOW`。详见 [v83 综合报告](docs/progressive_upgrade/twowiki_bridge_aware_precision_trim_synthesis_v83.md)。

v84 转向 HotpotQA 未触碰容量，在冻结 v76 路由和排序之后，用问题文本分类器为 comparison/bridge 分别保留前 3/4 条。2,600 条已暴露案例只用于有限模型选择并永久排除；九文件锁后，开发与确认各取 600 例（bridge 400、comparison 200），阶段及历史 ID 完全互斥。开发候选 F1/完整证据召回为 `0.644873/0.643333`，相对最强安全合格控制提高 `+0.029039`；确认分别为 `0.639898/0.625000`，提高 `+0.026492`（95% CI `[+0.019841,+0.033051]`），两阶段 18 项严格门全过。原始 F1 最高的固定前 3 条仍更高，但完整证据召回仅 `0.571667`，低于预注册 `0.62` 安全下限，因此结论只是在安全约束内成立，不是无条件 F1 优势。选择器/CANARY/DEFAULT 不授权，Gate 2 保持 `NO-GO/SHADOW`。详见 [v84 综合报告](docs/progressive_upgrade/hotpot_question_type_cardinality_synthesis_v84.md)。

v85—v89 把问题推进到 IIRC。BeerQA 因只含既有 SQuAD/HotpotQA 派生记录而在评分前关闭；v86 冻结 HotpotQA 基数模型零调参迁移到 400 条 IIRC 样本，F1 `0.616381`，相对 `frc_fixed2` 为 `-0.004512`，跨数据集优势未建立。v87 用已揭示 v86 开发集训练目标域分数差模型，新 400 条样本点增益 `+0.007831` 但区间跨 0。v88 汇聚两批历史训练后在新 800 条开发集取得 `+0.028773`（95% CI `[+0.015490,+0.041769]`），但确认准备被 3 候选源边界阻断。v89 不重训，只加入“少于 4 个候选时选择全部真实源”的无填充适配；新的 800/1200 条开发/确认样本与全部历史及彼此零重叠，候选 F1 为 `0.636527/0.627097`，相对 `frc_fixed2` 为 `+0.030402/+0.028902`，95% CI 分别为 `[+0.017976,+0.042557]` 与 `[+0.017896,+0.039656]`，两个阶段全部门禁通过。它建立的是冻结模型在 IIRC 上的同数据集源选择复制支持，不是独立训练数据迁移、答案生成、真实 SetR 或防汛专家结论；Gate 2 继续 `NO-GO/SHADOW`。详见 [v85—v89 综合报告](docs/progressive_upgrade/iirc_cardinality_transfer_iteration_synthesis_v89.md)。
