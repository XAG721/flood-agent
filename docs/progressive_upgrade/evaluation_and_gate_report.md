# 受控模拟评测与 Go/No-Go 报告

执行日期：2026-07-13。本文只记录仓库内真实运行结果；所有数值均来自受控模拟或仓库标注集，不代表真实洪水预测、真实跨部门接入或生产环境性能。

## 可复现命令

```powershell
python scripts/generate_floodagent_bench.py
python scripts/run_candidate_evaluation.py
python scripts/run_rag_evaluation.py
python -m pytest -q --basetemp D:\graduation_project\tmp\acceptance
cd frontend
npm.cmd run test -- --run
npm.cmd run build
npm.cmd run test:e2e
```

## 当前结果

| 验收项 | 实测结果 | 结论 |
|---|---:|---|
| 后端自动测试 | 155/155 通过 | 通过 |
| Python 静态质量 | Ruff 0.15.21：0 个问题 | 通过 |
| 响应工作流专项 | 54 个测试节点在全量套件通过 | 通过 |
| 前端测试 | 13/13 通过 | 通过 |
| 前端生产构建 | Vite 构建成功 | 通过 |
| Chromium 浏览器验收 | 2/2；新版桌面与 390px 移动视口、旧页面回退截图、0 横向溢出、0 控制台错误 | 通过 |
| Compose 服务健康 | Web、API、Worker、PostGIS 均 healthy | 通过 |
| PostGIS 版本 | PostgreSQL 17 镜像，PostGIS 3.5 | 通过 |
| SQLite→PostGIS 影子迁移 | 映射 v2，12 类、28 条；来源/目标数量一致 | 通过 |
| PostGIS 空间投影 | 1 个 EPSG:4326 Polygon，GiST 索引已建 | 通过 |
| 迁移隔离 | 0 条 | 通过 |
| 迁移幂等 | 同一源快照连续执行两次，批次 ID 与目标计数不变 | 通过 |
| 迁移对账 | 12/12 类规范载荷 SHA-256 一致 | 通过 |
| 受控 event list P95 | 12.816 ms / 预算 150 ms | 通过 |
| 受控 event dashboard P95 | 30.641 ms / 预算 250 ms | 通过 |
| 受控 event timeline P95 | 27.958 ms / 预算 200 ms | 通过 |
| 受控 audit trail P95 | 31.603 ms / 预算 300 ms | 通过 |
| 受控 API 错误率 | 4 条路径各 40 次采样，均为 0 | 通过 |
| Python 依赖漏洞 | pip-audit 2.10.1：0 个已知漏洞 | 通过 |
| 主前端依赖漏洞 | npm audit：0 个漏洞 | 通过 |
| Cesium 依赖漏洞 | npm audit：0 个漏洞 | 通过 |
| FloodAgent-Bench | 24 事件、30 对象、24 个正式场景、固定种子 `20260712`、全记录 `SYNTHETIC` | Gate 0 通过 |
| 24 场景自动化追溯 | 24/24 场景绑定全量 CI 测试节点 | 通过 |
| Candidate Recall@5 | 1.000000 | 达到 Gate 1 建议值 |
| 关键对象漏检率 | 0.000000 | 达到 Gate 1 建议值 |
| 校准后 ECE | 0.037792（校准前 0.300000） | 达到 Gate 1 建议值 |
| 30%缺失代理场景 Recall@5 降幅 | 0.00 个百分点 | 达到 Gate 1 建议值 |
| FRC 字段覆盖率 | 1.0000 | 达到 Gate 2 覆盖建议值 |
| FRC 引用正确率 | 1.0000 | 达到 Gate 2 建议值 |
| FRC 无依据证据比例 | 0.0000 | 达到关键字段无依据率 0 的要求 |
| 冲突检测 F1 | 1.0000（4 条受控模拟用例） | 达到 Gate 2 建议值 |
| Full 相对 w/o Role 引用正确率增益 | +0.0833（1.0000 对 0.9167） | Full 严格更优 |
| Full 相对 w/o Field 引用正确率增益 | +0.0833（1.0000 对 0.9167） | Full 严格更优 |
| 字段覆盖率相对最强基线增益 | 0 个百分点（SetR 同为 1.0000） | 未达到至少 5 个百分点 |

Candidate 指标来自生成器明确标记的模拟金标准，不能外推为真实业务表现。FRC 指标来自仓库内 3 条小规模工程标注集；它证明实现可复现，但样本规模不足以形成论文级或生产结论。

PostGIS v2 数值、两条 Chromium 生产路由验收和浏览器截图归档来自 GitHub Actions 受控容器运行 `29202239706`（提交 `c62b877`）。该结果证明模拟 Schema 的 DDL、空间扩展、12 类单向投影、重复执行、对账及容器化新版/回退页面可运行，不代表真实生产库已经切换，也不替代生产密钥、权限、性能、恢复或真实岗位验收。

受控性能数值来自 Windows 单进程 FastAPI TestClient，包含加密 SQLite、可信身份签名和 nonce 防重放；预算文件为 `benchmarks/controlled_performance_budget.json`，原始报告位于 `output/performance/`。它只冻结仓库回归预算，不包含网络、TLS、反向代理、外部 IdP、生产 PostgreSQL、多主机并发或真实数据规模。

GitHub Actions Linux 运行 `29196681614` 对同一预算复测通过：event list、dashboard、timeline、audit trail 的 P95 分别为 3.647、18.658、28.579、19.848 ms，四条路径错误率均为 0。

## Gate 判定

- Gate 0：`GO（受控模拟）`。来源、版本、种子、数据来源、模拟标记、家族隔离及 24 场景八类验收字段均由测试校验。
- Gate 1：`GO（受控模拟）`。当前规则候选链达到建议阈值，仍强制人工确认。
- Gate 2：`NO-GO`。`run_rag_evaluation.py` 现在逐项生成机器可读 `gate_2` 判定；Full 在引用正确率上严格优于 w/o Role 和 w/o Field，引用正确率、无依据率和受控冲突 F1 也达到建议值，但字段覆盖率相对最强 SetR 基线没有达到至少 5 个百分点，且样本仅为 3 条内部工程标注。系统默认保持 `SHADOW`，`DEFAULT/CANARY` 必须由事件级 `feature.frc_rag_formal_enabled` 显式放行。
- Gate 3：`GO（自动化安全不变量）`。未审批、越权、非法迁移、职责分离、证据/规则/审批/下发哈希和幂等 Outbox 有自动测试。
- Gate 4：`CONDITIONAL NO-GO`。受控业务闭环、本地恢复、进程内 API 冻结预算和三套已知依赖漏洞审计通过，但生产网络/并发性能预算、真实 IdP/TLS/KMS、异地主机恢复与第三方渗透测试尚无外部运行证据。

## Migration Gate 判定

| 门禁 | 状态 | 证据或缺口 |
|---|---|---|
| M0 基线冻结 | GO | 基线提交 `2db016a`、资产和 RAG 输出已记录 |
| M1 资产盘点 | GO | `asset_inventory.md` |
| M2 架构边界 | GO（响应域） | Core API 唯一写响应新表；Legacy Adapter 只读并记调用日志 |
| M3 迁移演练 | GO（本地模拟） | 幂等、隔离、计数、哈希、备份/恢复自动测试 |
| M4 影子验证 | GO（机制）/ NO-GO（算法切换） | 双轨同输入差异已落入证据包；Gate 2 未通过 |
| M5 工作流安全 | GO | 自动化不变量测试通过 |
| M6 模块切换 | CONDITIONAL | 功能开关、审计和人工接管已实现；生产路由稳定性未验证 |
| M7 旧链路退役 | NO-GO | 需要真实流量归零、在途事件清空和生产账号撤权证据 |

因此，当前仓库完成的是首期受控模拟系统的本地可验证迭代；FRC 正式切流、生产部署和旧链路物理退役不得宣称完成。
