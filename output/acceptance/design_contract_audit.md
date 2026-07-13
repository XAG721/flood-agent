# 渐进式设计合同逐条审计

- 合同条目：89/89
- 本地/受控通过：87
- 外部 No-Go：2
- 外部条目：`18.4-10, 22.4-10`
- 总体边界：`CONTROLLED_CONTRACT_ACCOUNTED_EXTERNAL_NO_GO`

> 该报告证明 89 条显式合同均有唯一归属和证据。生产旧流量归零、无在途事件、数据归档恢复与账号撤权仍需真实生产证据。

## 状态统计

| 状态 | 条目数 |
|---|---:|
| `PASS_LOCAL` | 7 |
| `PASS_CONTROLLED` | 80 |
| `NO_GO_GATE` | 0 |
| `NO_GO_EXTERNAL` | 2 |

## 旧系统基线

- Git：`2db016af915ada1912afe91d315defe844dba09d`
- SQLite：`40198e9fe2bd9b5e55dcb9db8cc897639695bfc8668f9a58dd440dfe490b8908`
- RAG 索引：`d557eee3a09f32ae4d059bc6e88693946d606073d7603a2aeeee82b6ba43a539`
- 重建要求：离线、不写正式响应状态。

## 89 条合同账本

| ID | 原文 | 状态 | 证据组 | 原文行 |
|---|---|---|---|---:|
| `18.3-01` | 迁移批次有唯一ID和映射版本 | `PASS_CONTROLLED` | `migration_controlled` | 2205 |
| `18.3-02` | 旧记录数量、迁移成功、隔离和忽略数量可对账 | `PASS_CONTROLLED` | `migration_controlled` | 2206 |
| `18.3-03` | 迁移前后关键字段哈希或抽样校验通过 | `PASS_CONTROLLED` | `migration_controlled` | 2207 |
| `18.3-04` | 外键、唯一约束和业务不变量通过 | `PASS_CONTROLLED` | `migration_controlled` | 2208 |
| `18.3-05` | 旧数据缺失不通过默认值伪造成正式事实 | `PASS_CONTROLLED` | `migration_controlled` | 2209 |
| `18.3-06` | 模拟数据没有进入真实或匿名化数据集 | `PASS_CONTROLLED` | `migration_controlled` | 2210 |
| `18.3-07` | 新Schema备份和恢复已演练 | `PASS_CONTROLLED` | `migration_controlled` | 2211 |
| `18.3-08` | 回滚不会删除已经产生的审批、任务和审计记录 | `PASS_CONTROLLED` | `migration_controlled` | 2212 |
| `18.3-09` | 历史事件按原始版本回放 | `PASS_CONTROLLED` | `migration_controlled` | 2213 |
| `18.3-10` | 新事件不再写入旧正式状态表 | `PASS_CONTROLLED` | `migration_controlled` | 2214 |
| `18.4-01` | 新旧接口契约映射完成 | `PASS_CONTROLLED` | `api_switch_controlled` | 2218 |
| `18.4-02` | 所有调用方有清单 | `PASS_CONTROLLED` | `api_switch_controlled` | 2219 |
| `18.4-03` | Legacy Adapter有调用日志 | `PASS_CONTROLLED` | `api_switch_controlled` | 2220 |
| `18.4-04` | 新写接口具备权限、幂等、乐观锁和审计 | `PASS_CONTROLLED` | `api_switch_controlled` | 2221 |
| `18.4-05` | 旧写接口已拒绝或路由到受控命令 | `PASS_CONTROLLED` | `api_switch_controlled` | 2222 |
| `18.4-06` | 回调不能绕过状态机 | `PASS_CONTROLLED` | `api_switch_controlled` | 2223 |
| `18.4-07` | 影子请求不会产生外部副作用 | `PASS_CONTROLLED` | `api_switch_controlled` | 2224 |
| `18.4-08` | 功能开关支持按事件回退 | `PASS_CONTROLLED` | `api_switch_controlled` | 2225 |
| `18.4-09` | 错误码和前端提示完成 | `PASS_CONTROLLED` | `api_switch_controlled` | 2226 |
| `18.4-10` | 旧接口调用归零后才允许移除 | `NO_GO_EXTERNAL` | `legacy_retirement_external` | 2227 |
| `18.5-01` | 旧页面主要操作和截图已归档 | `PASS_CONTROLLED` | `page_switch_controlled` | 2231 |
| `18.5-02` | 新页面所有字段来自新版API | `PASS_CONTROLLED` | `page_switch_controlled` | 2232 |
| `18.5-03` | 页面显示数据版本、来源和STALE状态 | `PASS_CONTROLLED` | `page_switch_controlled` | 2233 |
| `18.5-04` | 权限不仅在前端隐藏，也由后端校验 | `PASS_CONTROLLED` | `page_switch_controlled` | 2234 |
| `18.5-05` | 危险操作要求二次确认和理由 | `PASS_CONTROLLED` | `page_switch_controlled` | 2235 |
| `18.5-06` | 模拟数据水印始终可见 | `PASS_CONTROLLED` | `page_switch_controlled` | 2236 |
| `18.5-07` | 页面刷新、重复提交和并发编辑测试通过 | `PASS_CONTROLLED` | `page_switch_controlled` | 2237 |
| `18.5-08` | 地图不可用时列表模式可继续 | `PASS_CONTROLLED` | `page_switch_controlled` | 2238 |
| `18.5-09` | 新页面失败可通过功能开关回退 | `PASS_CONTROLLED` | `page_switch_controlled` | 2239 |
| `18.5-10` | 回退不允许使用旧页面绕过新版状态机 | `PASS_CONTROLLED` | `page_switch_controlled` | 2240 |
| `18.6-01` | 数据库备份和迁移预演完成 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2246 |
| `18.6-02` | 当前可运行版本镜像和配置已保存 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2247 |
| `18.6-03` | 新旧服务健康检查通过 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2248 |
| `18.6-04` | 功能开关初始状态已审核 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2249 |
| `18.6-05` | 模拟端点与未来真实端点隔离 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2250 |
| `18.6-06` | baseline与FRC-RAG配置均可复现 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2251 |
| `18.6-07` | 审计、告警和Outbox运行正常 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2252 |
| `18.6-08` | 回滚负责人、触发条件和操作步骤明确 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2253 |
| `22.1-01` | 全文统一使用FRC-RAG | `PASS_LOCAL` | `name_boundary_local` | 2425 |
| `22.1-02` | 不宣称系统进行洪水预测 | `PASS_LOCAL` | `name_boundary_local` | 2426 |
| `22.1-03` | 不宣称AI替代指挥决策 | `PASS_LOCAL` | `name_boundary_local` | 2427 |
| `22.1-04` | 模拟下发不表述为真实跨部门接入 | `PASS_LOCAL` | `name_boundary_local` | 2428 |
| `22.1-05` | 已实现与待实现状态明确区分 | `PASS_LOCAL` | `name_boundary_local` | 2429 |
| `22.2-01` | 每条数据具有来源、版本和模拟标记 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2433 |
| `22.2-02` | 候选输出包含理由、缺失项和运行模式 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2434 |
| `22.2-03` | FRC-RAG两层Schema实现 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2435 |
| `22.2-04` | 证据三态贯穿数据库、接口和页面 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2436 |
| `22.2-05` | 冲突检测和适用性裁决明确分离 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2437 |
| `22.2-06` | 不用一次检索失败断言源文件没有规定 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2438 |
| `22.2-07` | 测试集按事件、文档和模板族隔离 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2439 |
| `22.3-01` | AI服务不能写审批、下发和任务状态表 | `PASS_CONTROLLED` | `business_safety_controlled` | 2443 |
| `22.3-02` | 通用状态修改接口不存在 | `PASS_CONTROLLED` | `business_safety_controlled` | 2444 |
| `22.3-03` | 审批后修改必须重新审批 | `PASS_CONTROLLED` | `business_safety_controlled` | 2445 |
| `22.3-04` | 下发载荷与审批载荷哈希一致 | `PASS_CONTROLLED` | `business_safety_controlled` | 2446 |
| `22.3-05` | 幂等和Outbox实现 | `PASS_CONTROLLED` | `business_safety_controlled` | 2447 |
| `22.3-06` | 执行、审批和核验职责分离 | `PASS_CONTROLLED` | `business_safety_controlled` | 2448 |
| `22.3-07` | 受阻、延期、改派、撤回和部分成功分支实现 | `PASS_CONTROLLED` | `business_safety_controlled` | 2449 |
| `22.3-08` | 接收、开始、完成和核验四类时限实现 | `PASS_CONTROLLED` | `business_safety_controlled` | 2450 |
| `22.3-09` | 规则、权限或审计故障时失败关闭 | `PASS_CONTROLLED` | `business_safety_controlled` | 2451 |
| `22.3-10` | 事件归档后可完整回放 | `PASS_CONTROLLED` | `business_safety_controlled` | 2452 |
| `22.4-01` | 原系统可运行基线、数据库和索引快照已冻结 | `PASS_LOCAL` | `legacy_baseline` | 2456 |
| `22.4-02` | 复用、适配、重构、重建和退役矩阵完成 | `PASS_CONTROLLED` | `engineering_controlled` | 2457 |
| `22.4-03` | 新Core API是正式业务状态唯一写入口 | `PASS_CONTROLLED` | `api_switch_controlled` | 2458 |
| `22.4-04` | Legacy Adapter不存在正式状态写入 | `PASS_CONTROLLED` | `api_switch_controlled` | 2459 |
| `22.4-05` | 新旧ID、状态和接口映射可追溯 | `PASS_CONTROLLED` | `api_switch_controlled` | 2460 |
| `22.4-06` | 数据库迁移可重复执行 | `PASS_CONTROLLED` | `migration_controlled` | 2461 |
| `22.4-07` | 数据迁移数量、哈希、外键和业务语义对账通过 | `PASS_CONTROLLED` | `migration_controlled` | 2462 |
| `22.4-08` | 影子运行和新旧差异报告完成 | `PASS_CONTROLLED` | `engineering_controlled` | 2463 |
| `22.4-09` | 功能开关、切换和回滚演练完成 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2464 |
| `22.4-10` | 旧写接口关闭且旧链路退役条件满足 | `NO_GO_EXTERNAL` | `legacy_retirement_external` | 2465 |
| `22.4-11` | Docker Compose可启动完整模拟环境 | `PASS_CONTROLLED` | `engineering_controlled` | 2466 |
| `22.4-12` | 模拟数据可一键生成且不可误发真实端点 | `PASS_CONTROLLED` | `engineering_controlled` | 2467 |
| `22.4-13` | API、规则、模型和状态机均有自动测试 | `PASS_CONTROLLED` | `engineering_controlled` | 2468 |
| `22.4-14` | 状态转换表完成穷举和属性测试 | `PASS_CONTROLLED` | `engineering_controlled` | 2469 |
| `22.4-15` | 监控、日志、备份和恢复验证完成 | `PASS_CONTROLLED` | `engineering_controlled` | 2470 |
| `22.4-16` | Go/No-Go结果来自真实运行，不填写虚构数值 | `PASS_CONTROLLED` | `engineering_controlled` | 2471 |
| `22.4-17` | 论文结论限定为受控场景中的原型验证 | `PASS_CONTROLLED` | `engineering_controlled` | 2472 |
| `23-01` | 原系统可复现，复用与退役范围有记录 | `PASS_LOCAL` | `legacy_baseline` | 2505 |
| `23-02` | 不存在新旧系统同时写同一事件 | `PASS_CONTROLLED` | `api_switch_controlled` | 2506 |
| `23-03` | 每个事件固定工作流引擎版本 | `PASS_CONTROLLED` | `business_safety_controlled` | 2507 |
| `23-04` | 数据和API迁移经过对账 | `PASS_CONTROLLED` | `migration_controlled` | 2508 |
| `23-05` | 新功能具备开关、回滚和人工接管路径 | `PASS_CONTROLLED` | `release_rollback_controlled` | 2509 |
| `23-06` | baseline仍可复现并作为FRC-RAG比较和降级策略 | `PASS_CONTROLLED` | `engineering_controlled` | 2510 |
| `23-07` | AI无正式状态写权限 | `PASS_CONTROLLED` | `business_safety_controlled` | 2511 |
| `23-08` | 未审批、越权、重复和载荷不一致任务不能进入执行 | `PASS_CONTROLLED` | `business_safety_controlled` | 2512 |
| `23-09` | 关键字段始终显示支持、冲突或缺失 | `PASS_CONTROLLED` | `data_algorithm_controlled` | 2513 |
| `23-10` | 每个正式任务可以追溯到预警版本、对象快照、原文证据、规则版本、审批人和执行反馈 | `PASS_CONTROLLED` | `business_safety_controlled` | 2514 |
| `23-11` | 模型或外部服务失败时能够安全降级或人工接管 | `PASS_CONTROLLED` | `business_safety_controlled` | 2515 |
| `23-12` | 模拟数据、接口和结果不会与未来真实环境混淆 | `PASS_CONTROLLED` | `engineering_controlled` | 2516 |

## 外部边界

- `18.4-10`：只有真实旧接口调用归零后才允许移除。
- `22.4-10`：只有真实旧写流量归零、无在途事件、归档可恢复且账号/写权限撤销后，旧链路退役条件才满足。
