# PostGIS 影子迁移试点

该目录实现《渐进式迭代开发与升级设计》要求的 PostgreSQL/PostGIS 迁移演练，但不把试点误写为生产切库完成。

当前契约：

- SQLite 响应域仍是 Core API 唯一权威写入口；
- 迁移进程使用只读事务获取一致快照，不执行双向双写；
- 目标固定为 `flood_simulation` Schema，并以数据库约束拒绝 `is_simulated = false`；
- 加密载荷原样保留，另存密文 SHA-256 和解密后规范 JSON SHA-256 用于对账；
- 预警范围以 `geometry(Polygon, 4326)` 写入并建立 GiST 索引；
- 映射失败记录只保存来源 ID、载荷哈希和错误，不把敏感明文写入隔离表；
- 每张来源表必须同时通过数量和规范载荷哈希对账，批次才标记 `completed`。
- 当前映射版本为 `sqlite-response-to-postgis-shadow-v4`，支持 16 类响应记录，在既有事件、任务、Outbox、异步回调和风险对象主数据基础上加入 CandidateObjectListVersion；受控基础夹具仍为 12 类、28 条、0 隔离，主数据和冻结清单另有专项投影测试。

Compose 演练：

```powershell
docker compose up -d --wait
docker compose exec backend python scripts/seed_response_migration_fixture.py --db /app/data/postgis-source.db
docker compose exec backend python scripts/migrate_response_to_postgis.py `
  --source-db /app/data/postgis-source.db `
  --dsn "postgresql://flood:simulation-only-change-me@postgis:5432/flood" `
  --report /app/data/postgis-migration-report.json
```

幂等性可通过原命令再次执行验证；逻辑来源哈希和映射版本相同会复用同一批次 ID，并覆盖同一投影主键，不产生重复记录。

生产切换前仍必须完成真实数据质量、生产密钥、最小权限账号、连接池/慢查询、备份恢复、性能预算和切换回滚评审。严禁将 Compose 演示口令用于试点或生产。
