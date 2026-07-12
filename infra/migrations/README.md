# 响应域数据库迁移

响应域采用只向前、幂等建表与版本账本策略。实际 DDL 的单一来源为 `flood_system/storage/schema.py`；`SQLiteRepository` 在事务中执行幂等 DDL，并将完整 DDL 的 SHA-256 写入 `response_schema_migrations`。同一校验和重复启动不会产生重复版本。

执行：

```powershell
python scripts/migrate_response_schema.py --db data/flood_warning_system_v2.db
```

迁移原则：

- 不修改或删除旧 V2/V3 表；
- 新响应域只写 `response_*` 表；
- 审批、反馈、任务版本和时间线不可破坏性回滚；
- 旧事件缺少权威预警原始载荷时进入隔离清单，不用默认值伪造正式事实；
- 每个结构版本保存 DDL 校验和，检测历史版本漂移。
