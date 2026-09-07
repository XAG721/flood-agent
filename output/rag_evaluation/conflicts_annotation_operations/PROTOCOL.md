# CONFLICTS 独立评审批次作业说明

- 状态：`PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW`
- 作业 ID：`CONFLICTS-ANNOTATION-OPS-5A6EE0F6420F2251`
- 原盲评包 ID：`CONFLICTS-BEHAVIOR-5ABAB8ED31F83E6C`
- 角色：3；每角色批次：8
- 每角色主任务：458；隐藏复测：23
- 私有路由承诺：`b4818883beb4ed3e256895a2161f2641656707639ded4ee858633e5e4bb25ef3`

## 执行要求

1. `reviewer_1`、`reviewer_2` 和 `adjudicator` 必须由三名不同人员承担，不得互看决定。
2. 每人依次完成自己目录下 8 个 `submission-template`，8 份文件使用同一个私有 annotator_id。
3. 不修改 task、batch、package 或 operations ID；每项四个评分、理由和偏好都必须完成。
4. 部分题目会跨批重复，用于组内一致性质控；重复身份不在公开批次中标识。
5. 使用 `merge` 合并同一角色的 8 份表单。合并器验证私有路由哈希、恢复原 458 例顺序并输出组内一致性。
6. 组内任一评分或偏好精确一致率低于 0.80 时状态为 `REVIEW_REQUIRED`，不得进入跨人员比较。
7. 三个合并提交均通过后，再使用冻结 expected-behavior 工作流的 `compare` 和 `finalize`。

> 批次文件、空模板和质控工具不是人工结果；它们不改变 Gate 2，也不会发布方法映射。
