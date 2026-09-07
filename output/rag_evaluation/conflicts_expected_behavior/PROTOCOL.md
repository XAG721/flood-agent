# CONFLICTS expected-behavior 独立盲评说明

- 状态：`GENERATED_AWAITING_INDEPENDENT_HUMAN_ANNOTATION`
- 包 ID：`CONFLICTS-BEHAVIOR-5ABAB8ED31F83E6C`
- 用例：458；盲化回答：916
- 方法映射承诺：`28417c34d37a6afc30cc3a1eefecb9e81d01567babbdedd6fc00310c142d95a9`
- 当前工件只是待评审包，不是人工评测结果。

## 评审步骤

1. 两名评审分别复制 `annotator-template.json`，替换匿名 ID，独立完成全部条目。
2. 每个回答分别评定 expected behavior、事实依据、引用正确性和答案正确性；每项必须写理由。
3. `correct_answer=null` 时，答案正确性可填 `NOT_APPLICABLE`；其他维度仍须评定。
4. 完成后运行 `compare`，只查看一致性和分歧，不解盲方法身份。
5. 第三名独立裁决员逐条给出最终评分，再运行 `finalize`；工具验证映射 SHA-256 后才解盲汇总。
6. 任何空白、占位 ID、同一人员兼任或缺少理由都会被拒绝。

## 判定口径

- `PASS`：回答满足该维度；`FAIL`：明确不满足；`UNCERTAIN`：证据不足以判断。
- 偏好为 `A`、`B`、`TIE`、`NEITHER` 或 `UNCERTAIN`。
- A/B 每例独立随机化；包中不包含方法名、分数、角色分或选择器解释。
- 该跨领域评审不能替代真实防汛领域双专家验证，也不能单独改变 Gate 2。
