# 原系统可复现基线冻结报告

- 冻结日期：`2026-07-14`
- Git 提交：`2db016af915ada1912afe91d315defe844dba09d`
- Git Tree：`11ad151eb65101d4786f1815420fb2304cf55d65`
- 原提交时间：`2026-05-12T15:09:20+08:00`
- 原提交说明：Refine AgentTwin demo map and dialog flow
- Git archive SHA-256：`c95f450964eca8a824bb445efbfaf874e6bc69820856b4d3cbf99b2bb170dc1c`

## 数据库与索引

- 从冻结提交离线启动旧系统生成 SQLite 基线：37 张表、38 个显式索引、487424 bytes。
- 数据库 SHA-256：`40198e9fe2bd9b5e55dcb9db8cc897639695bfc8668f9a58dd440dfe490b8908`。
- RAG 运行索引：22 篇文档、11527 bytes，SHA-256=`d557eee3a09f32ae4d059bc6e88693946d606073d7603a2aeeee82b6ba43a539`。
- 二进制数据库和索引仅保留在忽略的 `.cache/legacy_baseline/`，避免把运行数据或密钥提交到 Git；仓库提交完整表结构、逐表行数/内容哈希和所有重建输入 Git blob 身份。

## 重建输入

| 类别 | 文件数 | Blob bytes | 清单 SHA-256 |
|---|---:|---:|---|
| application | 188 | 1709557 | `3cd4ffadd067bba0e8c0f7c9510f2f8daa69e97d7bc58ef7ff5306548662aa98` |
| database_rebuild_inputs | 27 | 91216 | `8b312b21cce129b18bcf8db3640cfbdd98db4938176c4f41631ecde0e82c5416` |
| rag_index_rebuild_inputs | 31 | 40815 | `f8eada98ee8a20d052f887f76b94323b46bbd457aad5a59333b424a54a6012e5` |
| models_prompts_and_contracts | 7 | 85433 | `da97461f711259e096b1e07723bffc81a936edab024e6776e0e0d21a3dd7cbec` |

该报告证明受控旧系统代码、数据库重建输入、RAG 原文/索引输入、Prompt 与合同可以从指定 Git 对象离线重建。它不代表旧生产流量已经归零，也不替代生产数据库归档、账号撤权或真实恢复演练。
