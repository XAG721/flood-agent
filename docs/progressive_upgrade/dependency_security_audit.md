# 依赖供应链安全审计

执行日期：2026-07-12。范围为 Python 完整测试/PostGIS 环境、主前端锁文件和 Cesium 前端锁文件。该审计查询公开漏洞数据库，只证明执行时已知公告状态，不替代渗透测试、源代码人工审计或生产配置评估。

## 修复前发现

- Python 工具环境：pip 25.2 和 pytest 8.4.2 命中 2026 年公告；
- 主前端：`form-data`、`ws`、Vite 和 Vitest 链含高危/严重公告；
- Cesium 前端：`protobufjs` 和 Vite 链含高危公告；
- DOMPurify、PostCSS、React Router、Babel 和旧 esbuild 另有低/中危公告。

## 修复动作

- CI 在审计前升级到 pip 26.1.2；
- 测试运行器升级为 pytest 9.0.3 以上，当前实装验证版本为 9.1.1；
- 主前端升级到 Vite 8.1.4、Vitest 4.1.10、`@vitejs/plugin-react` 6.0.3；
- Cesium 前端升级到 Vite 8.1.4、`@vitejs/plugin-react` 6.0.3、`vite-plugin-static-copy` 4.1.1；
- 两份 npm 锁文件通过兼容更新清除 `form-data`、`ws`、`protobufjs`、DOMPurify、PostCSS 和 React Router 公告链。

## 当前结果

| 范围 | 工具 | 结果 |
|---|---|---|
| Python 完整已安装环境 | pip-audit 2.10.1，Python Packaging Advisory Database | 0 个已知漏洞；本地项目包因不在 PyPI 而跳过，项目代码由自动测试和其他安全检查覆盖 |
| 主前端 | npm audit，`--audit-level=high` | 0 个漏洞 |
| Cesium 前端 | npm audit，`--audit-level=high` | 0 个漏洞 |
| 主前端兼容性 | Vitest / TypeScript / Vite production build | 11/11 测试与构建通过 |
| Cesium 兼容性 | TypeScript / Vite production build | 构建通过 |
| Python 代码质量 | Ruff 0.15.21 | 0 个问题；CI 持续门禁 |

## 持续门禁

GitHub CI 的 `Dependency security audit` 作业每次 PR/主分支运行：

```text
pip upgrade → 安装 .[test,postgres] → pip-audit --local
npm audit --prefix frontend --audit-level=high
npm audit --prefix 3D_visual --audit-level=high
```

任何高危/严重 npm 公告或任何 pip-audit 已知漏洞都会使 CI 失败。中低危公告也会显示在日志中，并应在不破坏回归的前提下尽快处理。

剩余外部门禁：第三方渗透测试、生产镜像/主机漏洞扫描、真实反向代理与 TLS 配置审查、生产 IdP/KMS 权限复核以及未知漏洞评估。
