# 依赖供应链安全审计

执行日期：2026-07-14。范围为 Python 隔离项目环境/PostGIS 依赖、主前端锁文件和 Cesium 前端锁文件。该审计查询公开漏洞数据库，只证明执行时已知公告状态，不替代渗透测试、源代码人工审计或生产配置评估。

## 修复前发现

- Python 工具环境：pip 25.2 和 pytest 8.4.2 命中 2026 年公告；
- 主前端：`form-data`、`ws`、Vite 和 Vitest 链含高危/严重公告；
- Cesium 前端：`protobufjs` 和 Vite 链含高危公告；
- DOMPurify、PostCSS、React Router、Babel 和旧 esbuild 另有低/中危公告。

## 修复动作

- CI 在审计前升级到 pip 26.1.2；
- 测试运行器升级为 pytest 9.0.3 以上，当前实装验证版本为 9.1.1；
- 风险对象 XLSX 导入增加 `openpyxl>=3.1.5,<4`；读取前先由标准库检查 OOXML ZIP 边界，随后使用 `read_only=True`、`data_only=True`、`keep_links=False`，并拒绝宏、外链和公式；
- 主前端升级到 Vite 8.1.4、Vitest 4.1.10、`@vitejs/plugin-react` 6.0.3；
- 主前端增加固定 Playwright 1.55.1 和 Chromium 验收；
- Cesium 前端升级到 Vite 8.1.4、`@vitejs/plugin-react` 6.0.3、`vite-plugin-static-copy` 4.1.1；
- 两份 npm 锁文件通过兼容更新清除 `form-data`、`ws`、`protobufjs`、DOMPurify、PostCSS 和 React Router 公告链。

## 当前结果

| 范围 | 工具 | 结果 |
|---|---|---|
| Python 隔离项目环境 | pip-audit 2.10.1，`.[test,postgres]`，pip 26.1.2 | 0 个已知漏洞；本地项目包因不在 PyPI 而跳过，项目代码由自动测试和其他安全检查覆盖 |
| 主前端 | npm audit，`--audit-level=high` | 0 个漏洞 |
| Cesium 前端 | npm audit，`--audit-level=high` | 0 个漏洞 |
| 主前端兼容性 | Vitest / TypeScript / Vite production build | 16/16 测试与构建通过 |
| 主前端浏览器兼容性 | Playwright / Chromium | 2/2 新版与回退页面验收通过 |
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

本机 Anaconda 全局环境不属于项目部署环境：对其直接执行 `pip-audit --local` 检出 44 个旧工具包的 162 条公告，主要来自 Jupyter、Scrapy、Bokeh、aiohttp 等非项目依赖。项目结论只采用从空 venv 安装 `.[test,postgres]` 的隔离审计；若继续将该 Anaconda 环境用于其他工作，应由环境所有者单独升级或重建，不能把项目隔离审计误当作全局环境已安全。

剩余外部门禁：第三方渗透测试、生产镜像/主机漏洞扫描、真实反向代理与 TLS 配置审查、生产 IdP/KMS 权限复核以及未知漏洞评估。
