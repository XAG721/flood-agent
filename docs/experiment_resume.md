# FRC-RAG 跨电脑续研说明

## 1. 当前冻结边界

- 公开仓库：`XAG721/flood-agent`。
- 当前发布分支：`agent/refactor-system-structure`；合并完成后改为使用 `main`。
- 当前研究终点：v89 已关闭并冻结，Gate 2 保持 `NO-GO/SHADOW`。
- 下一项研究：先做 v90 独立数据源资格筛选，不在已经揭示的 IIRC 目标上继续调参。
- v88 ExtraTrees 路由模型、v89 协议、实现锁、开发/确认结果和逐例公开结果已经纳入 Git。

## 2. 克隆与大文件

```powershell
git clone https://github.com/XAG721/flood-agent.git
Set-Location flood-agent
git fetch origin
git switch --track origin/agent/refactor-system-structure
git lfs install
git lfs pull
```

`3D_visual/public/models/cityengine_scene.glb` 由 Git LFS 管理。备用 GLB、构建产物、依赖目录、公开数据集和公开模型缓存不进入仓库。

## 3. 两套 Python 环境

系统应用要求 Python 3.12 及以上：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test,rag-evaluation]"
```

已经完成的本地神经实验使用独立的 Windows/CUDA 环境：Python 3.10.20、PyTorch 2.4.1+cu124、CUDA 12.4。其冻结包清单为 `requirements/rag-experiment-win-cu124.txt`：

```powershell
py -3.10 -m venv .venv-rag
.\.venv-rag\Scripts\python.exe -m pip install --upgrade pip
.\.venv-rag\Scripts\python.exe -m pip install -r requirements\rag-experiment-win-cu124.txt
```

该文件是 Windows/CUDA 复现快照，不是跨操作系统通用锁。CPU、其他 CUDA 版本或 Linux 环境应另建锁文件，不能静默替换冻结运行时后仍声称逐字节复现。

## 4. 公开模型 revision

公开模型权重不随 Git 仓库发布，应在新电脑下载到任意 Hugging Face 缓存目录，并在命令中显式传入 `--hf-home`：

- `BAAI/bge-large-en-v1.5`：revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`；
- `BAAI/bge-reranker-large`：revision `55611d7bca2a7133960a6d3b71e083071bbfc312`；
- `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`：revision `e9c932ac1893a49ae0fc497ad6e1e86e2e39af20`。

模型文件哈希和实验参数仍以各冻结协议为准。不得把本机旧路径 `D:\RAG_test\.hf_cache` 当成新电脑的固定要求。

## 5. v89 可复现入口

以下命令只用于核验已经关闭的 v89，不得把已揭示确认结果重新包装成独立确认：

```powershell
$hfHome = "E:\hf-cache"
$python = ".\.venv-rag\Scripts\python.exe"

& $python scripts\run_iirc_low_candidate_robustness_transfer.py prepare --stage development --hf-home $hfHome
& $python scripts\run_iirc_low_candidate_robustness_transfer.py score --stage development --hf-home $hfHome
& $python scripts\run_iirc_low_candidate_robustness_transfer.py select --stage development
& $python scripts\run_iirc_low_candidate_robustness_transfer.py evaluate --stage development
```

公开 IIRC 数据按 `docs/progressive_upgrade/iirc_cardinality_transfer_source_registration_v86.json` 的来源、大小和 SHA-256 重建。v90 必须先登记新来源，再访问数据内容。

## 6. 私有文件边界

下列内容必须由操作者单独加密保存，不得提交到公开仓库：

- `.env`、`api_key.txt`、`3D_visual/cesium_token.txt` 和任何 `*.db.key`；
- CONFLICTS 的 `blind_mapping.json`、`annotation_operations_routing.json`、真实人工草稿、回执和私有合并结果；
- 含个人身份、机构数据或未脱敏业务数据的文件；
- `graduation_project.rar` 等个人完整备份。

公开盲评批次只用于分发给独立评审员。人工批次未完成前，不得生成 adherence 数值或宣称 Gate 2 已通过。
