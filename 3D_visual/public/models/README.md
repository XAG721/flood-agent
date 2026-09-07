# 三维模型资源说明

主 CityEngine 场景通过 Git LFS 发布，普通 Git 历史只保存 LFS 指针。克隆后需安装 Git LFS 并执行：

```powershell
git lfs install
git lfs pull
```

默认场景文件：

- `cityengine_scene.glb`，SHA-256 `c7ad86b8a3688aa6e25ef0871e6d7c55260091f9fd9fd8a66498b549a5e528ec`；
- `cityengine_scene.small-backup.glb` 仅是本地回退副本，不随仓库发布。

默认配置文件 [`scene-config.json`](../scene-config.json) 仍会从 `/models/cityengine_scene.glb` 加载模型。

如果你的模型文件名或位置不同，请同步修改：

- [`3D_visual/public/scene-config.json`](../scene-config.json)
- [`3D_visual/src/App.tsx`](../../src/App.tsx)
