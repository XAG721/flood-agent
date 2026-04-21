# 本地模型资源说明

本目录中的 CityEngine `.glb` 模型文件体积较大，不再纳入 Git 仓库历史。

如需本地运行三维展示端，请在本目录自行放置以下文件：

- `cityengine_scene.glb`
- `cityengine_scene.small-backup.glb`

默认配置文件 [`scene-config.json`](../scene-config.json) 仍会从 `/models/cityengine_scene.glb` 加载模型。

如果你的模型文件名或位置不同，请同步修改：

- [`3D_visual/public/scene-config.json`](/d:/graduation_project/3D_visual/public/scene-config.json)
- [`3D_visual/src/App.tsx`](/d:/graduation_project/3D_visual/src/App.tsx)
