# 洪水预警系统三类 RAG 详细设计

## 1. 设计目标

本设计细化系统中的三类核心 RAG：

- `预案规则 RAG`
- `案例经验 RAG`
- `区域画像 RAG`

目标不是“把所有资料都塞进向量库”，而是构建：

- 可检索
- 可解释
- 可追踪
- 可分层调用

的知识增强体系。

## 2. 总体原则

### 2.1 文档分层

每类 RAG 都不要只保存原始文档，而应拆成三层：

1. 原始文档层
2. 清洗片段层
3. 结构化索引层

含义：

- 原始文档层：保留原文件、原表格、原附件
- 清洗片段层：供向量检索和关键词检索
- 结构化索引层：供过滤、精确查询和引用

### 2.2 检索方式

每类 RAG 都建议采用混合检索：

- 向量检索：查语义相关内容
- BM25/关键词检索：查法规条款、地名、机构名、灾种名
- 元数据过滤：查行政区、时间、预警等级、适用对象

### 2.3 Chunk 原则

不要“一刀切”按固定字符长度切分。三类库要按内容类型分别切。

### 2.4 检索输出原则

每次 RAG 检索返回的结果建议包含：

- `chunk_text`
- `source_title`
- `source_type`
- `region`
- `time_scope`
- `applicable_stage`
- `confidence_hint`
- `citation_id`

这样后续 Agent 输出时可以引用依据，而不是只给结论。

## 3. 预案规则 RAG

## 3.1 主要用途

服务于：

- Orchestrator Agent
- Communication Agent
- Resource Coordination Agent

适用场景：

- 当前状态下应该执行什么流程
- 哪个等级的预警要通知哪些部门
- 哪些动作必须人工审批
- 哪些对象需要采用特定模板话术

## 3.2 适合纳入的文档类型

### 3.2.1 制度与预案类文档

- 国家防汛抗旱应急预案
- 省、市、县级山洪灾害防御预案
- 乡镇防汛预案
- 村级转移避险预案
- 部门协同联动机制文件

### 3.2.2 规则与阈值说明类文档

- 预警分级说明
- 响应级别启动条件
- 转移触发条件
- 部门职责分工表
- 重点人群转移规则

### 3.2.3 发布与沟通模板类文档

- 预警短信模板
- 广播稿模板
- 网格员通知模板
- 群众转移提示模板
- 信息报送模板

## 3.3 推荐字段设计

每条记录建议至少包含：

- `doc_id`
- `chunk_id`
- `source_title`
- `source_type`
- `issuing_authority`
- `region_level`
- `region_code`
- `region_name`
- `applicable_disaster`
- `applicable_stage`
- `warning_level`
- `target_audience`
- `effective_date`
- `expiry_date`
- `section_title`
- `section_number`
- `keywords`
- `chunk_text`
- `citation_id`
- `file_path`

### 3.3.1 关键字段说明

- `source_type`
  - 例如：国家预案 / 地方预案 / 模板 / 规则说明

- `applicable_stage`
  - 例如：Monitoring / Warning / Response / Recovery

- `target_audience`
  - 例如：政府 / 网格员 / 群众 / 学校 / 医院

- `warning_level`
  - 例如：蓝 / 黄 / 橙 / 红 / 不限

## 3.4 Chunk 切分方式

预案规则类文档最适合按“条款语义单元”切。

### 3.4.1 推荐切分单位

优先按以下边界切：

- 一级标题
- 二级标题
- 条款编号
- 表格行
- 模板段落

### 3.4.2 推荐 chunk 大小

- 正文条款：`300 到 800` 中文字
- 模板文本：按单条模板切，不必强行拼接
- 表格内容：按一行或一个逻辑块切

### 3.4.3 overlap 建议

- 条款类：`50 到 100` 中文字
- 模板类：通常不需要 overlap

### 3.4.4 不建议的切法

- 按固定 1000 字暴力切
- 把多个预警等级混在一个 chunk
- 把多个目标人群模板混在一个 chunk

## 3.5 检索策略建议

### 3.5.1 查询样例

- 当前橙色预警状态下需要通知哪些单位
- 村级转移避险的启动条件是什么
- 面向群众的红色预警通知模板

### 3.5.2 检索流程

1. 先做元数据过滤
   - `region`
   - `applicable_stage`
   - `warning_level`
   - `target_audience`
2. 再做向量 + BM25 混合召回
3. 最后 rerank

### 3.5.3 Top K 建议

- 初召回：`8 到 15`
- 重排序保留：`3 到 5`

## 4. 案例经验 RAG

## 4.1 主要用途

服务于：

- Risk Assessment Agent
- Evacuation Planning Agent
- Postmortem Agent

适用场景：

- 找相似洪水案例
- 参考历史疏散经验
- 对照本次事件做复盘分析

## 4.2 适合纳入的文档类型

### 4.2.1 历史事件类

- 历史洪水事件总结
- 山洪灾害案例报告
- 地方灾情通报
- 应急处置过程记录

### 4.2.2 复盘类

- 灾后复盘报告
- 响应评估报告
- 专家复盘纪要
- 经验教训清单

### 4.2.3 研究与经验类

- 典型流域洪水研究报告
- 历史演练总结
- 专家建议汇编

## 4.3 推荐字段设计

- `doc_id`
- `chunk_id`
- `event_id`
- `event_name`
- `event_type`
- `event_time`
- `event_region`
- `river_basin`
- `rainfall_level`
- `peak_water_level`
- `affected_population`
- `casualties`
- `damage_level`
- `response_level`
- `case_stage`
- `action_type`
- `outcome_label`
- `lessons_learned`
- `section_title`
- `keywords`
- `chunk_text`
- `citation_id`
- `file_path`

### 4.3.1 强烈建议增加的标签

- `similarity_anchor`
  - 用于描述这个案例为什么能成为相似案例
  - 例如：短历时强降雨 / 山区沟道 / 夜间突发 / 村庄转移

- `outcome_label`
  - 例如：成功转移 / 迟报 / 漏报 / 道路阻断 / 安置不足

- `action_type`
  - 例如：预警发布 / 疏散 / 救援 / 安置 / 灾后恢复

## 4.4 Chunk 切分方式

案例经验类最适合按“事件阶段”或“经验单元”切。

### 4.4.1 推荐切分单位

- 背景与成因
- 演化过程
- 响应措施
- 结果与后果
- 经验教训

### 4.4.2 推荐 chunk 大小

- 叙事类：`400 到 900` 中文字
- 教训总结类：`200 到 500` 中文字
- 案例表格类：按单条案例切

### 4.4.3 overlap 建议

- 叙事段：`80 到 150` 中文字
- 教训清单：通常不需要 overlap

### 4.4.4 特别建议

对于一个完整案例，不要只切成普通文本块，建议额外生成：

- `案例摘要 chunk`
- `经验教训 chunk`
- `响应动作 chunk`

也就是说，一个案例最好有“多视角索引”，便于不同 Agent 检索。

## 4.5 检索策略建议

### 4.5.1 查询样例

- 与当前降雨型相似的山洪案例
- 历史上道路中断情况下的转移经验
- 过去复盘中导致漏报的主要原因

### 4.5.2 检索流程

1. 先按区域、灾种、阶段过滤
2. 再按事件标签做召回
3. 再做语义相似检索
4. 最后 rerank

### 4.5.3 Top K 建议

- 初召回：`10 到 20`
- 重排序保留：`4 到 6`

## 5. 区域画像 RAG

## 5.1 主要用途

服务于：

- Risk Assessment Agent
- Evacuation Planning Agent
- Resource Coordination Agent

适用场景：

- 当前区域有哪些脆弱群体
- 哪些道路和桥梁是关键点
- 哪些安置点容量有限
- 哪些村组历史上更易受灾

## 5.2 适合纳入的文档类型

### 5.2.1 基础画像类

- 行政区画像
- 村镇基本信息表
- 人口与户籍信息摘要
- 特殊人群统计

### 5.2.2 空间设施类

- 安置点信息
- 学校、医院、养老院、桥梁、道路节点清单
- 地灾隐患点清单
- 重点基础设施清单

### 5.2.3 区域历史特征类

- 区域历史灾害频次
- 典型受灾位置
- 历史阻断道路
- 常见转移路线

## 5.3 推荐字段设计

- `doc_id`
- `chunk_id`
- `region_code`
- `region_name`
- `town_name`
- `village_name`
- `poi_type`
- `poi_name`
- `latitude`
- `longitude`
- `elevation`
- `population_total`
- `vulnerable_population`
- `facility_capacity`
- `road_accessibility`
- `historical_risk_level`
- `preferred_evac_route`
- `resource_owner`
- `contact_info`
- `update_time`
- `keywords`
- `chunk_text`
- `citation_id`
- `file_path`

### 5.3.1 对区域画像库的建议

这类库不能只做非结构化文本向量库，必须文本与结构化数据结合。

建议：

- 文本描述进向量索引
- 关键字段进结构化表
- 地理位置进 GIS 表

## 5.4 Chunk 切分方式

区域画像类最适合按“对象实体”切，而不是按段落切。

### 5.4.1 推荐切分单位

- 一个村组一条 chunk
- 一个安置点一条 chunk
- 一条关键道路一条 chunk
- 一个重点设施一条 chunk
- 一类脆弱人群画像一条 chunk

### 5.4.2 推荐 chunk 大小

- `150 到 500` 中文字最佳

原因：

- 这类内容本身就结构明确
- chunk 太大反而稀释地理和实体特征

### 5.4.3 overlap 建议

- 一般不需要 overlap

### 5.4.4 推荐写法

不要直接把数据库字段拼进去，建议为每个实体生成自然语言画像摘要，例如：

- 某村位于沟道出口，常见短时暴雨型山洪，村内老年人口比例较高，主要转移路线经过某桥梁，历史上该桥梁在强降雨条件下有中断记录。

这种摘要更利于向量检索。

## 5.5 检索策略建议

### 5.5.1 查询样例

- 当前受影响区域有哪些需要优先转移的脆弱群体
- 距离风险区最近且容量足够的安置点
- 某村的历史高风险特征和推荐转移路线

### 5.5.2 检索流程

1. 先做结构化过滤
   - `region`
   - `poi_type`
   - `capacity`
   - `road_accessibility`
2. 再做语义检索
3. 必要时叠加空间距离排序

### 5.5.3 Top K 建议

- 初召回：`5 到 10`
- 重排序保留：`3 到 5`

## 6. 三类 RAG 的 chunk 策略对比

| RAG 类型 | 主要内容 | 推荐切分单位 | 推荐大小 | overlap |
|---|---|---|---|---|
| 预案规则 RAG | 预案、规则、模板 | 条款、标题、模板单元 | 300-800 字 | 50-100 字 |
| 案例经验 RAG | 案例、复盘、经验 | 事件阶段、经验单元 | 400-900 字 | 80-150 字 |
| 区域画像 RAG | 村镇、安置点、道路、设施 | 单个实体 | 150-500 字 | 通常 0 |

## 7. 文档预处理建议

### 7.1 通用清洗

- 去页眉页脚
- 去重复标题
- 表格转结构化文本
- 统一行政区名称
- 统一时间格式
- 统一预警等级命名

### 7.2 预案规则库额外处理

- 保留条款编号
- 标记适用阶段
- 标记适用对象
- 标记地区层级

### 7.3 案例经验库额外处理

- 提取事件时间
- 提取区域
- 提取行动类型
- 提取结果标签
- 提取经验教训

### 7.4 区域画像库额外处理

- 提取实体名称
- 提取经纬度
- 提取容量和可达性
- 生成自然语言摘要

## 8. 三类 RAG 与 Agent 的调用关系

### 8.1 Orchestrator Agent

主要调用：

- `预案规则 RAG`

查询重点：

- 当前状态允许做什么
- 哪些动作要审批
- 哪些部门要同步

### 8.2 Risk Assessment Agent

主要调用：

- `案例经验 RAG`
- `区域画像 RAG`

查询重点：

- 相似受灾场景
- 当前区域高风险对象

### 8.3 Evacuation Planning Agent

主要调用：

- `案例经验 RAG`
- `区域画像 RAG`

查询重点：

- 历史转移经验
- 当前安置点与路线

### 8.4 Communication Agent

主要调用：

- `预案规则 RAG`

查询重点：

- 规范模板
- 目标人群话术

### 8.5 Resource Coordination Agent

主要调用：

- `预案规则 RAG`
- `区域画像 RAG`

查询重点：

- 资源调度规则
- 本地资源分布

### 8.6 Postmortem Agent

主要调用：

- `案例经验 RAG`

查询重点：

- 相似历史复盘
- 典型失误模式

## 9. 实现建议

### 9.1 不建议只有向量库

推荐组合：

- `PostgreSQL / PostGIS`：结构化与空间数据
- `Vector DB`：文本语义检索
- `BM25`：关键词召回

### 9.2 推荐检索顺序

1. 结构化过滤
2. 关键词召回
3. 向量召回
4. 混合合并
5. rerank
6. 返回带引用结果

### 9.3 返回格式建议

给 Agent 的最终检索结果建议统一成：

```json
{
  "query": "...",
  "results": [
    {
      "citation_id": "plan-001-sec-03",
      "source_title": "某县山洪灾害防御预案",
      "source_type": "地方预案",
      "region_name": "某县某镇",
      "applicable_stage": "Warning",
      "chunk_text": "...",
      "score": 0.91
    }
  ]
}
```

## 10. 最终建议

如果你后续准备真正实现，优先顺序建议是：

1. 先建 `预案规则 RAG`
   - 最容易出效果
   - 最容易支撑解释和通知生成

2. 再建 `区域画像 RAG`
   - 直接提升风险评估和疏散规划

3. 最后建 `案例经验 RAG`
   - 建设成本高，但对复盘和高级推理最有价值

一句话总结：

**预案规则 RAG 要按条款切，案例经验 RAG 要按事件阶段切，区域画像 RAG 要按实体对象切。**
