# ACG 番剧人物识别 YOLO 训练实施计划

**目标：** 对用户上传的番剧截图或人物图像，识别画面中的已收录人物姓名，并返回其所属番剧；支持单人、多人和未知人物。

**架构：** 采用“人物区域检测 → 人物裁剪分类 → 人物与番剧映射 → 结果聚合”的两阶段闭集识别。人物分类负责姓名，人物映射负责番剧归属；对于同一人物跨季、剧场版或衍生作出现的情况，返回系列归属与候选作品，不用人物标签强行推断具体季度。

**技术栈：** Python 3.11、PyTorch、Torchvision、Ultralytics YOLO、OpenCV、Pillow、ONNX Runtime、pytest。

**需求依据：** 2026-08-31 确认将原“仅识别番名”的目标升级为“识别番剧及人物姓名”。现有具名人物数据可支撑闭集原型；正式模型仍需完成数据许可复核、人物簇命名和人工抽查。

## 1. 可行性结论与版本边界

- 🟢 可实施：AOT Characters Dataset 提供按人名组织的 14 名角色、4,041 张高清截图，可直接用于具名人物识别试点。
- 🟢 可扩展：BangumiBase 提供大量按番剧组织的真实截图和人物聚类；聚类编号不是人名，必须通过角色资料与人工审核生成稳定的 `character_id` 和标准人名映射后才能训练。
- 🟡 可辅助：DAF:re、DeepGHS Character Similarity 和 Danbooru2023 具有人物身份或角色标签，但以立绘、同人图为主，只用于预训练、困难样本补充或域外测试，不进入正式截图测试集。
- 🟡 仅作元数据：AniPersonaCaps 可补充番名、人物名、别名和参考头像映射，但每个人物图像过少，不作为主要训练数据。
- 🔴 不作为人物主数据：`diraizel/anime-images-dataset` 只有番剧级标签；Anime-2026 当前标签映射、数据卡和许可信息不足。二者不得作为正式人物分类依据。
- V1 是**受控类别集合的闭集识别系统**，不是“识别全世界所有动漫人物”。类别外人物、真人、无人物画面必须能够返回 `unknown`，不得硬猜姓名。
- V1 的 `anime_id` 表示规范化系列/作品归属；若需求必须细分季度、剧场版或衍生作，需增加整图番剧分类分支及独立数据标注，不只依赖人物姓名映射。

## 2. 数据源与用途

| 数据源 | 在本计划中的用途 | 必做处理 | 禁止用途 |
| --- | --- | --- | --- |
| [AOT Characters Dataset](https://data.mendeley.com/datasets/fzrj2xc9rt/2) | 1 部番剧、10–14 名具名角色的端到端试点 | 校验目录名、类别数量、重复图及使用条款 | 不把试点结果外推为跨番剧能力 |
| [BangumiBase](https://huggingface.co/BangumiBase/datasets) | 正式模型的真实番剧截图、人物簇和候选裁剪 | 人物簇映射到标准姓名；逐类抽查；按剧集分组 | 未完成人名映射的聚类编号不得当作最终类别名 |
| 自有合法来源视频/截图 | 补齐长尾角色，建立冻结测试集和片头片尾压力集 | 保留来源、集数、帧时间和权利信息 | 原始媒体不得进入提交包 |
| [AniPersonaCaps](https://huggingface.co/datasets/mrzjy/AniPersonaCaps) | 人物名、番名、别名和参考头像的辅助映射 | 与人工审核结果交叉核对 | 不单独支撑训练或验收 |
| [DAF:re](https://github.com/arkel23/animesion)、[DeepGHS Character Similarity](https://huggingface.co/datasets/deepghs/character_similarity)、[Danbooru2023](https://huggingface.co/datasets/nyanko7/danbooru2023) | 可选预训练、风格增强和域外鲁棒性测试 | SFW、单人物、角色标签、许可和来源过滤 | 不进入正式截图验证集/测试集 |

所有数量以下载后的审计报告为准，网页标称规模不能替代本地清点。任何许可不明、来源不可追溯或明显误标的数据默认排除。

## 3. 数据与标签规范

- 稳定标识使用 `anime_id`、`character_id`，中英文显示名仅作可变元数据；人物别名不得成为重复类别。
- 人物与番剧采用显式映射表；一个人物可关联多个作品，一个作品可关联多个人物。
- 每个标注框对应一名可辨识人物。遮挡严重、背影、过小、变装无法可靠命名的实例标记为 `ignore`，而不是猜测标签。
- `unknown` 校准集必须包含未收录番剧人物、收录番剧的未收录配角、真人、海报/周边、纯场景和检测误框。
- 清单字段至少包含：

  `sample_id,path,crop_path,anime_id,anime_name_zh,character_id,character_name_zh,character_aliases,character_cluster_id,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax,source_dataset,source_url,source_group,source_episode,frame_time_ms,sha256,phash64,duplicate_cluster_id,split,is_op_ed,is_fanart,is_unknown,review_status,license`

- 固定随机种子 `20260831`，数据切分目标为 `70/15/15`。先按 `source_group`（数据集＋作品＋剧集/视频/原始帖子）分组，再分配训练、验证、测试集合。
- SHA256 精确去重；64 位 pHash 汉明距离 `<= 4` 的样本归入同一重复簇。来源组、连续镜头和重复簇均不得跨集合。
- 正式测试集只包含真实动画截图，并在首次训练前冻结；同人图、立绘和网页抓图单独报告域外指标。

## 4. 训练与推理设计

### 4.1 人物检测

- 优先评估已有动漫人脸/头部检测权重；只有在目标截图漏检率不能达标时，才基于人工框选子集微调 YOLO 检测器。
- 检测输出人物框及置信度；多人画面逐框识别。无检测结果时返回空 `detections` 和可解释状态，不执行整图硬分类。
- 建立至少 500 张、覆盖单人/多人/遮挡/侧脸/片头片尾的人工框选验证集，用于选择检测阈值和 NMS 参数。

### 4.2 人物姓名分类

- 试点采用 `yolo26n-cls.pt`、`imgsz=224`、`epochs=30`、`patience=10`。
- 正式采用 `yolo26s-cls.pt`、`imgsz=320`、`epochs=100`、`patience=20`、`cos_lr=true`、`dropout=0.1`。
- 共用训练参数：`batch=0.70`、`device=0`、`workers=4`、`amp=true`、`optimizer=auto`、`deterministic=true`、`cache=disk`。
- 裁剪图按人物框扩边后等比缩放并填充为正方形；训练、验证、推理使用同一预处理，不使用会裁掉发型、饰品或服装特征的默认中心裁剪。
- 训练支持类别均衡采样、标签平滑、`last.pt` 断点续训和验证集早停。不得用测试集选阈值、调超参或筛选类别。

### 4.3 未知人物与番剧归属

- 在验证集上依据最大概率、Top-1/Top-2 间隔和温度校准选择拒识阈值；低于阈值时输出 `unknown`。
- 姓名分类结果通过 `character_anime_map.json` 映射到规范化番剧系列；存在多个合法作品归属时返回 `anime_candidates`。
- 若后续要求识别具体季度，在不改变人物分支的前提下增加整图番剧分类器，并用验证集学习融合规则；该扩展不属于 V1 必验范围。

### 4.4 预测接口

```json
{
  "model_version": "...",
  "class_map_version": "...",
  "detections": [
    {
      "bbox": [0, 0, 100, 100],
      "character_id": "...",
      "character_name": "...",
      "character_confidence": 0.95,
      "is_unknown": false,
      "anime_id": "...",
      "anime_title": "...",
      "anime_candidates": []
    }
  ],
  "latency_ms": {
    "detection": 0,
    "classification": 0,
    "total": 0
  }
}
```

## 5. 数据规模与验收门槛

- 试点：使用 AOT Characters Dataset，至少 10 名人物、每类清洗后不少于 120 张；先完成 3 类 2 轮冒烟，再完成全部合格类别的 30 轮训练。
- 正式：至少覆盖 10 部番剧、50 名主要人物；每类清洗后硬下限 150 张真实截图，目标不少于 300 张。未达硬下限的类别不进入正式闭集。
- 人工抽查：每类按固定种子抽查不少于 30 张，误标率必须 `<= 5%`；人名映射须双人复核，冲突进入待定队列。
- 人物检测：人工框选验证集上 Recall@IoU0.5 `>= 0.90`，并报告 Precision、mAP50 和多人画面召回率。
- 已知人物分类：冻结截图测试集 Top-1 `>= 0.80`、Top-3 `>= 0.92`、Macro-F1 `>= 0.78`，同时报告逐类召回率和类不均衡。
- 端到端人物命名：预测框 IoU `>= 0.5` 且人名正确才算命中，F1 `>= 0.75`。
- 番剧系列归属：在姓名识别正确的样本上 Top-1 `>= 0.90`；多归属人物按候选集合命中计算，并单独报告。
- 未知拒识：在已知/未知平衡测试集上 F1 `>= 0.80`，同时报告误接收率、误拒绝率和最终阈值。
- 延迟：RTX 4060 Laptop GPU 上预热 100 张后报告单人、多人场景的 P50/P95；本阶段先记录基线，不预设未经实测的硬阈值。

## 6. 实施任务

### Task 0：同步需求与设计文档

- 更新 `docs/基于YOLO的动漫番剧截图识别与管理系统-项目需求规格说明书-V1.0.docx`：删除“不做人物识别”的旧边界，补充闭集范围、多人输出、未知拒识和验收指标。
- 更新 `docs/基于YOLO的动漫番剧截图识别与管理系统-系统设计说明书-V1.0.docx`：加入检测、裁剪分类、人物—番剧映射及错误状态的数据流。
- 文档评审通过前不开始正式数据抓取和训练；本次仅修改实施计划，不修改上述 Word 文档。

### Task 1：工程骨架、契约与单元测试

- 创建 `training/pyproject.toml`、`training/src/anime_yolo/`、`training/tests/`，提供 `anime-yolo` CLI 入口。
- 先在 `training/tests/test_manifest.py`、`test_class_map.py`、`test_dedupe.py`、`test_split.py`、`test_prediction_schema.py` 编写失败测试，再实现对应模块。
- 实现 `training/src/anime_yolo/manifest.py`、`class_map.py`、`dedupe.py`、`split.py`、`prediction_schema.py`。
- 建立 `training/configs/pilot-aot.yaml`、`baseline.yaml`、`formal.yaml` 和锁定的直接依赖清单。

### Task 2：数据接入、人物映射与审计

- 实现 `training/src/anime_yolo/prepare.py` 与数据源适配器，生成统一清单，不把原始数据复制到版本库。
- 实现 BangumiBase 的“人物簇 → 候选标准人名 → 人工复核 → 稳定 ID”流程；输出 `character_map.json`、`character_anime_map.json`、审核 CSV/HTML 和冲突报告。
- 实现无效图、重复图、类别失衡、来源泄漏、许可缺失、NSFW、误标和域类型审计。
- 测试覆盖：缺失字段、非法框、同名异人、人物别名合并、多番剧归属、重复簇跨集合和许可缺失拒绝。

### Task 3：人物检测与裁剪流水线

- 实现 `training/src/anime_yolo/detection.py`、`crop.py` 和对应测试，统一框扩边、最小尺寸、NMS、多人顺序及无人物行为。
- 对至少两组候选动漫人物检测权重运行固定验证集评估；记录来源、版本、许可和指标，按召回率优先选择。
- 若均未达检测验收门槛，再创建 YOLO 检测标注与微调配置；不得在没有对比证据时直接自训检测器。

### Task 4：人物分类训练

- 实现 `training/src/anime_yolo/dataset.py`、`preprocess.py`、`trainer.py`，自定义 Ultralytics ClassificationDataset/Trainer/Validator，保证 Letterbox 一致。
- 先运行 3 类 2 轮冒烟并验证类别映射、断点续训和确定性；再运行 AOT 试点；试点通过后才扩展 BangumiBase 正式类别。
- 保存每次运行的配置、代码版本、数据清单哈希、类别映射版本、随机种子、环境和完整指标。

### Task 5：评估、阈值校准与错误分析

- 实现 `training/src/anime_yolo/evaluate.py`、`calibrate.py` 和相应测试。
- 输出检测指标、Top-1、Top-3、Macro-F1、逐类指标、混淆矩阵、可靠性图、未知拒识指标、端到端命名 F1 和错误样例。
- 分别报告真实截图、片头片尾、多人遮挡、同人/立绘域外集；总指标不得掩盖任一关键子集失败。
- 冻结测试集只在最终候选模型与阈值确定后运行一次；任何返工必须记录新的模型版本。

### Task 6：推理、CLI 与模型打包

- CLI 子命令：`prepare`、`audit`、`detect`、`train`、`evaluate`、`calibrate`、`predict --image`、`package`。
- 实现 `training/src/anime_yolo/predict.py`、`cli.py`、`package.py`，覆盖单人、多人、无人、未知人物、坏图和 GPU 不可用路径。
- 模型包包含：检测权重、`best.pt`、`last.pt`、ONNX、`character_map.json`、`character_anime_map.json`、`preprocess.json`、`thresholds.json`、配置、依赖锁、`dataset_audit.json`、`metrics.json` 及图表。
- 原始图片和视频不得进入提交包；打包检查发现原始媒体、绝对路径、密钥或缺失许可记录时必须失败。

### Task 7：完整验证与交付门禁

- 运行全部单元测试、CLI 冒烟、确定性复跑、断点续训、ONNX 导出与 ONNX/PyTorch 一致性检查。
- 按第 5 节逐项核验数据规模、人工抽查、冻结测试集指标和未知拒识；任一硬门槛未达即标记为未验收，不得只展示最好类别。
- 交付训练说明、数据来源/许可清单、已知类别清单、限制、复现实验命令和模型卡；明确 V1 只能识别已收录人物。

## 7. 执行顺序与停止条件

1. 先完成 Task 0 的需求/设计同步，再进行 AOT 试点。
2. AOT 试点验证“检测 → 姓名 → 番剧映射 → unknown”闭环后，再批量映射 BangumiBase 人物簇。
3. 任一数据源无法确认许可、人物映射抽查误标率超过 5%，或来源分组无法阻止泄漏时，停止该数据源进入正式集。
4. 正式模型未通过冻结测试集全部硬门槛时，只能交付为实验原型，不能标记为验收完成。
