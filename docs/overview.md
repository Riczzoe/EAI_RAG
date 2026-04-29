项目整体结构。

```text
rag_rare_object_project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── dataset.yaml
│   ├── kb.yaml
│   ├── qdrant.yaml
│   ├── inference.yaml
│   ├── evaluation.yaml
│   └── experiment/
│       ├── baseline_no_rag.yaml
│       ├── rag_text_only.yaml
│       ├── rag_image_clean.yaml
│       ├── rag_image_perturbed.yaml
│       ├── rag_text_image_clean.yaml
│       └── rag_text_image_perturbed.yaml
│
├── data/
│   ├── raw/
│   │   ├── imagenet_val/
│   │   ├── imagenet_wiki/
│   │   └── mappings/
│   │       └── LOC_synset_mapping.txt
│   │
│   ├── interim/
│   │   ├── sampled_kb_images/
│   │   ├── attack_images/
│   │   ├── query_split/
│   │   └── perturbation_preview/
│   │
│   └── processed/
│       ├── kb/
│       │   ├── entries.jsonl
│       │   ├── images/
│       │   └── descriptions/
│       └── queries/
│           ├── query_metadata.jsonl
│           └── images/
│
├── storage/
│   └── qdrant/
│       ├── collections/
│       ├── snapshots/
│       └── state/
│
├── src/
│   ├── utils/
│   │   ├── io.py
│   │   ├── synset_map.py
│   │   └── metrics.py
│   │
│   ├── data/
│   │   ├── extract_description.py
│   │   └── split_kb.py
│   │
│   ├── kb/
│   │   ├── build_kb.py
│   │   ├── perturbation_generator.py
│   │   └── upsert_qdrant.py
│   │
│   ├── qdrant/
│   │   └── client.py
│   │
│   ├── rag/
│   │   ├── prompt_builder.py
│   │   ├── context_builder.py
│   │   └── condition_manager.py
│   │
│   ├── models/
│   │   ├── vlm_interface.py
│   │   └── captioner.py
│   │
│   ├── evaluation/
│   │   ├── eval_retrieval.py
│   │   ├── eval_downstream.py
│   │   ├── eval_pipeline.py
│   │   └── judge.py
│   │
│   └── pipelines/
│       ├── run_build_kb.py
│       ├── run_sync_qdrant.py
│       ├── run_retrieval.py
│       ├── run_inference.py
│       ├── run_experiment.py
│       └── run_analysis.py
│
├── scripts/
│   ├── prepare_imagenet.sh
│   ├── build_kb.sh
│   ├── sync_qdrant.py
│   ├── run_baselines.sh
│   ├── run_rag_conditions.sh
│   └── evaluate_all.sh
│
├── outputs/
│   ├── retrieval/
│   │   ├── qdrant_raw/
│   │   ├── merged_results/
│   │   └── reranked_results/
│   ├── inference/
│   │   ├── no_rag/
│   │   ├── text_only/
│   │   ├── image_clean/
│   │   ├── image_perturbed/
│   │   ├── text_image_clean/
│   │   └── text_image_perturbed/
│   ├── logs/
│   └── reports/
│       ├── tables/
│       ├── figures/
│       └── summary.md
│
├── notebooks/
│   ├── 01_check_dataset.ipynb
│   ├── 02_inspect_kb.ipynb
│   ├── 03_qdrant_retrieval_analysis.ipynb
│   └── 04_result_visualization.ipynb
│
└── docs/
    ├── project_plan.md
    ├── kb_design.md
    ├── qdrant_schema.md
    ├── experiment_protocol.md
    └── result_template.md
```

## 0. 整个项目的主流程

这套结构大致对应下面这条流水线：

1. 从 `data/raw/` 读取 ImageNet 原始数据和映射文件
2. 用 `src/data/` 里的脚本把原始数据整理成“类别级条目”和“查询集”
3. 用 `src/kb/` 构建知识库条目、生成描述、生成扰动图
4. 用 `src/kb/upsert_qdrant.py` 读取 `entries.jsonl`，把类别描述编码成稠密向量并写入本地 Qdrant
5. 用 `src/retrieval/` 做图像/文本检索与融合
6. 用 `src/rag/` 构造 prompt，把参考图和描述注入模型输入
7. 用 `src/models/` 调用下游模型推理
8. 用 `src/evaluation/` 统计检索和回答指标
9. 把结果放到 `outputs/`，用 `notebooks/` 做分析，用 `docs/` 记录设计和协议

------

## 1. 根目录下的文件

1. `README.md`

2. `requirements.txt`：依赖

3. `.gitignore`：忽略：

    - `data/raw/`
    - `outputs/`
    - `__pycache__/`
    - 大模型缓存
    - 各种中间结果

------

## 2. `configs/`：所有实验参数

不把参数写死在代码里。

1. `dataset.yaml`

    - 原始 ImageNet 路径
    - devkit 路径
    - mapping 文件路径
    - 每类抽几张做知识库
    - 每类保留几张做 query
    - 随机种子
    - 是否只保留一张 KB 参考图

2. `kb.yaml`

    - 描述生成方式
    - 是否只保留一张图
    - 是否生成 clean / perturbed 双版本
    - 扰动参数

3. `qdrant.yaml`

    - 本地 `storage_path`
    - collection 名称
    - `entries.jsonl` 路径
    - embedding model
    - 距离函数（cosine / dot / euclidean）
    - 是否重建 collection
    - batch size
    
4. `inference.yaml`
    - 下游模型名
    - prompt 模板
    - 是否注入文本
    - 是否注入图片

5. `evaluation.yaml` 定义评测策略

6. `configs/experiment/*.yaml` 每个实验条件一个配置文件，方便批量跑。


------

## 3. `data/`：数据目录

1. `data/raw/`：原始数据，不改。

    - 原始 ImageNet val
    - ImageNet Wiki
    - 原始映射文件

2. `data/interim/`：中间结果。

    - 抽样出来作为 KB 候选的图。
    - 扰动图
    
3. `data/processed/`：最终喂给系统的正式数据。

    - kb/entries.jsonl: 知识库主表，通常一行一个类别条目，包含：
      - entry_id
      - synset_id
	  - class_name
	  - description
	  - image_path
	- kb/images 保存每个类别的参考图(包括投毒的样本)。
	- kb/descriptions/ 保存每个类别对应的描述文本。每类一个 txt
	- v1 中不单独导出 `qdrant_payload/points.jsonl`，而是直接从 `entries.jsonl` 写入本地 Qdrant
	
	复现实验时，只需要保留 `processed/` 。

------

## 4. `storage/` qdrant 存储目录

## 5. `src/utils/`：通用工具函数。

1. ``io.py``：负责输入输出。

   - 读写 json/jsonl/yaml
   - 读写文本
   - 路径检查

2. `synset_map.py`：map wind 和类名。

4.  `metrics.py`负责通用指标函数。


## 6. `src/data/`：数据预处理脚本

 将 raw 数据整理成 processed 数据。

1. `extract_description.py`
    - 从 Imagenet-wiki中为每一类提取出描述

2. `split_kb.py`
    - 每类抽 1 张做 KB


------

## 7. `src/kb/`：负责知识库构建。


1. `build_kb.py`：知识库构建主入口。

    - 读取 KB 抽样结果
    - 生成标准知识条目
    - 写入 `entries.jsonl`
    - 组织 descriptions 和 images 路径

3. `perturbation_generator.py`: 负责把 clean 图变成 perturbed 图。
    - 从 clean reference image 生成 perturbed version
    - 保存到 `images_perturbed/`
    - 记录扰动元信息

4. `upsert_qdrant.py`: 把 `entries.jsonl` 中的知识条目写入 Qdrant。
    - 连接 Qdrant
    - 创建或检查 collection
    - 使用 `description` 做 dense text embedding
    - upsert dense vector points
    

## 8. `src/qdrant/` 负责 Qdrant 基础设施。

1. `client.py` 封装 Qdrant 客户端。

   - 统一创建连接

   - 统一 collection 创建与重建

   - 统一 batched upsert 入口

   - 检索数据库

2. `__init__.py`: 导出 Qdrant 基础设施入口。

------

## 9. `src/rag/`：上下文注入模块

这是“把知识条目变成 prompt 输入”的地方。

1. `prompt_builder.py`:统一生成 prompt 文本。

    - 定义统一的提示模板
    - 把 query、类别描述、任务指令拼成 prompt

2. `context_builder.py`：负责组织上下文对象。

    -  决定是否加入 text
    - 决定是否加入 clean/perturbed image
    - 把 retrieval 结果转成模型可消费的上下文结构

3. `condition_manager.py`:管理不同实验条件：

    - no_rag
    - text_only
    - image_clean
    - image_perturbed
    - text_image_clean
    - text_image_perturbed

    这样整个实验就不乱了。

------

## 11. `src/models/`：下游模型接口

不要把模型调用写散在各处。

1. `vlm_interface.py`: 统一 VLM 推理接口。

2. `captioner.py`: 如果要给 query 图生成 caption。


------

## 12. `src/evaluation/`：评测模块

1.  `eval_retrieval.py`: 评测：

    - top-1
    - top-5
    - recall
    - MRR

2. `eval_downstream.py`: 评测下游模型性能：
    - 最终识别准确率
    - 条件间性能差异
    - 统计图文注入对结果的影响
    
3. `eval_pipeline.py`: 整合：
    - 先检索
    - 再注入
    - 再推理
    - 再统计
    
4. `judge.py`: 如果后面输出是自然语言而不是类名，这里做规则匹配或 LLM judge。

------

## 13. `src/pipelines/`：一键运行入口

让项目能“像系统一样运行”，而不是一堆零散脚本。

1. `run_build_kb.py`: 一键构建知识库

2. `run_retrieval.py`: 单独跑检索实验

3. `run_inference.py`: 单独跑下游推理

4. `run_experiment.py`: 一键跑某个实验配置

5. `run_analysis.py`: 汇总结果、导出表格

------

## 14. `outputs/`：所有实验产物

1. `outputs/retrieval/`

    - 检索结果
    - top-k 候选
    - 分数

1. `qdrant_raw/`：保存 Qdrant 直接返回的原始检索结果。

2. `outputs/inference/`
    - 不同条件下的模型输出
    - prompt 副本
    - 回答结果
    
3. `outputs/logs/`：日志文件

4. `outputs/reports/`

    - 表格
    - 图
    - 总结 markdown

------

## 13. `notebooks/`：分析和可视化


1. `01_check_dataset.ipynb`：数据检查 notebook。

    - 查看每类图片是否正确
    - 检查抽样、类别分布、映射关系

2. `02_inspect_kb.ipynb`：知识库检查 notebook。

    - 看 `entries.jsonl` 是否合理
    - 查看 clean/perturbed 图
    - 查看描述质量

3. `03_qdrant_retrieval_analysis.ipynb`：检索分析 notebook。

    - 可视化 Qdrant 返回结果
    - 分析错误案例
    - 看 image/text/fusion 差异

4. `04_result_visualization.ipynb`：最终结果可视化 notebook。

    - 画表、画图、做 case study

------

## 14. `docs/`：文档

1. `project_plan.md`：项目规划文档。

    - 写阶段目标
    - 写里程碑
    - 记录未来要做的实验

2. `kb_design.md`：知识库设计文档。

    - 记录 KB 的字段设计
    - 一类几张图
    - 描述如何生成
    - clean/perturbed 如何组织

3. `qdrant_schema.md`：Qdrant 结构文档。

    - 记录 collection 设计
    - point id 规则
    - payload 字段
    - filter 策略

4. `experiment_protocol.md`：实验协议文档。

    - 明确实验条件
    - 明确哪些是 baseline、哪些是对照组
    - 明确评测方式和统计口径

5. `result_template.md`：结果模板文档。

    - 规定结果格式
    - 方便多次实验保持一致的表格和图

------


## 15. 精简版：

```text
project/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset.yaml
│   ├── experiment.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── kb/
│   │   └── queries/
├── src/
│   ├── data/
│   │   ├── parse_imagenet.py
│   │   └── split_kb_query.py
│   ├── kb/
│   │   ├── build_kb.py
│   │   ├── description_generator.py
│   │   └── perturbation_generator.py
│   ├── retrieval/
│   │   ├── build_index.py
│   │   └── retrieve.py
│   ├── rag/
│   │   └── prompt_builder.py
│   ├── evaluation/
│   │   ├── eval_retrieval.py
│   │   └── eval_downstream.py
│   └── run_experiment.py
└── outputs/
```
