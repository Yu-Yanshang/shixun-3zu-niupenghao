# A1：推理指令数据构造

> **BEYOND LIKELIHOOD** 项目 · 数据预处理模块  
> 作者：牛鹏浩（20235775）

本模块负责将原始 Parquet 格式的 Python 代码指令数据集转换为下游 SFT 训练可直接使用的 Alpaca 风格 JSON，同时通过可配置的过滤策略与审计输出，实现数据质量的透明化治理。

---

## 1. 模块概述

**A1 推理指令数据构造** 是整个系统的数据入口，核心目标：

- 从 HuggingFace `python_code_instructions_18k_alpaca` 读取原始数据；
- 完成字段标准化、空值清洗、MD5 去重、长度过滤；
- 提供**可选的进阶质量过滤**（AST 语法检查 + 指令完整性）；
- 用固定随机种子（42）生成 **永久不变** 的训练/验证/测试切分；
- 输出 LLaMA-Factory 所需的 Alpaca JSON 和 `dataset_info.json` 数据契约；
- 生成全套审计文件，使数据清洗过程可追溯。

最终为 A2（SFT）、A3（偏好构造）、A5（评测）提供标准化、可复现的高质量数据基础。

---

## 2. 整体流程与模块边界

### 2.1 数据处理流水线

```
原始 Parquet (18,612 条)
    │
    ├─ 1. 字段自适应标准化 → instruction / input / output
    ├─ 2. 空值过滤
    ├─ 3. MD5 精确去重
    ├─ 4. 输出长度过滤 (10 ~ 2048)
    ├─ 5. [可选] AST 语法 + 指令完整性质量过滤
    ├─ 6. 固定种子切分 (90% / 5% / 5%)
    └─ 7. 输出 Alpaca JSON + 审计文件
```

### 2.2 模块边界与下游集成

| 方向 | 对接模块 | 传递内容 |
| :--- | :--- | :--- |
| **输出 → A2** | SFT 微调（苏焜） | `code_sft_train.json`, `code_sft_valid.json`, `dataset_info.json` |
| **输出 → A5** | 推理增强评测（冯隆腾） | `code_sft_test.json`（固定测试集） |
| **输出 → A6** | 统一评测 | `data_statistics.json`, `bad_cases.json` 等审计文件 |
| **上游输入** | 原始数据 | `python_code_instructions_18k_alpaca/*.parquet` |

**关键集成约定**：所有下游模块只需通过 `dataset_info.json` 中注册的数据集名称即可加载数据，无需关心内部清洗逻辑。

---

## 3. 环境、模型与数据依赖

### 3.1 运行环境

- **Python** ≥ 3.10
- **Conda 环境**：`shixun`
- 主要依赖库：
  - `pandas`
  - `pyarrow`
  - `numpy`
- 标准库：`hashlib`, `ast`, `json`, `logging`, `pathlib`, `random`, `argparse`

无需 GPU，纯 CPU 数据处理。

### 3.2 输入数据

- 数据源：HuggingFace `python_code_instructions_18k_alpaca`
- 格式：Parquet 文件（脚本递归查找第一个 `.parquet`）
- 内容：每条样本包含指令、输入（可能为空）、输出代码。原始共 18,612 条。

---

## 4. 文件结构与输入配置

### 4.1 脚本文件

```
sft/
├── scripts/
│   ├── prepare_code_sft_data.py   # 核心流水线
│   └── prepare_data.sh           # 一键运行脚本
└── data/                         # 输出目录
```

### 4.2 命令行参数一览

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--source_dir` | str | `python_code_instructions_18k_alpaca` | 输入 Parquet 目录 |
| `--output_dir` | str | `sft/data` | 输出目录 |
| `--train_ratio` | float | `0.9` | 训练集比例 |
| `--valid_ratio` | float | `0.05` | 验证集比例 |
| `--test_ratio` | float | `0.05` | 测试集比例 |
| `--seed` | int | `42` | 随机种子（保证可复现） |
| `--min_output_len` | int | `10` | 代码最小字符数 |
| `--max_output_len` | int | `2048` | 代码最大字符数 |
| `--remove_duplicates` | flag | True | 是否启用 MD5 去重 |
| `--enable_quality_filter` | flag | False | **进阶功能**：AST 语法 + 指令完整性过滤 |
| `--debug_limit` | int | `None` | 仅处理前 N 条（快速调试） |

---

## 5. 运行命令示例

### 5.1 默认运行（快速开始）

```bash
cd /root/project
conda activate shixun
bash sft/scripts/prepare_data.sh
```

支持环境变量覆盖，例如：

```bash
TRAIN_RATIO=0.8 SEED=123 bash sft/scripts/prepare_data.sh
```

### 5.2 开启进阶质量过滤

```bash
python sft/scripts/prepare_code_sft_data.py \
    --source_dir python_code_instructions_18k_alpaca \
    --output_dir sft/data \
    --enable_quality_filter
```

### 5.3 仅处理前 100 条（调试）

```bash
python sft/scripts/prepare_code_sft_data.py --debug_limit 100
```

---

## 6. 输出文件与结果说明

### 6.1 数据产物

| 文件名 | 内容 | 说明 |
| :--- | :--- | :--- |
| `code_sft_train.json` | 9,294 条 | Alpaca 格式训练集（`[{instruction, input, output}]`） |
| `code_sft_valid.json` | 516 条 | 验证集 |
| `code_sft_test.json` | 517 条 | **固定测试集**，seed=42 保证永久不变 |
| `dataset_info.json` | 数据集注册信息 | LLaMA-Factory 直接可用的数据契约 |

### 6.2 审计产物

| 文件名 | 说明 |
| :--- | :--- |
| `data_statistics.json` | 四维度统计：规模、切分比例、过滤原因分布、长度分布 |
| `sample_preview.json` | 前 5 条训练样本预览，用于人工检查格式 |
| `bad_cases.json` | 所有被过滤样本及具体原因，便于回溯分析 |
| `quality_bad_cases.json` | 仅质量过滤的坏例，便于单独审查过滤规则 |

### 6.3 最终数据统计（正式版，对齐海报）

- 原始数据：18,612  
- 长度过滤后：18,165  
- **质量过滤后**：**10,327**  
- 训练/验证/测试：9,294 / 516 / 517  
- 训练集平均指令长度：91.5 字符  
- 训练集平均代码长度：330.8 字符  

---

## 7. 进阶功能实现说明

### 7.1 可配置的质量过滤参数

所有过滤行为均通过命令行参数暴露，用户可根据需求自由组合：

- `--remove_duplicates`：控制是否去重；
- `--min_output_len` / `--max_output_len`：灵活调整代码长度阈值；
- `--enable_quality_filter`：开关 AST 语法 + 指令完整性联合过滤。

这使得基础任务与进阶任务可以在一套代码中无缝切换，无需修改源码。

### 7.2 基于 AST 的代码语法过滤

启用 `--enable_quality_filter` 后，系统会：

1. 使用 `ast.parse(output)` 进行静态语法检查，不执行代码，安全高效；
2. 识别括括号不匹配、缩进错误、缺失冒号等常见语法错误；
3. 结合指令完整性检查（长度≥5 且包含任务关键词如 `write`, `implement`, `?` 等）。

**过滤原因被精确分类为三种**，并逐条记录在 `quality_bad_cases.json` 中：

- `invalid python syntax`  
- `instruction too short or lack task keywords`  
- `invalid python syntax; instruction too short or lack task keywords`

这种细粒度分类使人工审查和策略调优变得简单直接，完全符合进阶要求中“输出 bad_cases.json 保存被过滤样本及过滤原因”并“分析过滤规则是否合理”的目标。

### 7.3 全维度数据统计审计

`data_statistics.json` 自动化生成四维度统计：

| 维度 | 包含内容 |
| :--- | :--- |
| 规模维度 | 原始样本数、最终可用样本数 |
| 划分维度 | 训练/验证/测试集大小及占比 |
| 过滤维度 | 各过滤原因（空值、去重、长度、语法、指令）的计数 |
| 长度分布 | 训练集 instruction / input / output 的均值、中位数、四分位数 |

为 A6 统一评测报告中数据质量章节提供客观依据，也便于模型表现不佳时回溯数据分布。

### 7.4 关于任务类型均衡性

当前数据集未提供任务类别标签，暂无法实现按任务类型均衡采样。但指令完整性关键词校验间接保证了任务描述的多样性。若未来补充分类标签，可基于现有审计框架快速扩展均衡策略，并纳入 `data_statistics.json` 统计。

### 7.5 接口契约与固定评测基准

- `dataset_info.json` 将数据文件名与列映射抽象为数据集名称（如 `code_sft_train`），下游模块只需引用名称即可加载数据，**解耦了实现细节**。
- 固定种子（42）保证无论何时何地运行，`code_sft_test.json` 中的 517 条样本永远不变，使 Base、SFT、DPO、PPO 等所有模型的评测结果具备可比性。

这些设计使得 A1 不仅是一个数据清洗脚本，更是一个**可复现、可审计、可协作的数据治理组件**。

---
