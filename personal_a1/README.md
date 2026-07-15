# A1：推理指令数据构造 — 基于 Qwen3-0.6B 的执行引导 Python 代码生成系统

本仓库实现 **BEYOND LIKELIHOOD** 项目的 **A1 数据治理与预处理模块**，负责将原始 Parquet 格式的代码指令数据集转换为下游 SFT 训练可直接使用的 Alpaca JSON 数据集，并配套生成数据契约、统计审计和质量追溯文件。

---

## 目录

- [模块定位](#模块定位)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [输出产物](#输出产物)
- [数据清洗流水线](#数据清洗流水线)
- [设计动机与核心问题](#设计动机与核心问题)
- [进阶功能：AST 质量过滤](#进阶功能ast-质量过滤)
- [与下游模块的集成](#与下游模块的集成)
- [依赖环境](#依赖环境)
- [项目成员与分工](#项目成员与分工)
- [团队仓库](#团队仓库)

---

## 模块定位

**A1 推理指令数据构造** 是整个系统的**数据入口层**，承担以下核心职责：

1. **格式适配**：将 HuggingFace `python_code_instructions_18k_alpaca` 的 Parquet 数据转换为 Alpaca 风格 JSON；
2. **质量门控**：通过空值过滤、MD5 精确去重、长度过滤以及可选的 AST 语法检查，确保训练样本的干净度；
3. **评测基准固化**：使用固定随机种子（`seed=42`）完成 90%/5%/5% 的训练/验证/测试划分，保证所有模型在同一测试集上公平比较；
4. **透明审计**：输出 `data_statistics.json`（四维度统计）、`bad_cases.json` 和 `quality_bad_cases.json`（过滤原因追溯）等文件，使数据清洗过程完全可解释。

---

## 系统架构

```
原始 Parquet (18,612 条)
        │
        ▼
   ┌────────────┐
   │ A1 数据治理 │  ← 牛鹏浩
   └─────┬──────┘
         │ 训练/验证/测试集 + dataset_info.json
         ▼
   ┌────────────┐
   │ A2 SFT 微调 │  ← 苏焜
   └─────┬──────┘
         │ SFT 模型
         ▼
   ┌────────────┐
   │ A3 偏好构造 │  ← 苏焜
   └─────┬──────┘
         │ chosen/rejected 偏好对
         ▼
   ┌────────────┐
   │ A4 偏好对齐 │  ← 冯隆腾 (DPO/PPO)
   └─────┬──────┘
         │ 对齐模型
         ▼
   ┌────────────┐
   │ A5 推理增强 │  ← 冯隆腾 (CoT/SC/Best-of-N/Reflexion …)
   └─────┬──────┘
         │ 最终评测结果
         ▼
   ┌────────────┐
   │ A6 统一评测 │
   └────────────┘
```

> 详细架构及数据流向请参见 [团队系统文档](https://github.com/13flix/BEYOND-LIKELIHOOD)。

---

## 快速开始

### 1. 环境准备

```bash
cd /root/project
conda activate shixun
pip install pandas pyarrow numpy
```

### 2. 一键运行（推荐）

```bash
bash sft/scripts/prepare_data.sh
```

该脚本会使用默认参数执行以下操作：
- 读取 `python_code_instructions_18k_alpaca/` 下的 Parquet 文件
- 执行字段标准化、空值过滤、MD5 去重、长度过滤
- 默认关闭 AST 质量过滤（可通过环境变量开启）
- 输出所有数据产物到 `sft/data/`

### 3. 自定义参数运行

```bash
python sft/scripts/prepare_code_sft_data.py \
    --source_dir python_code_instructions_18k_alpaca \
    --output_dir sft/data \
    --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 \
    --seed 42 --min_output_len 10 --max_output_len 2048
```

### 4. 开启进阶质量过滤

```bash
python sft/scripts/prepare_code_sft_data.py \
    --source_dir python_code_instructions_18k_alpaca \
    --output_dir sft/data \
    --enable_quality_filter
```

### 5. 调试模式（只处理前 N 条）

```bash
python sft/scripts/prepare_code_sft_data.py --debug_limit 100
```

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--source_dir` | str | `python_code_instructions_18k_alpaca` | 原始 Parquet 文件所在目录 |
| `--output_dir` | str | `sft/data` | 所有输出文件的存放目录 |
| `--train_ratio` | float | `0.9` | 训练集占比 |
| `--valid_ratio` | float | `0.05` | 验证集占比 |
| `--test_ratio` | float | `0.05` | 测试集占比 |
| `--seed` | int | `42` | 随机种子，保证划分可复现 |
| `--min_output_len` | int | `10` | 代码最小字符数，过短则过滤 |
| `--max_output_len` | int | `2048` | 代码最大字符数，与 Qwen3 cutoff_len 对齐 |
| `--remove_duplicates` | flag | `True` | 是否开启 MD5 精确去重 |
| `--enable_quality_filter` | flag | `False` | 是否开启 Python AST 语法检查 + 指令完整性校验 |
| `--debug_limit` | int | `None` | 仅处理前 N 行，方便快速验证 |

---

## 输出产物

运行成功后，`sft/data/` 目录下会生成以下文件：

```
sft/data/
├── code_sft_train.json          # 训练集（Alpaca 格式）
├── code_sft_valid.json          # 验证集
├── code_sft_test.json           # 测试集（固定种子，不可变）
├── dataset_info.json            # LLaMA-Factory 数据契约
├── data_statistics.json         # 四维度统计报告（规模、划分、过滤、长度分布）
├── sample_preview.json          # 前5条训练样本预览
├── bad_cases.json               # 所有被过滤样本及详细原因
└── quality_bad_cases.json       # 质量过滤专用坏例清单（仅在开启质量过滤时生成）
```

**最终数据量（正式版，对齐海报）**：

| 数据集 | 样本数 |
| :--- | :--- |
| 训练集 | 9,294 |
| 验证集 | 516 |
| 测试集 | 517 |
| 平均指令长度 | 91.5 字符 |
| 平均代码长度 | 330.8 字符 |

---

## 数据清洗流水线

脚本 `prepare_code_sft_data.py` 内部按以下顺序执行 9 个阶段：

1. **加载原始 Parquet**（递归查找目录内第一个 `.parquet` 文件）
2. **字段自适应标准化**：将 `prompt`/`task` 等不同列名统一映射到 `instruction`/`input`/`output`
3. **空值过滤**：剔除 `instruction` 或 `output` 为空白的样本
4. **MD5 哈希去重**：基于 `instruction + input + output` 拼接计算 MD5，移除完全重复项
5. **长度过滤**：保留 `output` 长度在 `[min_output_len, max_output_len]` 范围内的样本
6. **AST 质量过滤**（可选）：检查代码语法合法性以及指令是否包含任务关键词
7. **固定种子划分**：`seed=42`，按 90%/5%/5% 生成训练/验证/测试集
8. **保存 Alpaca JSON 与 `dataset_info.json`**
9. **审计产物输出**：统计报告、样本预览、坏例清单

每一步均通过 `logging` 输出详细处理数量，方便追踪。

---

## 设计动机与核心问题

在系统设计初期，我们识别出三个训练数据预处理的核心风险，A1 模块针对这些问题提供了明确的解决方案：

| 问题 | 风险 | A1 方案 |
| :--- | :--- | :--- |
| **噪声目标** | 语法错误、模糊指令会污染训练 | AST 语法检查 + 指令完整性校验，只保留高质量样本 |
| **划分漂移** | 不同随机划分导致模型评测不可比 | 固定种子 `seed=42`，测试集永久不变 |
| **不透明预处理** | 传统脚本无法解释过滤原因，调优无依据 | 输出 `bad_cases.json`、`quality_bad_cases.json`，逐条记录原因 |

---

## 进阶功能：AST 质量过滤

通过 `--enable_quality_filter` 开启，该过滤器会：

- 使用 `ast.parse(output)` 检查代码是否包含 Python 语法错误
- 检查 `instruction` 长度是否 ≥ 5 且包含任务关键词（如 `write`, `implement`, `debug` 等）
- 同时满足上述两个条件的样本才会保留
- 每一条被过滤样本均记录精确的失败原因（三种分类）：
  - `invalid python syntax`
  - `instruction too short or lack task keywords`
  - `invalid python syntax; instruction too short or lack task keywords`

所有质量过滤的坏例单独存储在 `quality_bad_cases.json` 中，便于验收时重点审查。

---

## 与下游模块的集成

### 数据契约：`dataset_info.json`

A1 与 A2 (SFT 微调) 之间通过 `dataset_info.json` 解耦。该文件内容如下：

```json
{
  "code_sft_train": {
    "file_name": "code_sft_train.json",
    "formatting": "alpaca",
    "columns": {"prompt": "instruction", "query": "input", "response": "output"}
  },
  ...
}
```

LLaMA-Factory 在训练配置中只需引用 `dataset: code_sft_train` 即可自动加载数据，无需关心文件路径或列名映射。

### 固定测试集

`code_sft_test.json` 由于固定种子的划分，在任意时间、任意机器上运行均可获得完全相同的 517 条测试样本。A5（推理增强评测）直接读取该文件，保证 Base / SFT / DPO / PPO 等所有模型版本在**完全相同**的条件下进行性能对比。

---

## 依赖环境

- Python 3.10+
- `pandas`
- `pyarrow`
- `numpy`
- 标准库：`hashlib`, `ast`, `json`, `logging`, `pathlib`, `random`, `argparse`

（无需额外安装深度学习框架）

---

## 项目成员与分工

| 成员 | 负责模块 | 个人仓库 |
| :--- | :--- | :--- |
| **牛鹏浩** | **A1：推理指令数据构造** | **[本仓库](https://github.com/Yu-Yanshang/shixun-3zu-niupenghao)** |
| 苏焜 | A2：监督微调；A3：偏好数据构造与评分 | [GitHub](https://github.com/HDSulfox/BEYOND-LIKELIHOOD) |
| 冯隆腾 | A4：偏好对齐；A5：推理增强；PathCoder | [GitHub](https://github.com/13flix/BEYOND-LIKELIHOOD-A4-A5) |

---

## 团队仓库

完整项目（含 A2~A6 全部模块、系统调度脚本与最终评测报告）位于：
[https://github.com/13flix/BEYOND-LIKELIHOOD](https://github.com/13flix/BEYOND-LIKELIHOOD)

---
