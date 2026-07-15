# A1 推理指令数据构造模块

> **作者**：牛鹏浩（20235775）  
> **模块定位**：系统最前端的数据入口层

## 功能概述

将 Hugging Face `python_code_instructions_18k_alpaca` 原始 Parquet 数据，转换为 LLaMA-Factory 可消费的 Alpaca 风格 JSON。

### 核心功能
- 字段自适应标准化（支持 prompt/task/question 等多种列名）
- 三层基础清洗：空值过滤 → MD5 去重 → 长度截断（对齐 2048）
- 进阶质量过滤（可选）：AST 语法检查 + 指令完整性校验
- 固定种子划分（90%/5%/5%，seed=42）
- 输出数据契约：`dataset_info.json`
- 审计产物：`data_statistics.json`、`bad_cases.json`、`quality_bad_cases.json`

## 运行命令

```bash
# 基础运行
python sft/scripts/prepare_code_sft_data.py

# 开启高质量过滤
python sft/scripts/prepare_code_sft_data.py --enable_quality_filter

# 一键运行
bash sft/scripts/prepare_data.sh