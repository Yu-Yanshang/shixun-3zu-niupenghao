#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
模块名称：A1 推理指令数据构造（完整版）
所属系统：基于 Qwen3 的 Python 代码生成系统（监督微调数据入口层）
作者：牛鹏浩（20235775）
最后更新：2026-07-14

================================================================================
模块定位：
    本模块处于整个系统的最前端，负责将原始的 Parquet 格式代码指令数据
    转换为 LLaMA-Factory 可直接消费的 Alpaca 风格 JSON 数据集，
    并生成稳定的训练/验证/测试划分、数据集注册信息以及质量审计文件。

================================================================================
核心职责（契约优先）：
    1. 字段标准化：将不同来源的列名统一为 instruction / input / output
    2. 数据清洗：过滤空值、去除完全重复样本、按输出长度筛选
    3. 可选进阶过滤：Python 语法检查 + 指令完整性校验（保证训练样本质量）
    4. 固定随机划分：90% / 5% / 5% 划分，种子固定（默认42），确保评测公平
    5. 输出标准格式：生成 code_sft_train/valid/test.json 及 dataset_info.json
    6. 质量审计：输出统计报告、样本预览、坏例清单，支撑可追溯性

================================================================================
验收支撑：
    - 本代码符合 proposal 中 A1 模块的所有设计原则与输出契约
    - 可独立运行，生成完整数据产物，供 A2（SFT）、A5（推理增强）、A6（统一评测）使用
    - 审计文件（data_statistics.json, bad_cases.json, quality_bad_cases.json,
      sample_preview.json）可直接用于验收展示，证明数据质量与清洗必要性

================================================================================
运行方式：
    cd /root/project
    conda activate shixun
    bash sft/scripts/prepare_data.sh

独立命令行示例：
    python sft/scripts/prepare_code_sft_data.py \
        --source_dir python_code_instructions_18k_alpaca \
        --output_dir sft/data \
        --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 \
        --seed 42 --min_output_len 10 --max_output_len 2048

开启高质量过滤示例：
    python sft/scripts/prepare_code_sft_data.py \
        --source_dir python_code_instructions_18k_alpaca \
        --output_dir sft/data \
        --enable_quality_filter

================================================================================
依赖库：
    - pandas, pyarrow: 读取 Parquet 数据
    - numpy: 随机数控制
    - hashlib: MD5 哈希去重
    - ast: Python 语法树解析（质量过滤）
    - json: 序列化输出
    - logging: 日志记录
    - pathlib: 路径操作
    - random, argparse: 随机控制与命令行解析
================================================================================
"""

import argparse
import ast                     # 用于 Python 语法树解析，检测代码是否合法
import hashlib                 # 计算 MD5 哈希，用于精确去重
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================== 日志配置 ======================================
# 配置日志格式与级别，便于跟踪流水线各阶段执行情况
# 验收时可查看运行日志，了解每一步的处理结果
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================== 字段映射字典 ==================================
# 目标字段名 -> 原始数据中可能的列名列表（按优先级排列）
# 设计目的：兼容不同版本/来源的数据集
# 例如：某版本数据中 "指令" 列可能叫 "prompt"，另一版本可能叫 "task"
# 通过候选列表，脚本可以自适应识别，无需人工修改
COLUMN_CANDIDATES = {
    "instruction": ["instruction", "prompt", "task", "question"],
    "input": ["input", "context", "additional"],
    "output": ["output", "answer", "code", "solution", "response"],
}


# ============================== 参数解析 ======================================
def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，定义所有可配置的数据处理选项。

    所有参数均提供合理的默认值，可直接运行无需手动指定。
    验收时可通过调整参数（如 --enable_quality_filter）展示进阶过滤效果。

    返回:
        argparse.Namespace: 包含所有参数值的对象
    """
    parser = argparse.ArgumentParser(
        description="A1: Parquet -> Alpaca JSON 数据处理流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值，便于查看
    )

    # ---- 路径配置 ----
    parser.add_argument(
        "--source_dir",
        type=str,
        default="python_code_instructions_18k_alpaca",
        help="包含原始 .parquet 文件的目录（脚本会递归查找）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sft/data",
        help="所有输出文件（JSON 数据、dataset_info.json、审计文件）的存放目录"
    )

    # ---- 数据集划分比例（必须和为 1.0） ----
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="训练集占比"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.05,
        help="验证集占比"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="测试集占比"
    )

    # ---- 随机种子与调试 ----
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，确保划分可复现，使不同模型版本在同一测试集上比较"
    )
    parser.add_argument(
        "--debug_limit",
        type=int,
        default=None,
        help="调试模式：仅处理前 N 行数据（用于快速验证脚本逻辑，正式运行请设为 None）"
    )

    # ---- 长度过滤参数（与 Qwen3 的 cutoff_len=2048 对齐） ----
    parser.add_argument(
        "--min_output_len",
        type=int,
        default=10,
        help="output（代码）最小字符数，低于此值视为无效样本（过滤掉）"
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=2048,
        help="output 最大字符数，超出将过滤（与后续 SFT 的 cutoff_len 保持一致）"
    )

    # ---- 去重开关 ----
    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        default=True,
        help="是否去除完全重复样本（基于 instruction+input+output 的 MD5 哈希）"
    )

    # ---- 进阶质量过滤开关（验收时可开启对比效果） ----
    parser.add_argument(
        "--enable_quality_filter",
        action="store_true",
        default=False,
        help="是否启用 Python 语法检查 + 指令完整性校验（严格模式，默认关闭）"
    )

    return parser.parse_args()


# ============================== 数据加载 ======================================
def load_parquet(source_dir: str) -> pd.DataFrame:
    """
    从指定目录递归查找并加载第一个 .parquet 文件。

    设计考虑：
        - 支持目录下可能存在多个 parquet 文件时自动选择第一个（并警告）
        - 返回 pandas DataFrame，便于后续列操作
        - 递归查找（rglob）可适应不同的目录层级结构

    参数:
        source_dir: 包含 .parquet 文件的目录路径

    返回:
        pd.DataFrame: 加载的原始数据

    异常:
        FileNotFoundError: 若未找到任何 .parquet 文件
    """
    source_path = Path(source_dir)
    # 递归查找所有 .parquet 文件
    parquet_files = list(source_path.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet file found in {source_dir}")

    # 如果找到多个，使用第一个并发出警告（避免混淆）
    if len(parquet_files) > 1:
        logger.warning(f"发现多个 parquet 文件，将使用第一个: {parquet_files[0]}")

    # 使用 pandas 读取 Parquet 文件
    df = pd.read_parquet(parquet_files[0])
    logger.info(f"成功加载原始数据，共 {len(df)} 行")
    return df


# ============================== 字段标准化 ====================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 的列名映射到标准三列：instruction, input, output。

    映射策略（按优先级从高到低）：
        1. 精确匹配：DataFrame 中恰好有目标列名（如 'instruction'）
        2. 别名匹配：按 COLUMN_CANDIDATES 中的列表顺序查找候选列名
        3. 部分匹配（不区分大小写）：遍历所有列名，检查是否包含候选词
           （例如 'my_prompt_text' 包含 'prompt'，会被映射到 instruction）

    如果任一目标字段无法找到映射，则抛出 ValueError 终止流程。
    所有缺失值填充为空字符串（保证 JSON 中无 NaN）。

    参数:
        df: 原始 DataFrame

    返回:
        pd.DataFrame: 仅包含 ['instruction', 'input', 'output'] 三列的 DataFrame
    """
    df = df.copy()
    new_map = {}  # 目标字段名 -> 实际列名

    # 遍历每个目标字段，尝试找到对应的源列
    for target, aliases in COLUMN_CANDIDATES.items():
        # ----- 情况1：精确匹配目标列名 -----
        if target in df.columns:
            new_map[target] = target
            continue

        # ----- 情况2：精确匹配候选别名 -----
        found = None
        for alias in aliases:
            if alias in df.columns:
                found = alias
                break

        # ----- 情况3：部分匹配（不区分大小写） -----
        if found is None:
            for col in df.columns:
                # 检查列名是否包含任一候选词（不区分大小写）
                if any(alias in col.lower() for alias in aliases):
                    found = col
                    # 发出警告，提醒用户确认映射是否正确
                    logger.warning(
                        f"字段 '{target}' 通过部分匹配映射到列 '{col}'，请确认是否正确"
                    )
                    break

        if found is not None:
            new_map[target] = found

    # 检查是否所有目标字段都已成功映射
    missing = [t for t in COLUMN_CANDIDATES if t not in new_map]
    if missing:
        raise ValueError(
            f"无法找到以下必须字段的映射: {missing}。"
            f"可用列名: {df.columns.tolist()}"
        )

    # 重命名列：将实际列名映射为目标列名
    df = df.rename(columns={v: k for k, v in new_map.items()})

    # 只保留所需的三列
    df = df[list(COLUMN_CANDIDATES.keys())]

    # 填充空值（尤其是 input 常为空），保证 JSON 序列化时不会出现 null
    df = df.fillna("")

    logger.info(f"字段标准化完成，保留列: {df.columns.tolist()}")
    return df


# ============================== 过滤函数 ======================================
# 每个过滤函数都返回 (保留集, 移除集) 两个 DataFrame
# 保留集用于后续处理，移除集用于审计（bad_cases.json）
# 这种设计使数据清洗过程完全透明，便于验收时追溯

def filter_empty(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    过滤掉 instruction 或 output 为空（或仅含空白字符）的样本。

    空值样本缺乏有效的监督信号，必须剔除。
    检查时使用 strip() 去除首尾空白，避免仅含空格或换行符的样本被保留。

    返回:
        - 保留的 DataFrame
        - 被移除的 DataFrame（新增 'reason' 列说明原因）
    """
    before = len(df)

    # 条件：instruction 和 output 去除首尾空白后均非空
    mask = (df["instruction"].str.strip() != "") & (df["output"].str.strip() != "")

    # 被移除的样本：添加 reason 列标记过滤原因
    removed = df[~mask].copy()
    removed["reason"] = "empty instruction or output"

    # 保留的样本
    df = df[mask].copy()

    logger.info(f"空值过滤：移除 {before - len(df)} 条，剩余 {len(df)} 条")
    return df, removed


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于 instruction+input+output 拼接字符串的 MD5 哈希进行精确去重。

    设计考虑：
        - 使用 MD5 哈希而非直接拼接文本作为键，可避免长文本导致的内存开销
        - 分隔符 "|||" 用于避免不同字段拼接时产生边界模糊（如 "abc"+"def" vs "ab"+"cdef"）
        - 保留第一次出现的样本（keep='first'），后续重复项被移除

    返回:
        - 去重后的 DataFrame（已删除临时哈希列）
        - 被移除的重复样本 DataFrame（含 'reason' 列）
    """
    before = len(df)

    # 计算每条样本的 MD5 哈希值
    # 使用 apply + lambda 逐行处理，axis=1 表示按行操作
    df["_hash"] = df.apply(
        lambda r: hashlib.md5(
            (r["instruction"] + "|||" + r["input"] + "|||" + r["output"]).encode()
        ).hexdigest(),
        axis=1
    )

    # 标记重复行（保留第一次出现，后续的标记为 True）
    dup_mask = df.duplicated(subset=["_hash"], keep="first")

    # 被移除的重复样本
    removed = df[dup_mask].copy()
    removed["reason"] = "exact duplicate"

    # 保留非重复样本，并删除临时哈希列
    df = df[~dup_mask].drop(columns=["_hash"])

    logger.info(f"去重：移除 {before - len(df)} 条，剩余 {len(df)} 条")
    return df, removed


def filter_by_length(
    df: pd.DataFrame,
    min_len: int,
    max_len: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按 output 列的字符长度筛选，保留长度在 [min_len, max_len] 范围内的样本。

    设计原因：
        - 过短的代码（如仅函数签名 "def foo(): pass"）缺乏训练价值
        - 过长的代码可能超出模型上下文窗口（与 Qwen3 cutoff_len=2048 对齐）

    参数:
        min_len: 最小字符数（默认10）
        max_len: 最大字符数（默认2048）

    返回:
        - 过滤后的 DataFrame
        - 被移除的样本 DataFrame（含 'reason' 列）
    """
    before = len(df)

    # 计算 output 长度
    df["_len"] = df["output"].str.len()

    # 构造长度掩码
    mask = (df["_len"] >= min_len) & (df["_len"] <= max_len)

    # 被移除的样本
    removed = df[~mask].copy()
    removed["reason"] = f"output length out of [{min_len}, {max_len}]"

    # 保留的样本，删除临时长度列
    df = df[mask].drop(columns=["_len"])

    logger.info(f"长度过滤：移除 {before - len(df)} 条，剩余 {len(df)} 条")
    return df, removed


# -------------------------- 进阶质量过滤（可选） ------------------------------
def check_python_syntax(code: str) -> bool:
    """
    使用 ast.parse 检查一段文本是否为合法的 Python 代码。

    设计考虑：
        - 仅检查语法正确性，不执行代码（安全且高效）
        - 能识别括号不匹配、缩进错误、缺失冒号等常见语法问题
        - 通过 ast.parse 返回 True，抛出 SyntaxError 则返回 False

    注意：此函数只检查语法，不检查逻辑错误（如死循环、算法错误）。
    逻辑正确性由后续 A5 推理增强阶段的单元测试执行把关。

    参数:
        code: 待检查的代码字符串

    返回:
        True 表示语法合法，False 表示包含 SyntaxError
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def quality_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    高质量过滤（默认关闭，由 --enable_quality_filter 开启）。

    ====================================================================
    过滤规则（两条规则同时满足才保留）：
        1. output 必须是语法正确的 Python 代码（通过 ast.parse 检查）
        2. instruction 长度 >= 5 字符 且 包含任务关键词

    ====================================================================
    reason 生成分类（严格按照验收要求的四种分类）：

        | 编号 | 触发条件 | 生成的 reason |
        |------|---------|---------------|
        | 1    | 仅指令不合格（太短或缺少关键词） | "instruction too short or lack task keywords" |
        | 2    | 仅语法错误 | "invalid python syntax" |
        | 3    | 语法错误 且 指令不合格 | "invalid python syntax; instruction too short or lack task keywords" |
        | 4    | 长度不合格（由 filter_by_length 产生，本函数不产出此分类） | "output length out of [10, 2048]" |

    ====================================================================
    目的：
        用更少但质量更高的数据实现高效 SFT。
        默认关闭以保留数据多样性，验收时可开启对比效果。

    返回:
        - 通过过滤的 DataFrame
        - 被移除的样本 DataFrame（每行含具体原因）
    """
    before = len(df)

    # ---- 规则1：语法检查 ----
    # 使用 ast.parse 检查 output 是否为合法 Python 代码
    syntax_ok = df["output"].apply(check_python_syntax)

    # ---- 规则2：指令完整性检查 ----
    # 2a. 长度至少 5 个字符
    length_ok = df["instruction"].str.len() >= 5

    # 2b. 包含任务关键词（或问号）
    # 关键词列表：write, implement, debug, fix, explain, create, define, ?（中英文问号）
    keyword_ok = df["instruction"].str.contains(
        r"[?？]|write|implement|debug|fix|explain|create|define",
        case=False,  # 不区分大小写
        na=False     # 如果值为 NaN，返回 False
    )

    # 综合指令完整性：长度合格 且 有关键词
    inst_ok = length_ok & keyword_ok

    # ---- 综合条件：语法正确 AND 指令完整 ----
    mask = syntax_ok & inst_ok

    # ---- 被过滤的样本 ----
    removed = df[~mask].copy()

    # ---- 生成详细的过滤原因（严格按照四种分类） ----
    reasons = []
    for idx in removed.index:
        has_syntax_error = not syntax_ok.loc[idx]
        has_inst_fail = not inst_ok.loc[idx]

        # 根据条件组合生成对应的 reason
        if has_syntax_error and has_inst_fail:
            # 分类3：语法错误 且 指令不合格
            reasons.append("invalid python syntax; instruction too short or lack task keywords")
        elif has_syntax_error:
            # 分类2：仅语法错误
            reasons.append("invalid python syntax")
        elif has_inst_fail:
            # 分类1：仅指令不合格
            reasons.append("instruction too short or lack task keywords")
        else:
            # 理论上不会进入这里（因为 mask 已保证至少一个条件不满足）
            reasons.append("unknown quality filter reason")

    removed["reason"] = reasons

    # ---- 保留通过过滤的样本 ----
    df = df[mask].copy()

    logger.info(f"质量过滤：移除 {before - len(df)} 条，剩余 {len(df)} 条")

    # 统计各原因的数量，便于快速了解过滤分布
    reason_counts = pd.Series(reasons).value_counts().to_dict()
    logger.info(f"质量过滤原因分布: {reason_counts}")

    return df, removed


# ============================== 数据集划分 ====================================
def split_data(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将数据集随机打乱后按比例划分为训练集、验证集、测试集。

    关键特性：
        - 固定随机种子（seed）确保每次运行划分一致
        - 保证不同模型版本（Base/SFT/DPO）在相同测试集上比较
        - 划分在清洗/去重之后进行，避免数据泄漏

    参数:
        df: 清洗后的完整数据集
        train_ratio, valid_ratio, test_ratio: 三个比例，需和为 1.0
        seed: 随机种子

    返回:
        (train_df, valid_df, test_df)
    """
    # 验证比例之和为 1（允许浮点误差）
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"三个比例之和必须为 1.0，当前为 {train_ratio + valid_ratio + test_ratio}")

    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 生成打乱的索引列表
    indices = list(range(len(df)))
    random.shuffle(indices)

    # 计算切分点
    train_end = int(train_ratio * len(df))
    valid_end = train_end + int(valid_ratio * len(df))

    # 根据索引提取各子集，重置索引（drop=True 避免保留原索引）
    train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
    valid_df = df.iloc[indices[train_end:valid_end]].reset_index(drop=True)
    test_df = df.iloc[indices[valid_end:]].reset_index(drop=True)

    logger.info(
        f"数据划分完成：训练 {len(train_df)}，验证 {len(valid_df)}，测试 {len(test_df)}"
    )
    return train_df, valid_df, test_df


# ============================== 输出保存函数 ==================================
def save_json(df: pd.DataFrame, filepath: Path) -> None:
    """
    将 DataFrame 保存为 Alpaca 风格的 JSON 文件（list of dicts）。

    格式要求：
        - 每个字典包含 instruction, input, output 三个字段
        - 使用 UTF-8 编码，保证中文可读性
        - 缩进 2 空格，便于人工查看

    参数:
        df: 要保存的 DataFrame
        filepath: 输出文件路径
    """
    records = df.to_dict(orient="records")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 {len(records)} 条样本至 {filepath}")


def generate_dataset_info(output_dir: str) -> None:
    """
    生成 LLaMA-Factory 所需的 dataset_info.json 配置文件。

    ====================================================================
    这是 A1 与 A2 之间的【数据契约】！！！
    ====================================================================

    该文件告诉 LLaMA-Factory：
        - 每个数据集的文件名（相对于 data 目录）
        - 格式类型（alpaca）
        - 列映射关系：prompt -> instruction, query -> input, response -> output

    A2 模块只需要在 LLaMA-Factory 配置中引用数据集名称
    （如 "code_sft_train"），即可自动加载数据，无需关心文件路径或列名。

    参数:
        output_dir: 输出目录路径
    """
    info = {
        "code_sft_train": {
            "file_name": "code_sft_train.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        },
        "code_sft_valid": {
            "file_name": "code_sft_valid.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        },
        "code_sft_test": {
            "file_name": "code_sft_test.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }

    path = Path(output_dir) / "dataset_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logger.info(f"已生成 dataset_info.json 至 {path}")


def compute_statistics(
    raw_df: pd.DataFrame,
    final_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_removed_df: pd.DataFrame,
    seed: int
) -> None:
    """
    计算并保存详细的数据统计报告（data_statistics.json）。

    报告包含四个维度：
        1. 规模维度：原始样本数、最终可用样本数
        2. 划分维度：训练/验证/测试各子集大小及比例
        3. 过滤维度：各过滤原因的样本计数（用于分析数据清洗影响）
        4. 长度分布维度：训练集的 instruction/input/output 长度分布
           （均值、中位数、25%分位数、75%分位数）

    这些数据将用于 A6 统一评测报告中的数据质量章节。
    如果 A6 发现模型在长代码生成上表现差，可回溯此报告判断训练集长度分布是否合理。

    参数:
        raw_df: 原始 DataFrame
        final_df: 清洗后的最终 DataFrame
        train_df, valid_df, test_df: 划分后的三个子集
        all_removed_df: 所有被过滤样本的合集（含 reason 列）
        seed: 随机种子
    """
    stats = {
        # ---- 规模维度 ----
        "raw_total": len(raw_df),
        "final_usable": len(final_df),

        # ---- 划分维度 ----
        "train_valid_test_counts": [len(train_df), len(valid_df), len(test_df)],
        "split_ratios": {
            "train": len(train_df) / len(final_df) if len(final_df) else 0,
            "valid": len(valid_df) / len(final_df) if len(final_df) else 0,
            "test": len(test_df) / len(final_df) if len(final_df) else 0,
        },

        # ---- 过滤维度：各过滤原因计数 ----
        "filter_breakdown": all_removed_df["reason"].value_counts().to_dict(),

        # ---- 长度分布维度 ----
        "instruction_len": {
            "mean": float(train_df["instruction"].str.len().mean()) if not train_df.empty else 0,
            "median": float(train_df["instruction"].str.len().median()) if not train_df.empty else 0,
            "percentiles": {
                "25%": float(train_df["instruction"].str.len().quantile(0.25)) if not train_df.empty else 0,
                "75%": float(train_df["instruction"].str.len().quantile(0.75)) if not train_df.empty else 0,
            }
        },
        "input_len": {
            "mean": float(train_df["input"].str.len().mean()) if not train_df.empty else 0,
            "median": float(train_df["input"].str.len().median()) if not train_df.empty else 0,
        },
        "output_len": {
            "mean": float(train_df["output"].str.len().mean()) if not train_df.empty else 0,
            "median": float(train_df["output"].str.len().median()) if not train_df.empty else 0,
            "percentiles": {
                "25%": float(train_df["output"].str.len().quantile(0.25)) if not train_df.empty else 0,
                "75%": float(train_df["output"].str.len().quantile(0.75)) if not train_df.empty else 0,
            }
        },

        # ---- 元信息 ----
        "seed": seed,
    }

    path = Path(args.output_dir) / "data_statistics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"统计报告已保存至 {path}")


def save_bad_cases(removed_dfs: List[pd.DataFrame], output_dir: str) -> None:
    """
    将所有被过滤样本合并保存为 bad_cases.json。

    每条记录包含：
        - instruction: 指令内容
        - input: 输入（可能为空）
        - output: 输出代码
        - reason: 过滤原因（具体分类）

    设计目的：
        - 让数据清洗过程完全透明，不再是"黑箱操作"
        - 如果 A2 训练效果不佳，可回溯分析是否因阈值设置过严
        - 为后续参数敏感性分析和策略迭代提供事实依据

    参数:
        removed_dfs: 各过滤阶段被移除样本的 DataFrame 列表
        output_dir: 输出目录
    """
    if not removed_dfs:
        combined = pd.DataFrame(columns=["instruction", "input", "output", "reason"])
    else:
        combined = pd.concat(removed_dfs, ignore_index=True)

    # 只保留必要列
    cols = ["instruction", "input", "output", "reason"]
    combined = combined[cols]

    path = Path(output_dir) / "bad_cases.json"
    combined.to_json(path, orient="records", force_ascii=False, indent=2)

    logger.info(f"总坏例清单已保存至 {path}，共 {len(combined)} 条")


def save_quality_bad_cases(df: pd.DataFrame, output_dir: str) -> None:
    """
    单独保存质量过滤的坏例到 quality_bad_cases.json。

    与 bad_cases.json 的区别：
        - bad_cases.json: 包含所有过滤（空值+去重+长度+质量）
        - quality_bad_cases.json: 仅包含质量过滤的样本

    设计目的：
        - 便于验收时重点展示高质量过滤的效果
        - 方便单独分析语法检查和指令完整性过滤的分布

    参数:
        df: 质量过滤被移除的样本 DataFrame
        output_dir: 输出目录
    """
    if df.empty:
        path = Path(output_dir) / "quality_bad_cases.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        logger.info("质量坏例清单为空，已保存空列表")
        return

    cols = ["instruction", "input", "output", "reason"]
    df = df[cols]

    path = Path(output_dir) / "quality_bad_cases.json"
    df.to_json(path, orient="records", force_ascii=False, indent=2)

    logger.info(f"质量坏例清单已保存至 {path}，共 {len(df)} 条")


def save_preview(train_df: pd.DataFrame, output_dir: str, n: int = 5) -> None:
    """
    从训练集中抽取前 n 条样本保存为 sample_preview.json。

    设计目的：
        - 最小化的冒烟测试：验收时可快速确认格式是否正确
        - 检查字段映射是否张冠李戴
        - 验证代码缩进是否保持（JSON 中保留换行符和空格）

    参数:
        train_df: 训练集 DataFrame
        output_dir: 输出目录
        n: 预览样本数量（默认5条）
    """
    preview = train_df.head(n).to_dict(orient="records")

    path = Path(output_dir) / "sample_preview.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)

    logger.info(f"样本预览已保存至 {path}，共 {len(preview)} 条")


# ============================== 主流水线 ======================================
def main() -> None:
    """
    A1 数据处理主流水线，按顺序执行所有步骤。

    完整流程（9 个阶段）：
        1. 加载原始 Parquet 数据
        2. 字段标准化（统一为 instruction/input/output）
        3. 空值过滤（剔除 instruction 或 output 为空的样本）
        4. 去重（基于 MD5 哈希，移除完全重复样本）
        5. 长度过滤（按 output 长度筛选，对齐 Qwen3 cutoff_len）
        6. 进阶质量过滤（可选：AST 语法检查 + 指令完整性校验）
        7. 数据划分（固定种子，90%/5%/5%）
        8. 保存三份 JSON 文件 + dataset_info.json（数据契约）
        9. 审计产物（统计报告、样本预览、坏例清单）

    每一步都会记录日志，便于追踪处理过程。
    """
    global args  # 使 args 在 compute_statistics 中可访问（避免传递过多参数）
    args = parse_args()

    # 确保输出目录存在
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- 阶段1：加载原始数据 ----
    raw_df = load_parquet(args.source_dir)
    if args.debug_limit:
        raw_df = raw_df.head(args.debug_limit)
        logger.info(f"调试模式：仅处理前 {len(raw_df)} 行")

    # ---- 阶段2：字段标准化 ----
    df = normalize_columns(raw_df)

    # 用于收集所有被移除的样本（各过滤阶段）
    all_removed = []
    quality_removed = None  # 专门收集质量过滤的样本

    # ---- 阶段3：空值过滤 ----
    df, removed = filter_empty(df)
    all_removed.append(removed)

    # ---- 阶段4：去重（默认开启） ----
    if args.remove_duplicates:
        df, removed = deduplicate(df)
        all_removed.append(removed)

    # ---- 阶段5：长度过滤（对齐 Qwen3 cutoff_len=2048） ----
    df, removed = filter_by_length(df, args.min_output_len, args.max_output_len)
    all_removed.append(removed)

    # ---- 阶段6：进阶质量过滤（可选，默认关闭） ----
    if args.enable_quality_filter:
        logger.info("启用高质量过滤（语法检查 + 指令完整性校验）...")
        df, quality_removed = quality_filter(df)

        # 单独保存质量过滤的坏例（便于验收展示）
        save_quality_bad_cases(quality_removed, args.output_dir)

        # 同时加入总坏例集合
        all_removed.append(quality_removed)

    # ---- 阶段7：数据划分（固定种子） ----
    train_df, valid_df, test_df = split_data(
        df, args.train_ratio, args.valid_ratio, args.test_ratio, args.seed
    )

    # ---- 阶段8：保存 Alpaca JSON 数据 + dataset_info.json ----
    save_json(train_df, output_path / "code_sft_train.json")
    save_json(valid_df, output_path / "code_sft_valid.json")
    save_json(test_df, output_path / "code_sft_test.json")
    generate_dataset_info(args.output_dir)

    # ---- 阶段9：审计产物 ----
    # 合并所有被移除样本（用于统计和坏例清单）
    if all_removed:
        all_removed_df = pd.concat(all_removed, ignore_index=True)
    else:
        all_removed_df = pd.DataFrame(columns=["instruction", "input", "output", "reason"])

    # 计算并保存统计报告
    compute_statistics(
        raw_df, df, train_df, valid_df, test_df, all_removed_df, args.seed
    )

    # 保存样本预览（冒烟测试）
    save_preview(train_df, args.output_dir)

    # 保存总坏例清单（所有过滤原因）
    save_bad_cases(all_removed, args.output_dir)

    # ---- 完成 ----
    logger.info("=" * 60)
    logger.info("A1 数据预处理流水线全部完成！")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("产物清单:")
    logger.info("  - code_sft_train.json / code_sft_valid.json / code_sft_test.json")
    logger.info("  - dataset_info.json（LLaMA-Factory 数据契约）")
    logger.info("  - data_statistics.json（四维度统计报告）")
    logger.info("  - sample_preview.json（样本预览）")
    logger.info("  - bad_cases.json（总坏例清单）")
    if args.enable_quality_filter:
        logger.info("  - quality_bad_cases.json（质量过滤坏例清单）")
    logger.info("=" * 60)


# ============================== 程序入口 ======================================
if __name__ == "__main__":
    main()