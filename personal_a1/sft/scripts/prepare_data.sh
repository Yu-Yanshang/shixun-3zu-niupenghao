#!/bin/bash
# A1: 推理指令数据构造一键运行脚本
# 支持通过环境变量覆盖参数，兼容 README 中的调试接口
set -euo pipefail  # 严格模式：出错即退出，未定义变量报错，管道错误传递

cd /root/project
source scripts/common_env.sh

PYTHON_BIN=/opt/conda/envs/shixun/bin/python

# ---- 环境变量配置（可覆盖默认值） ----
SOURCE_DIR="${SOURCE_DIR:-python_code_instructions_18k_alpaca}"
OUTPUT_DIR="${OUTPUT_DIR:-sft/data}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
VALID_RATIO="${VALID_RATIO:-0.05}"
TEST_RATIO="${TEST_RATIO:-0.05}"
SEED="${SEED:-42}"
MIN_OUTPUT_LEN="${MIN_OUTPUT_LEN:-10}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-2048}"

# 处理调试限制参数（兼容 README 中的 LIMIT=1000）
DEBUG_LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    DEBUG_LIMIT_ARG="--debug_limit $LIMIT"
fi

# 可选的质量过滤开关（通过环境变量 ENABLE_QUALITY_FILTER=1 启用）
QUALITY_ARG=""
if [ "${ENABLE_QUALITY_FILTER:-1}" = "0" ]; then
    QUALITY_ARG="--enable_quality_filter"
fi

# ---- 运行 Python 脚本 ----
$PYTHON_BIN sft/scripts/prepare_code_sft_data.py \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio "$TRAIN_RATIO" \
    --valid_ratio "$VALID_RATIO" \
    --test_ratio "$TEST_RATIO" \
    --seed "$SEED" \
    --min_output_len "$MIN_OUTPUT_LEN" \
    --max_output_len "$MAX_OUTPUT_LEN" \
    $QUALITY_ARG \
    $DEBUG_LIMIT_ARG

echo "A1 finished. Check output in $OUTPUT_DIR"