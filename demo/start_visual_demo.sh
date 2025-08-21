#!/bin/bash

# ASearcher可视化演示服务启动脚本
# 
# 使用方法:
# ./start_visual_demo.sh [vLLM服务器URL] [模型名称] [端口] [主机地址] [启用reload]
#
# 示例:
# ./start_visual_demo.sh http://localhost:8000 Qwen2.5-7B-Instruct 8080 0.0.0.0 false
# ./start_visual_demo.sh http://localhost:8000 Qwen2.5-7B-Instruct 8080 0.0.0.0 true  # 启用自动重载

# 默认配置
DEFAULT_VLLM_URL="http://localhost:50000"
DEFAULT_MODEL_NAME="ASearcher-Web-QwQ"
DEFAULT_PORT="8080"
DEFAULT_HOST="0.0.0.0"

# 从参数获取配置
VLLM_URL=${1:-$DEFAULT_VLLM_URL}
MODEL_NAME=${2:-$DEFAULT_MODEL_NAME}
PORT=${3:-$DEFAULT_PORT}
HOST=${4:-$DEFAULT_HOST}
ENABLE_RELOAD=${5:-"false"}

echo "🚀 启动ASearcher可视化演示服务"
echo "=================================================="
echo "服务地址: http://$HOST:$PORT"
echo "vLLM服务器: $VLLM_URL"
echo "模型名称: $MODEL_NAME"
echo "API密钥: EMPTY (vLLM默认)"
echo "自动重载: $ENABLE_RELOAD"
echo "=================================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查依赖包
echo "📦 检查依赖包..."

# 检查必需的包
REQUIRED_PACKAGES=("fastapi" "uvicorn" "openai" "pydantic")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "❌ 缺少依赖包: ${MISSING_PACKAGES[*]}"
    echo "请运行: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

echo "✅ 依赖包检查完成"

# 检查vLLM服务器连接
echo "🔗 检查vLLM服务器连接: $VLLM_URL"
if curl -s --connect-timeout 5 "$VLLM_URL/health" >/dev/null 2>&1; then
    echo "✅ vLLM服务器连接正常"
else
    echo "⚠️  警告: 无法连接到vLLM服务器 ($VLLM_URL)"
    echo "请确保vLLM服务器正在运行，或使用正确的URL"
    echo "继续启动服务..."
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}

# 启动服务
echo "🚀 启动服务..."
echo ""

cd "$(dirname "$0")"

# 构建命令参数
CMD_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --llm-url "$VLLM_URL"
    --model-name "$MODEL_NAME"
    --api-key "EMPTY"
)

# 如果启用reload，添加--reload选项
if [ "$ENABLE_RELOAD" = "true" ]; then
    CMD_ARGS+=(--reload)
fi

echo "启动命令: python3 asearcher_visual_demo.py ${CMD_ARGS[*]}"
echo ""

python3 asearcher_visual_demo.py "${CMD_ARGS[@]}"

echo "👋 服务已停止"
