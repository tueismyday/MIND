#!/bin/bash
# Launch script for MIND system with GPU-optimized configuration
#
# This script verifies GPU setup and provides recommendations for launching
# the vLLM server with optimal memory allocation for embedding/reranker models.
#
# Usage:
#   ./scripts/launch_with_gpu.sh [verify|vllm]
#
# Commands:
#   verify - Run GPU setup verification only
#   vllm   - Launch vLLM server with recommended settings
#   (no args) - Run verification and show recommendations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default vLLM configuration
MODEL_NAME="cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
MAX_MODEL_LEN=14000
MAX_NUM_SEQS=1
CPU_OFFLOAD_GB=32

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect GPU memory
get_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        # Get GPU memory in GB
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'
    else
        echo "0"
    fi
}

# Function to calculate recommended GPU utilization
calculate_gpu_utilization() {
    local total_mem=$1
    local models_mem=3  # GB needed for embedding + reranker

    if [ "$total_mem" -le 8 ]; then
        echo "0.50"  # Leave 50% for small GPUs
    elif [ "$total_mem" -le 16 ]; then
        echo "0.65"  # Leave 35% for medium GPUs
    elif [ "$total_mem" -le 24 ]; then
        echo "0.70"  # Leave 30% for 24GB GPUs
    else
        echo "0.75"  # Leave 25% for larger GPUs
    fi
}

# Function to run verification
run_verification() {
    print_info "Running GPU setup verification..."
    echo

    cd "$PROJECT_ROOT"
    python3 scripts/verify_gpu_setup.py

    return $?
}

# Function to launch vLLM with recommended settings
launch_vllm() {
    print_info "Preparing to launch vLLM server..."
    echo

    # Get GPU memory
    GPU_MEM=$(get_gpu_memory)

    if [ "$GPU_MEM" -eq 0 ]; then
        print_error "No NVIDIA GPU detected!"
        print_info "Please ensure nvidia-smi is available and working"
        exit 1
    fi

    print_info "Detected GPU memory: ${GPU_MEM}GB"

    # Calculate recommended utilization
    GPU_UTIL=$(calculate_gpu_utilization $GPU_MEM)
    FREE_MEM=$(echo "$GPU_MEM * (1 - $GPU_UTIL)" | bc -l | xargs printf "%.1f")

    print_info "Recommended GPU utilization: ${GPU_UTIL}"
    print_info "This leaves ~${FREE_MEM}GB for embedding/reranker models"
    echo

    # Show the command
    print_info "Launching vLLM with the following configuration:"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "vllm serve $MODEL_NAME \\"
    echo "    --max-model-len $MAX_MODEL_LEN \\"
    echo "    --gpu-memory-utilization $GPU_UTIL \\"
    echo "    --max-num-seqs $MAX_NUM_SEQS \\"
    echo "    --cpu-offload-gb $CPU_OFFLOAD_GB"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo

    read -p "Press Enter to launch, or Ctrl+C to cancel..."

    # Launch vLLM
    vllm serve "$MODEL_NAME" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --cpu-offload-gb "$CPU_OFFLOAD_GB"
}

# Function to show recommendations
show_recommendations() {
    echo
    print_info "Getting GPU configuration recommendations..."
    echo

    GPU_MEM=$(get_gpu_memory)

    if [ "$GPU_MEM" -eq 0 ]; then
        print_warning "No NVIDIA GPU detected. Models will run on CPU."
        echo
        print_info "To use GPU:"
        echo "  1. Ensure you have an NVIDIA GPU"
        echo "  2. Install CUDA drivers"
        echo "  3. Install PyTorch with CUDA support"
        return
    fi

    GPU_UTIL=$(calculate_gpu_utilization $GPU_MEM)
    FREE_MEM=$(echo "$GPU_MEM * (1 - $GPU_UTIL)" | bc -l | xargs printf "%.1f")

    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                      GPU CONFIGURATION RECOMMENDATIONS                         ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo
    echo "GPU Memory: ${GPU_MEM}GB"
    echo "Recommended vLLM utilization: ${GPU_UTIL} (~${FREE_MEM}GB free for embedding/reranker)"
    echo
    echo "RECOMMENDED LAUNCH COMMAND:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "vllm serve $MODEL_NAME \\"
    echo "    --max-model-len $MAX_MODEL_LEN \\"
    echo "    --gpu-memory-utilization $GPU_UTIL \\"
    echo "    --max-num-seqs $MAX_NUM_SEQS \\"
    echo "    --cpu-offload-gb $CPU_OFFLOAD_GB"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "QUICK LAUNCH OPTIONS:"
    echo "  1. Verify GPU setup:        ./scripts/launch_with_gpu.sh verify"
    echo "  2. Launch vLLM optimally:   ./scripts/launch_with_gpu.sh vllm"
    echo
}

# Main script
main() {
    case "${1:-help}" in
        verify)
            run_verification
            exit $?
            ;;
        vllm)
            launch_vllm
            ;;
        help|--help|-h|"")
            show_recommendations
            echo
            print_info "For detailed GPU verification, run:"
            echo "  ./scripts/launch_with_gpu.sh verify"
            ;;
        *)
            print_error "Unknown command: $1"
            echo
            echo "Usage: $0 [verify|vllm]"
            echo "  verify - Run GPU setup verification"
            echo "  vllm   - Launch vLLM with optimized settings"
            exit 1
            ;;
    esac
}

main "$@"
