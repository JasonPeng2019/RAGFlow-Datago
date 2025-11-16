#!/bin/bash
# Parallel KataGo Self-Play on Multiple GPUs
# Uses existing GPU-specific selfplay configs
#
# Usage: ./run_parallel_selfplay.sh [gpu1 gpu2 gpu3 ...]
#        ./run_parallel_selfplay.sh 0 1 2 3 4 5 7  # Use specific GPUs
#        ./run_parallel_selfplay.sh                # Use default GPU set

set -euo pipefail

echo "=========================================="
echo "KataGo Parallel Self-Play Launcher"
echo "=========================================="
echo ""

# Get script directory (project root) using relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

: "${DATAGO_INLINE_DEEP_ENABLED:=1}"
: "${DATAGO_INLINE_DEEP_STRIDE:=5}"
: "${DATAGO_INLINE_DEEP_VISITS:=4800}"
: "${DATAGO_INLINE_DEEP_MAX_PER_MOVE:=1}"
: "${DATAGO_INLINE_DEEP_MAX_PER_GAME:=256}"
: "${DATAGO_INLINE_DEEP_LOG:=0}"

export DATAGO_INLINE_DEEP_ENABLED DATAGO_INLINE_DEEP_STRIDE DATAGO_INLINE_DEEP_VISITS \
       DATAGO_INLINE_DEEP_MAX_PER_MOVE DATAGO_INLINE_DEEP_MAX_PER_GAME DATAGO_INLINE_DEEP_LOG

# Configuration (relative paths)
KATAGO_BIN="$PROJECT_ROOT/katago_repo/KataGo/cpp/build-opencl/katago"
MODEL_FILE="$PROJECT_ROOT/katago_repo/run/default_model.bin.gz"
BASE_OUTPUT_DIR="$PROJECT_ROOT/selfplay_output"

# GPU configs to use
declare -a GPU_CONFIGS
declare -a GPU_IDS

# If user provided GPU IDs as arguments, use those
if [ $# -gt 0 ]; then
    for gpu_id in "$@"; do
        cfg_file="$PROJECT_ROOT/selfplay_gpu${gpu_id}.cfg"
        if [ -f "$cfg_file" ]; then
            GPU_CONFIGS+=("$cfg_file")
            GPU_IDS+=("$gpu_id")
        else
            echo "Warning: Config file not found: $cfg_file"
        fi
    done
else
    # Otherwise, use pre-selected GPUs with sufficient free memory
    # Selected based on nvidia-smi on Nov 16, 2025
    for gpu_id in 0 1 2 3 4 5 7; do
        cfg_file="$PROJECT_ROOT/selfplay_gpu${gpu_id}.cfg"
        if [ -f "$cfg_file" ]; then
            GPU_CONFIGS+=("$cfg_file")
            GPU_IDS+=("$gpu_id")
        else
            echo "Warning: Config file not found for GPU $gpu_id: $cfg_file"
        fi
    done
fi

# Check if we found any configs
if [ ${#GPU_CONFIGS[@]} -eq 0 ]; then
    echo "Error: No selfplay_gpu*.cfg files found in $PROJECT_ROOT"
    echo "Please create GPU-specific config files (e.g., selfplay_gpu0.cfg, selfplay_gpu1.cfg, etc.)"
    exit 1
fi

# Verify KataGo binary exists
if [ ! -f "$KATAGO_BIN" ]; then
    echo "Error: KataGo binary not found at $KATAGO_BIN"
    echo "Please compile KataGo first or adjust KATAGO_BIN path"
    exit 1
fi

# Verify model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at $MODEL_FILE"
    exit 1
fi

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Pre-create all output directories for the GPUs we'll use
echo "Creating output directories for ${#GPU_CONFIGS[@]} GPU sessions:"
for i in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[$i]}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/gpu${GPU_ID}"
    mkdir -p "$OUTPUT_DIR"
    echo "  ✓ Created: $OUTPUT_DIR"
done
echo ""

echo "Configuration:"
echo "  Project root: $PROJECT_ROOT"
echo "  KataGo binary: $KATAGO_BIN"
echo "  Model: $(basename "$MODEL_FILE")"
echo "  Output directory: $BASE_OUTPUT_DIR"
echo "  Using ${#GPU_CONFIGS[@]} GPUs: ${GPU_IDS[*]}"
echo "  Games per GPU: 300"
echo "  Max visits: 300"
echo "  Inline deep search: enabled=${DATAGO_INLINE_DEEP_ENABLED}, stride=${DATAGO_INLINE_DEEP_STRIDE}, visits=${DATAGO_INLINE_DEEP_VISITS}, maxPerMove=${DATAGO_INLINE_DEEP_MAX_PER_MOVE}, maxPerGame=${DATAGO_INLINE_DEEP_MAX_PER_GAME}, log=${DATAGO_INLINE_DEEP_LOG}"
echo ""

echo "Output Structure:"
echo "  Each GPU will output to: ${BASE_OUTPUT_DIR}/gpu{N}/"
echo "  Files generated per GPU:"
echo "    - selfplay.log          (main log file)"
echo "    - game_*.json           (RAG data - JSON files with game positions)"
echo "    - *.npz                 (training data - numpy compressed files)"
echo "    - *.sgf                 (SGF game records)"
echo "    - shuffleddata/         (shuffled training data directory)"
echo ""

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --list-gpus 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  sudo apt-get install tmux"
    exit 1
fi

# Launch KataGo on each GPU in tmux sessions
for i in "${!GPU_CONFIGS[@]}"; do
    GPU_ID="${GPU_IDS[$i]}"
    CONFIG_FILE="${GPU_CONFIGS[$i]}"
    SESSION_NAME="katago-gpu${GPU_ID}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/gpu${GPU_ID}"
    LOG_FILE="${OUTPUT_DIR}/selfplay.log"

    # Create output directories
    mkdir -p "$OUTPUT_DIR"

    # Kill existing session if it exists
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

    # Launch KataGo in tmux session using selfplay mode
    tmux new -d -s "$SESSION_NAME" bash -c "
    cd '$OUTPUT_DIR'

    echo \"========================================\" | tee '$LOG_FILE'
    echo \"KataGo Self-Play on GPU $GPU_ID\" | tee -a '$LOG_FILE'
    echo \"========================================\" | tee -a '$LOG_FILE'
    echo \"Started at: \$(date)\" | tee -a '$LOG_FILE'
    echo \"Config: $(basename '$CONFIG_FILE')\" | tee -a '$LOG_FILE'
    echo \"Output directory: $OUTPUT_DIR\" | tee -a '$LOG_FILE'
    echo \"Model: $(basename '$MODEL_FILE')\" | tee -a '$LOG_FILE'
    echo \"\" | tee -a '$LOG_FILE'

    # Run KataGo selfplay (200 games, 300 max visits)
    '$KATAGO_BIN' selfplay \\
        -config '$CONFIG_FILE' \\
        -models-dir '$PROJECT_ROOT/katago_repo/run' \\
        -output-dir '$OUTPUT_DIR' \\
        -max-games-total 300 \\
        -override-config maxVisits=300 \\
        2>&1 | tee -a '$LOG_FILE'

    echo \"\" | tee -a '$LOG_FILE'
    echo \"Completed at: \$(date)\" | tee -a '$LOG_FILE'
    echo \"KataGo on GPU $GPU_ID completed.\" | tee -a '$LOG_FILE'
    echo \"\" | tee -a '$LOG_FILE'
    echo \"Press Enter to close this tmux session...\"
    read
    "

    echo "✓ Started session '$SESSION_NAME' on GPU $GPU_ID"
    echo "  Config: $(basename "$CONFIG_FILE")"
    echo "  Output: $OUTPUT_DIR"
    echo ""

    # Small delay between launches to avoid GPU initialization conflicts
    sleep 2
done

echo ""
echo "=========================================="
echo "All ${#GPU_CONFIGS[@]} KataGo instances launched!"
echo "=========================================="
echo ""
echo "TMUX SESSION INFO:"
echo "  ✓ ${#GPU_CONFIGS[@]} independent tmux sessions created"
echo "  ✓ Sessions are running in the background"
echo "  ✓ You can close this terminal - games will continue running"
echo "  ✓ Sessions will terminate automatically when games complete"
echo ""
echo "Active tmux sessions:"
for i in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[$i]}"
    echo "  - katago-gpu${GPU_ID}  (GPU ${GPU_ID})"
done
echo ""
echo "OUTPUT LOCATIONS:"
for i in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[$i]}"
    echo "  GPU ${GPU_ID}:"
    echo "    Directory:  ${BASE_OUTPUT_DIR}/gpu${GPU_ID}/"
    echo "    Log file:   ${BASE_OUTPUT_DIR}/gpu${GPU_ID}/selfplay.log"
    echo "    RAG data:   ${BASE_OUTPUT_DIR}/gpu${GPU_ID}/game_*.json"
    echo "    NPZ data:   ${BASE_OUTPUT_DIR}/gpu${GPU_ID}/*.npz"
    echo ""
done
echo "=========================================="
echo "USEFUL COMMANDS:"
echo "=========================================="
echo ""
echo "Monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "List all running KataGo sessions:"
echo "  tmux ls | grep katago"
echo ""
echo "Attach to a specific session (e.g., GPU ${GPU_IDS[0]}):"
echo "  tmux attach -t katago-gpu${GPU_IDS[0]}"
echo "  (Press Ctrl+B then D to detach without stopping)"
echo ""
echo "View real-time log for GPU ${GPU_IDS[0]}:"
echo "  tail -f ${BASE_OUTPUT_DIR}/gpu${GPU_IDS[0]}/selfplay.log"
echo ""
echo "View progress across ALL GPUs:"
echo "  for gpu in ${GPU_IDS[@]}; do echo \"=== GPU \$gpu ===\"  && tail -10 ${BASE_OUTPUT_DIR}/gpu\$gpu/selfplay.log 2>/dev/null || echo \"No log yet\"; echo \"\"; done"
echo ""
echo "Count games completed per GPU:"
echo "  for gpu in ${GPU_IDS[@]}; do echo -n \"GPU \$gpu: \"; ls ${BASE_OUTPUT_DIR}/gpu\$gpu/*.sgf 2>/dev/null | wc -l; done"
echo ""
echo "Stop a specific GPU session (e.g., GPU ${GPU_IDS[0]}):"
echo "  tmux kill-session -t katago-gpu${GPU_IDS[0]}"
echo ""
echo "Stop ALL KataGo sessions:"
echo "  for gpu in ${GPU_IDS[@]}; do tmux kill-session -t katago-gpu\$gpu 2>/dev/null || true; done"
echo ""
echo "Collect all RAG data:"
echo "  find ${BASE_OUTPUT_DIR} -name 'game_*.json' -type f"
echo ""
echo "Collect all training data:"
echo "  find ${BASE_OUTPUT_DIR} -name '*.npz' -type f"
echo ""
echo "=========================================="
echo "Sessions are now running independently."
echo "You can safely close this terminal."
echo "=========================================="
echo ""
