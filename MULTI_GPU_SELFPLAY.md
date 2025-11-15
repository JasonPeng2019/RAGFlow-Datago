# Multi-GPU Selfplay Commands

## Launch all 7 GPUs (copy-paste this entire block)

```bash
for gpu in {0..6}; do
    tmux new -d -s "selfplay-gpu${gpu}" bash -c "
        export CUDA_VISIBLE_DEVICES=${gpu}
        cd /scratch2/f004h1v/alphago_project/build
        ./katago selfplay -config ../selfplay_gpu7.cfg -models-dir ./models -output-dir ./selfplay_output
    "
    echo "Started selfplay on GPU ${gpu} in tmux session selfplay-gpu${gpu}"
done
```

## Launch with limited games (e.g., 200 games per GPU)

```bash
for gpu in 1 2 3 5 7; do
    tmux new -d -s "selfplay-gpu${gpu}" bash -c "
        export CUDA_VISIBLE_DEVICES=${gpu}
        cd /scratch2/f004h1v/alphago_project/build
        ./katago selfplay -config ../selfplay_gpu7.cfg -models-dir ./models -output-dir ./selfplay_output -max-games-total 200
    "
    echo "Started selfplay on GPU ${gpu} (max 200 games)"
done
```

## Monitor sessions

```bash
# List all running tmux sessions
tmux ls

# Attach to a specific GPU session (replace 0 with GPU number)
tmux attach -t selfplay-gpu0

# Detach from session (while inside tmux)
# Press: Ctrl+b then d
```

## Stop all sessions

```bash
# Kill all selfplay sessions
for gpu in 1 2 3 5 7; do
    tmux kill-session -t selfplay-gpu${gpu}
    echo "Killed selfplay-gpu${gpu}"
done
```

## Check GPU usage

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi
```

## Check output files

```bash
# Check RAG data files generated
ls -lh /scratch2/f004h1v/alphago_project/build/rag_data/

# Count how many games have been generated
ls /scratch2/f004h1v/alphago_project/build/rag_data/ | wc -l

# Check selfplay training data
ls -lh /scratch2/f004h1v/alphago_project/build/selfplay_output/data/
```

## View logs from specific GPU

```bash
# Attach to GPU 0 session and see the output
tmux attach -t selfplay-gpu0

# Or peek at recent output without attaching
tmux capture-pane -t selfplay-gpu0 -p | tail -50
```
