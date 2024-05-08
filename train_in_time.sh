#!/usr/bin/bash

echo "start training on specific time"

tmux kill-session -t train_on_time
tmux new-session -d -s train_on_time

tmux send-keys "cd /home/chenjie04/workstation/Multi-Task-Learning-PyTorch/" C-m
tmux send-keys "pwd" C-m

tmux send-keys "source /home/chenjie04/anaconda3/bin/activate mmdet-3.0" C-m

tmux send-keys "bash /home/chenjie04/workstation/Multi-Task-Learning-PyTorch/train.sh >> /home/chenjie04/workstation/Multi-Task-Learning-PyTorch/log/nddr_cnn_log.txt" C-m