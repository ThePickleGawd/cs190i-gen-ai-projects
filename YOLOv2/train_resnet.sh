#!/bin/bash
# ========================================
# Script: train_resnet.sh
# Description: Train the ResNet18 backbone model
# Usage: bash train_resnet.sh
# ========================================

MODEL="ResNet18"
BATCH=64
LR=1e-3
EPOCHS=120
LAMBDA_CLS=1.5

mkdir -p logs

echo "Training $MODEL..."
uv run train_resnet.py \
  --batch-size $BATCH \
  --lr $LR \
  --epochs $EPOCHS \
  --save-last-checkpoint \
  2>&1 | tee logs/train_${MODEL}.log

echo "Done: logs/train_${MODEL}.log"


