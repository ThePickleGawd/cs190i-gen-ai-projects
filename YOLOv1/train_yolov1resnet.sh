#!/bin/bash
# ========================================
# Script: train_YOLOv1ResNet.sh
# Description: Train YOLOv1 with ResNet backbone
# Usage: bash train_YOLOv1ResNet.sh
# ========================================

MODEL="YOLOv1ResNet"
BATCH=74
LR=1e-5
EPOCHS=200
LAMBDA_CLS=2.0
EVAL_N=10
SAVE_DIR="."

mkdir -p logs

echo "Training $MODEL..."
uv run train.py \
  --model $MODEL \
  --batch-size $BATCH \
  --lr $LR \
  --epochs $EPOCHS \
  --lambda-cls $LAMBDA_CLS \
  --eval-interval $EVAL_N \
  --save-dir $SAVE_DIR 2>&1 | tee logs/train_${MODEL}.log

echo "Done: logs/train_${MODEL}.log"