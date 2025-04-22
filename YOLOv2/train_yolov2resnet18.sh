#!/bin/bash
# ========================================
# Script: train_YOLOv2ResNet.sh
# Description: Train YOLOv2 with ResNet backbone
# Usage: bash train_YOLOv2ResNet.sh
# ========================================

MODEL="YOLOv2ResNet18"
BATCH=64
LR=1e-3
EPOCHS=120
LAMBDA_CLS=1

mkdir -p logs

echo "Training $MODEL..."
echo "LR=$LR EPOCHS=$EPOCHS BATCH=$BATCH"
echo "CLASSIFICATION BOOST=$LAMBDA_CLS"
uv run train.py \
  --model $MODEL \
  --batch-size $BATCH \
  --lr $LR \
  --epochs $EPOCHS \
  --lambda-cls $LAMBDA_CLS \
  --save-last-checkpoint \
  2>&1 | tee logs/train_${MODEL}.log

echo "Done: logs/train_${MODEL}.log"