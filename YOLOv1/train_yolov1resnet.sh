#!/bin/bash
# ========================================
# Script: train_YOLOv1ResNet.sh
# Description: Train YOLOv1 with ResNet backbone
# Usage: bash train_YOLOv1ResNet.sh
# ========================================

MODEL="YOLOv1ResNet"
BATCH=64
LR=1e-4
EPOCHS=200
LAMBDA_CLS=2.0
SAVE_DIR="."

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
  2>&1 | tee logs/train_${MODEL}.log

echo "Done: logs/train_${MODEL}.log"