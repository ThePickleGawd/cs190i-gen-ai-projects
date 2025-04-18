#!/bin/bash
# ========================================
# Script: train_YOLOv1ResNet.sh
# Description: Train YOLOv1 with ResNet backbone
# Usage: bash train_YOLOv1ResNet.sh
# ========================================

MODEL="YOLOv1ResNet"
BATCH=64
LR=5e-2
EPOCHS=300
LAMBDA_CLS=50.0

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