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
LAMBDA_CLS=2.0  # stronger classification weight
EVAL_N=10
SAVE_DIR="."

mkdir -p logs

echo "Training $MODEL..."
python train.py \
  --model $MODEL \
  --batch-size $BATCH \
  --lr $LR \
  --epochs $EPOCHS \
  --lambda-cls $LAMBDA_CLS \
  --eval-interval $EVAL_N \
  --save-dir $SAVE_DIR \
  > logs/train_${MODEL}.log 2>&1

echo "Done: logs/train_${MODEL}.log"