#!/bin/bash
# ========================================
# Script: train_YOLOv1.sh
# Description: Train the vanilla YOLOv1 model
# Usage: bash train_YOLOv1.sh
# ========================================

MODEL="YOLOv1"
BATCH=32
LR=1e-3
EPOCHS=200
LAMBDA_CLS=1.0
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


