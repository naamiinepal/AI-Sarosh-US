#!/bin/bash

# Dataset preparation paths
IMAGE_ROOT='/mnt/Enterprise2/AI-Sarosh/datasets/samples'
OUT_ROOT='/mnt/Enterprise2/AI-Sarosh/datasets/csvs'
LABELS_ROOT='/mnt/Enterprise2/AI-Sarosh/datasets/csvs'

# Run dataset preparation script
python3 /mnt/Enterprise2/AI-Sarosh/src/src/data/components/utils/prepare_dataset.py \
   --image-root "$IMAGE_ROOT" \
   --out-root "$OUT_ROOT" \
   --labels-root "$LABELS_ROOT" \
   --val 10 \
   --test 10