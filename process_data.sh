#!/bin/bash

python preprocess_data.py \
  --dataset ml-100k \
  --raw-dir data/raw/ml-100k \
  --output-dir data/processed/ml-100k-custom \
  --min-rating none \
  --num-negatives 99 \
  --seed 42