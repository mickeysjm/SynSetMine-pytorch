#!/bin/bash

# create folder to save model results
if [ ! -d ./results ]; then
  mkdir ./results
fi

# create folder to TensorBoard logging
if [ ! -d ./runs ]; then
  mkdir ./runs
fi

# create folder to save model snapshots
if [ ! -d ./runs ]; then
  mkdir ./runs
fi

python main.py \
	-mode train \
	-device-id 0 \
	-dataset NYT \
	-batch-size 32 \
	-neg-sample-size 20 \
	-dropout 0.3 \
	-loss-fn self_margin_rank_bce \
	-modelName np_lrlr_sd_lrlrdl \
	-embedSize 50 \
	-combine-hiddenSize 500 \
	-node-hiddenSize 250 \
	-margin 0.5 \
	-lr 0.0001 \
	-epochs 500 \
	-early-stop 200 \
	-remark nyt-set-training