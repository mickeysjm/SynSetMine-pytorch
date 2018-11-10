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
if [ ! -d ./snapshots ]; then
  mkdir ./snapshots
fi

# for NYT dataset
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
	-lr 0.0001 \
	-epochs 500 \
	-early-stop 200 \
	-remark nyt-set-training

# for Wiki dataset
# python main.py \
# 	-mode train \
# 	-device-id 0 \
# 	-dataset Wiki \
# 	-node-hiddenSize 250 \
# 	-embedSize 50 \
# 	-combine-hiddenSize 500 \
# 	-modelName np_lrlr_sd_lrlrdl \
# 	-loss-fn self_margin_rank_bce \
# 	-lr 0.0001 \
# 	-early-stop 200 \
# 	-epochs 800 \
# 	-batch-size 64 \
# 	-dropout 0.4 \
# 	-neg-sample-size 50 \
# 	-remark wiki-set-training

# for PubMed dataset
# python main.py \
# 	-mode train \
# 	-device-id 0 \
# 	-dataset PubMed \
# 	-batch-size 32 \
# 	-neg-sample-size 50 \
# 	-dropout 0.3 \
# 	-loss-fn self_margin_rank_bce \
# 	-modelName np_lrlr_sd_lrlrdl \
# 	-combine-hiddenSize 500 \
# 	-node-hiddenSize 250 \
# 	-lr 0.0003 \
# 	-epochs 500 \
# 	-early-stop 100 \
# 	-remark pubmed-set-training

