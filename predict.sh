#!/bin/bash

# for NYT dataset
python main.py \
	-mode cluster_predict \
	-snapshot ./snapshots/Nov10_01-46-52_nyt-set-training/best_steps_290.pt \
	-modelName np_lrlr_sd_lrlrdl \
	-device-id 0 \
	-dataset NYT \
	-embedSize 50 \
	-combine-hiddenSize 500 \
	-node-hiddenSize 250 

# for Wiki dataset
# python main.py \
# 	-mode cluster_predict \
# 	-snapshot ./snapshots/Nov10_01-55-01_wiki-set-training/best_steps_415.pt \
# 	-modelName np_lrlr_sd_lrlrdl \
# 	-device-id 0 \
# 	-dataset Wiki \
# 	-embedSize 50 \
# 	-combine-hiddenSize 500 \
# 	-node-hiddenSize 250 

# for PubMed dataset
# python main.py \
# 	-mode cluster_predict \
# 	-snapshot ./snapshots/Nov10_14-31-20_pubmed-set-training/best_steps_145.pt \
# 	-modelName np_lrlr_sd_lrlrdl \
# 	-device-id 0 \
# 	-dataset PubMed \
# 	-embedSize 50 \
# 	-combine-hiddenSize 500 \
# 	-node-hiddenSize 250 