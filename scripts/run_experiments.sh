#!/bin/bash
python lightgcn_sparse/train.py --dataset min5_win10
python simgcl/train_new.py --dataset min5_win10
python contrastive_learning_new/train.py --dataset min5_win10
