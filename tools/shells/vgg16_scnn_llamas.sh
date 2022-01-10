#!/bin/bash
# Trained weights: vgg16_scnn_llamas_20210625.pt
# Training
python main_landet.py --train --config=configs/lane_detection/scnn/vgg16_llamas.py --mixed-precision
# Predicting lane points for testing
python main_landet.py --test --config=configs/lane_detection/scnn/vgg16_llamas.py --mixed-precision
# Testing with official scripts
./autotest_llamas.sh vgg16_scnn_llamas test checkpoints
