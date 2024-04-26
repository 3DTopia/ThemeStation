#!/bin/bash
scene="$1"

image_path=$3

accelerate launch --config_file 1gpu.yaml --gpu_ids "$2" test_mvdiffusion_seq.py --config configs/mvdiffusion-joint-ortho-6views.yaml \
    validation_dataset.filepath=$image_path\
    seed=$4