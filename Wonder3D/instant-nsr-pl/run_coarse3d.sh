#!/bin/bash
root_dir="../outputs/cropsize-192-cfg3.0/"
scene="$1"
port=$3
python launch.py --config configs/neuralangelo-ortho-wmask.yaml --master_port $port --gpu "$2" --train dataset.root_dir=$root_dir dataset.scene=$scene
