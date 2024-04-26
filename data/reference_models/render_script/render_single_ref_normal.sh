#!/bin/bash
OLDIFS=$IFS
IFS=$'\n';

elevation=rand
num_img=30
normal='_normal'
output_dir="./renderings/elevation_$elevation"

obj_file_name=$1
blender_path=$2

for file in ./models/*
do
    if [ "${file##*./models/}"x = "$obj_file_name"x ];
        then

        # normal (optional)
        $blender_path -b -P render_script/blender_script.py -- \
            --object_path $file \
            --output_dir $output_dir \
            --only_northern_hemisphere \
            --num_renders $num_img \
            --engine CYCLES \
            --camera_dist 1.5 \
            --elevation $elevation \
            --normal "$normal"
    fi
done
