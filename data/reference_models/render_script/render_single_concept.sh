#!/bin/bash
OLDIFS=$IFS
IFS=$'\n';

elevation=$1
num_img_each_obj=30
output_dir="./renderings/elevation_$elevation"

obj_file_name=$2
blender_path=$3

for file in ./models/*
do
    if [ "${file##*./models/}"x = "$obj_file_name"x ];
        then
            # rgb
            $blender_path -b -P render_script/blender_script.py -- \
                --object_path $file \
                --output_dir $output_dir \
                --only_northern_hemisphere \
                --num_renders $num_img_each_obj \
                --engine CYCLES \
                --ortho_camera \
                --ortho_scale 1.3 \
                --elevation $elevation \
                --normal ""
    fi
done
