#!/bin/bash
name="coarse-3d-model-video"

dir_name=$1
wonder3d_ckpt=$2
echo elevation offset $5;
CUDA_VISIBLE_DEVICES=$3 python launch.py --config $4 --test \
    name=$name \
    tag="$dir_name" \
    data.n_test_views=180 \
    data.render_img=false \
    data.test_fixed_fovy_dist=true \
    data.input_image_elevation_offset=$5 \
    system.prompt_processor.prompt="$dir_name" \
    system.prompt_processor.use_cache=true \
    system.material_convert_from="same_as_geometry"\
    system.prev_cfg_dir=Wonder3D/instant-nsr-pl/configs/parsed.yaml\
    system.geometry_convert_from=$wonder3d_ckpt \