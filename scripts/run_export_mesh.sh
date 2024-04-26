# refine texture with sds loss (stable diffusion)
obj_name=$1

ckpt_name=$2
config_name=$3

yaml=$4


name="$5"

CUDA_VISIBLE_DEVICES="$6" python launch.py --config $config_name --export \
    name="$name" \
    tag="$obj_name" \
    resume=$ckpt_name \
    system.prev_cfg_dir=$config_name \
    system.geometry_convert_from=$ckpt_name \
    system.material_convert_from="same_as_geometry"\
    system.guidance.dreambooth_ckpt_path=\"\"  \
    system.prompt_processor.dreambooth_ckpt_path=\"\" \
    system.ref_guidance.dreambooth_ckpt_path=\"\"  \
    system.ref_prompt_processor.dreambooth_ckpt_path=\"\" \
    system.ref_prompt_norm_processor.dreambooth_ckpt_path=\"\" \
    system.exporter_type=mesh-exporter \
