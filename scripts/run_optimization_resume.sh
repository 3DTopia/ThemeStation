# refine texture with sds loss (stable diffusion)
obj_name=$1
obj_name_no_id=$2

resume_base_dir=$3
ckpt_name=$resume_base_dir/ckpts/epoch=0-step=5000.ckpt
config_name=$resume_base_dir/configs/parsed.yaml

prompt="$4"
ref_prompt="$5"
ref_prompt_norm="$ref_prompt, normal map"

yaml=$8

base_ckpt=300
base_dreambooth_ckpt_path="./dreambooth/ckpt/base_model/"$obj_name"_text/checkpoint-"$base_ckpt""

ckpt_ref=150
name="$7"
ref_dreambooth_ckpt_path="./dreambooth/ckpt/reference_model_elevation_rand/"$obj_name"/checkpoint-"$ckpt_ref""

CUDA_VISIBLE_DEVICES="$6" python launch.py --config configs/$yaml --train \
    name="$name" \
    tag="$obj_name" \
    resume=$ckpt_name \
    data.input_image_elevation_offset=${9} \
    system.use_reference_model_dreambooth=true \
    system.use_prompt_norm=true \
    system.optimizer.update_lr_at_certain_step=true \
    trainer.max_steps=10000 \
    system.prev_cfg_dir=$config_name \
    system.geometry_convert_from=$ckpt_name \
    system.material_convert_from="same_as_geometry"\
    system.prompt_processor.prompt="$prompt" \
    system.ref_prompt_processor.prompt="$ref_prompt" \
    system.ref_prompt_norm_processor.prompt="$ref_prompt_norm" \
    system.guidance.use_camera_embedding=false \
    system.ref_guidance.use_camera_embedding=false \
    system.guidance.dreambooth_ckpt_path="$base_dreambooth_ckpt_path"  \
    system.prompt_processor.dreambooth_ckpt_path="$base_dreambooth_ckpt_path" \
    system.ref_guidance.dreambooth_ckpt_path="$ref_dreambooth_ckpt_path"  \
    system.ref_prompt_processor.dreambooth_ckpt_path="$ref_dreambooth_ckpt_path" \
    system.ref_prompt_norm_processor.dreambooth_ckpt_path="$ref_dreambooth_ckpt_path" \