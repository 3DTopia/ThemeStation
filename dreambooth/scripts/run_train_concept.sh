# OBJECTIVE: learn concept prior

obj_name=$1
PROMPT=$2
PORT=$5
init_train_steps=150
base_train_steps=300

INSTANCE_DIR="./data/img2img_20views/"$obj_name"_text/checkpoint-"$init_train_steps"_strength50"
OUTPUT_DIR="./ckpt/base_model/"$obj_name"_text"
VAR_IMG_DIR="./data/data_rgb/"$obj_name""

MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
accelerate launch --config_file ../configs/acc_config.yaml --main_process_port $PORT --num_processes 1 --gpu_ids $4 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --variation_data_dir=$VAR_IMG_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$PROMPT" \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate="$6" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="$base_train_steps" \
  --train_text_encoder \
  --checkpointing_steps="$init_train_steps" \
  --resume_from_checkpoint="./ckpt/base_model/"$obj_name"_text/checkpoint-"$init_train_steps"" \
  --seed=$7 \

  # inference_save_dir="./output/concept_prior_ckpt"$base_train_steps"/"$obj_name""
  # CUDA_VISIBLE_DEVICES=$4 python text2img.py \
  #         --root_dir=$OUTPUT_DIR \
  #         --output_dir=$inference_save_dir \
  #         --prompt="$PROMPT" \
  #         --ckpt_i="$base_train_steps"
  