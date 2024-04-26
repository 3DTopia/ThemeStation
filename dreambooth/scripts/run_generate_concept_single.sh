# OBJECTIVE: obtain concept images as Wonder3D inputs to obtain initial 3d model

ref_type=$1
elevation=$6
PORT=$5
echo $elevation;
obj_name=$2
PROMPT=$3
EDIT_PROMPT=$7
editing_prefix=$8
echo $EDIT_PROMPT;

INSTANCE_DIR="../data/reference_models/renderings/elevation_"$elevation"/"$obj_name""
OUTPUT_DIR="./ckpt/reference_model_elevation_"$elevation"/"$obj_name""

MODEL_NAME="stabilityai/stable-diffusion-2-1-base"


accelerate launch --config_file ../configs/acc_config.yaml --main_process_port $PORT --num_processes 1 --gpu_ids $4 \
  train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$PROMPT" \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --train_text_encoder \
  --checkpointing_steps=25 \
  --strat_checkpointing_step=75 \
  # --class_embed_type="projection" \
  # --class_labels_conditioning="camera_emb" \

  inference_save_dir="./output/concept_image_"$ref_type"_elevation_"$elevation"/"$editing_prefix""$obj_name""
  CUDA_VISIBLE_DEVICES=$4 python text2img.py \
          --root_dir=$OUTPUT_DIR \
          --output_dir=$inference_save_dir \
          --prompt="$EDIT_PROMPT" \
          --ckpt_i=-1
done;
