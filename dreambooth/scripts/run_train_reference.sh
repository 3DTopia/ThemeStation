# learn reference piror given:
# rendered images of the reference model + 
# concept image itself + 
# multi-views of concept image (from Wonder3D) + 
# sampled pseudo multi-views (augmented views of initial 3D model)

elevation=rand
PORT=$6
VAR_IMG_DIR=$7
PSEUDO_IMG_DIR=$8
OUTPUT_DIR=${9}
learning_rate=${10}

obj_name=$1
obj_name_no_id=$2

PROMPT=$3
num_ref_imgs=30

INSTANCE_DIR="../data/reference_models/renderings/elevation_"$elevation"/"$obj_name_no_id"" # edit this line

MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
accelerate launch --config_file ../configs/acc_config.yaml --main_process_port $PORT --num_processes 1 --gpu_ids $5 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="$PROMPT" \
  --resolution=512 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --learning_rate=$learning_rate \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=150 \
  --train_text_encoder \
  --checkpointing_steps=150 \
  --pseudo_data_dir=$PSEUDO_IMG_DIR \
  --variation_data_dir=$VAR_IMG_DIR \
  --seed=${11} \
  # --strat_checkpointing_step=150 \
  # --class_embed_type="projection" \
  # --class_labels_conditioning="camera_emb" \