OFFLINE_MODE=0
export TRANSFORMERS_OFFLINE=$OFFLINE_MODE
export DIFFUSERS_OFFLINE=$OFFLINE_MODE
export HF_HUB_OFFLINE=$OFFLINE_MODE

# export mesh
python run.py --test_img_dir data/concept_images/elevation_20 --obj_name owl_1 --gpu_ids 0 --export_mesh --task_name demo --seed 42 --port 30000