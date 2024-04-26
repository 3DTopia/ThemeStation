OFFLINE_MODE=0
export TRANSFORMERS_OFFLINE=$OFFLINE_MODE
export DIFFUSERS_OFFLINE=$OFFLINE_MODE
export HF_HUB_OFFLINE=$OFFLINE_MODE

# run a single case
python run.py --test_img_dir data/concept_images/elevation_20 --obj_name owl_freepik-reimagine --gpu_ids 0 --get_multi_view --get_coarse_3d --train_dreambooth --get_optim_3d  --task_name demo --seed 42 --port 30000 --remove_dreambooth_ckpt
# python run.py --test_img_dir data/concept_images/elevation_20 --obj_name owl_1 --gpu_ids 0 --get_multi_view --get_coarse_3d --train_dreambooth --get_optim_3d  --task_name demo --seed 42 --port 30000 --remove_dreambooth_ckpt