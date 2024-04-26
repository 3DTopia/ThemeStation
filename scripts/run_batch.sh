OFFLINE_MODE=0
export TRANSFORMERS_OFFLINE=$OFFLINE_MODE
export DIFFUSERS_OFFLINE=$OFFLINE_MODE
export HF_HUB_OFFLINE=$OFFLINE_MODE

# run all cases under a folder
python run.py --run_batch --test_img_dir data/concept_images/elevation_0 --gpu_ids 0 --get_multi_view --get_coarse_3d --train_dreambooth --get_optim_3d --task_name demo_batch --seed 42 --batch_range all --port 30000  --remove_dreambooth_ckpt
