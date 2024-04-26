#controllable generation: --editing_prompt
python generate_concept.py --obj_name owl --editing_prompt "pink" --obj_type single --base_dir ../data/reference_models/renderings/elevation_$1 --gpu_ids 0 --port 20000

