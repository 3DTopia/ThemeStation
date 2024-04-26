"""
This file is adapted from https://github.com/Fyusion/LLFF.
"""

import os
import argparse
import subprocess
import random


def main(gpu_ids, base_dir, obj_type, port, args):
    image_root = base_dir
    assert os.path.exists(image_root)
    image_list = os.listdir(image_root)
    random.shuffle(image_list)
    for obj_name in image_list:
        if obj_name != args.obj_name:
            continue
        elevation = base_dir.split("_")[-1]
        print(f"elevation:{elevation}")
        if not os.path.exists(f"./output/concept_image_{obj_type}_elevation_{elevation}"):
            os.makedirs(f"./output/concept_image_{obj_type}_elevation_{elevation}")
        if not os.path.exists(
            f"./ckpt/reference_model_elevation_{elevation}"
        ):
            os.makedirs(
                f"./ckpt/reference_model_elevation_{elevation}"
            )

        if not len(args.editing_prompt) > 0:
            if obj_name in os.listdir(
                f"./ckpt/reference_model_elevation_{elevation}"
            ):
                continue

        print(f"train theme-driven diffusion for {obj_name}")

        obj_name_only = " ".join([x for x in obj_name.split("_") if not x.isnumeric()])
        prompt = f"a 3d model of {obj_name_only}, in the style of sks"
        edit_prompt = prompt

        # controllable concept image generation
        if len(args.editing_prompt) > 0:
            edit_prompt = f"a 3d model of {args.editing_prompt} {obj_name_only}, in the style of sks"

        print(f"obj_name: {obj_name}")
        print(f"prompt: {prompt}")
        print(f"editing: {edit_prompt}")

        if obj_type == "single":
            bash_file_name = "run_generate_concept_single.sh"
        elif obj_type == "group":
            bash_file_name = "run_generate_concept_group.sh"
        else:
            raise ValueError

        optimization_args = [
            f'sh scripts/{bash_file_name} {obj_type} {obj_name} "{prompt}" "{gpu_ids[0]}" {port} {elevation} "{edit_prompt}" "{args.editing_prompt}"; \
                python run_remove_optimizer.py --obj_name {obj_name} --remove_ckpt --dreambooth_type variation_model;'
        ]

        process = subprocess.Popen(
            optimization_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )

        while True:
            if process.poll() is not None:
                break

            output = process.stdout.readline().decode("utf-8")
            print(output, end="")

    print(f"finished {len(os.listdir(image_root))} objects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_type",
        type=str,
        required=True,
        help="single or group. if None run all images under image_root",
    )
    parser.add_argument(
        "--obj_name",
        type=str,
        required=True,
        help="single or group. if None run all images under image_root",
    )
    parser.add_argument(
        "--editing_prompt",
        type=str,
        required=False,
        default="",
        help="single or group. if None run all images under image_root",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./",
        help="rendered images of the reference model",
    )
    parser.add_argument("--gpu_ids", type=str, required=True, help="0,1,2,3")
    parser.add_argument("--port", type=int, required=True, help="20000~21000")
    args = parser.parse_args()

    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
    print(f"CUDA_VISIBLE_DEVICES:{gpu_ids}")
    gpu_ids = [gpu_ids[int(args.gpu_ids.split(",")[0])]]
    print(f"gpu_ids:{gpu_ids}")

    assert args.obj_type in ["group", "single"]
    main(gpu_ids, args.base_dir, args.obj_type, args.port, args)
