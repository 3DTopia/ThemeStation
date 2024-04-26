from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--image_type", default="rgb", help="rgb or normal")
parser.add_argument("--ckpt_i", type=int, required=False, default=-1)

args, extras = parser.parse_known_args()
root_dir = args.root_dir
out_dir = args.output_dir
prompt = args.prompt
ckpt_i = args.ckpt_i

assert args.image_type in ["rgb", "normal"]
if args.image_type == "normal":
    prompt = prompt + ", normal map"

print(prompt)
num_images_per_prompt = 5

if ckpt_i == -1:  # concept image of a single reference model
    ckpt_list = range(75, 251, 25)
    num_images_per_prompt = 20
elif ckpt_i == -2:  # reference prior visualization
    ckpt_list = [150]
elif ckpt_i == -3:  # concept image of a group of reference models
    ckpt_list = range(100, 351, 25)
    num_images_per_prompt = 20
else:
    ckpt_list = [ckpt_i]  # concept prior visualization

for i in ckpt_list:
    for cfg_scale in [7.5]:
        # inputs
        ckpt_path = f"{root_dir}/checkpoint-{i}"
        print(ckpt_path)

        model_id = "stabilityai/stable-diffusion-2-1-base"

        unet = UNet2DConditionModel.from_pretrained(f"{ckpt_path}/unet")

        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(f"{ckpt_path}/text_encoder")

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            unet=unet,
            text_encoder=text_encoder,
            dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        if ckpt_i in [-1, -3]:  # run twrice for more samples
            images = pipe(
                prompt,
                num_inference_steps=100,
                guidance_scale=cfg_scale,
                num_images_per_prompt=num_images_per_prompt,
            ).images  # [0,255]
            images += pipe(
                prompt,
                num_inference_steps=100,
                guidance_scale=cfg_scale,
                num_images_per_prompt=num_images_per_prompt,
            ).images
        else:
            generator = [
                torch.Generator(device="cuda").manual_seed(i)
                for i in range(num_images_per_prompt)
            ]
            images = pipe(
                prompt,
                generator=generator,
                num_inference_steps=100,
                guidance_scale=cfg_scale,
                num_images_per_prompt=num_images_per_prompt,
            ).images  # [0,255]

        for j, image in enumerate(images):
            if not os.path.exists(f"{out_dir}/checkpoint-{i}"):
                os.makedirs(f"{out_dir}/checkpoint-{i}")
            image.save(
                f"{out_dir}/checkpoint-{i}/cfg_{int(cfg_scale*10)}_{prompt}_{j}.png"
            )
