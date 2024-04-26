from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--round_1_rendering_path", required=True)
parser.add_argument("--obj_name", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--round_i", type=int, required=False, default=1)
parser.add_argument("--normal", type=str, required=True, default="")
parser.add_argument("--seed", type=int, required=False, default=42)

args, extras = parser.parse_known_args()
round_1_rendering_path = args.round_1_rendering_path
obj_name = args.obj_name
prompt = args.prompt
normal = args.normal

round_i = args.round_i if args.round_i != 1 else 1
if round_i == 1:
    strength_list = [0.5]
else:
    strength_list = np.array(range(10, 55, 5)) / 100.0
train_step = 150
lr = "-lr2e-6"
bs = "_bs8"
num_view = "20views"
save_dir = f"./data/img2img_{num_view}"
n_steps = 200
print(prompt)
for strength in strength_list:
    for ckpt_i in [train_step]:
        # inputs
        model_tag = f"{obj_name}_text/checkpoint-{ckpt_i}"

        # fixed code
        ckpt_path = f"./ckpt/base_model/{model_tag}"
        # ckpt_path = f"./ckpt/camera_ckpt_init{train_step}{lr}{bs}/{model_tag}"
        model_id = "stabilityai/stable-diffusion-2-1-base"

        unet = UNet2DConditionModel.from_pretrained(f"{ckpt_path}/unet")

        # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
        text_encoder = CLIPTextModel.from_pretrained(f"{ckpt_path}/text_encoder")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            unet=unet,
            text_encoder=text_encoder,
            dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        for f in os.listdir(round_1_rendering_path):
            if not f.endswith(".png"):
                continue
            if "normal map" in prompt:
                init_image = Image.fromarray(
                    np.array(Image.open(os.path.join(round_1_rendering_path, f)))[
                        :, 512:1024
                    ]
                )
            else:
                init_image = Image.fromarray(
                    np.array(Image.open(os.path.join(round_1_rendering_path, f)))[
                        :, :512
                    ]
                )
            generator = [torch.Generator(device="cuda").manual_seed(args.seed)]
            image = pipe(
                prompt=prompt,
                generator=generator,
                image=init_image,
                strength=strength,
                num_inference_steps=n_steps,
                guidance_scale=7.5,
                num_images_per_prompt=1,
            ).images[0]

            if not os.path.exists(
                f"{save_dir}/{model_tag}_strength{int(strength*100)}"
            ):
                os.makedirs(f"{save_dir}/{model_tag}_strength{int(strength*100)}")
            if "normal map" in prompt:
                image.save(
                    f"{save_dir}/{model_tag}_strength{int(strength*100)}/normal_{f.split('.')[0]}.png"
                )
            else:
                image.save(
                    f"{save_dir}/{model_tag}_strength{int(strength*100)}/{f.split('.')[0]}.png"
                )
