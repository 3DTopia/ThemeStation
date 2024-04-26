import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_name", type=str, default=None, help="if None run all images under image_root"
)
parser.add_argument(
    "--remove_ckpt",
    action="store_true",
    help="delete the target ckpt",
)
parser.add_argument(
    "--dreambooth_type",
    type=str,
    default=None,
    help="delete the target ckpt",
)
args = parser.parse_args()

base_list = []

if args.dreambooth_type == "reference_model":
    base_list.append(
        "ckpt/reference_model_elevation_rand"
    )
elif args.dreambooth_type == "variation_model":
    base_list.append(
        "ckpt/reference_model_elevation_0"
    )
    base_list.append(
        "ckpt/reference_model_elevation_20"
    )
elif args.dreambooth_type == "base_model":
    base_list.append(
        "ckpt/base_model"
    )

for base in base_list:
    if not os.path.exists(base):
        continue
    for i in os.listdir(base):
        if not i.startswith(args.obj_name):
            continue
        if args.remove_ckpt:
            try:
                os.remove(os.path.join(base, i))
            except:
                shutil.rmtree(os.path.join(base, i))
            break

        for j in os.listdir(os.path.join(base, i)):
            if j.startswith("checkpoint"):
                for k in os.listdir(os.path.join(base, i, j)):
                    if k not in ["text_encoder", "unet"]:
                        try:
                            os.remove(os.path.join(base, i, j, k))
                        except:
                            os.removedirs(os.path.join(base, i, j, k))
