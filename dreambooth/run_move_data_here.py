# move data_rgb (concept image + 6 views from wonder3d)
import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_name", type=str, default=None, help="if None run all images under test_img_dir"
)
parser.add_argument(
    "--test_img_dir",
    type=str,
    required=True,
    default='"data/concept_images/elevation_0',
    help="",
)
args = parser.parse_args()

obj_name_list = []

if args.obj_name is not None:
    obj_name_list.append(args.obj_name)

include_normal = False
variation_image_root = (
    f"../{args.test_img_dir}"
)
views_6_root = "../Wonder3D/outputs/cropsize-192-cfg3.0"
data_rgb_dir = (
    "data/data_rgb"
)
if include_normal:
    data_rgb_dir += "_normal"

for f in os.listdir(variation_image_root):
    scene = None
    scene_name = None
    for obj_name in obj_name_list:
        if f.startswith(obj_name):
            scene = f.split(".png")[0]
            if "_clipdrop-background-removal.png" in f:
                scene_name = f.split("_clipdrop-background-removal.png")[0]
            else:
                scene_name = scene
            os.makedirs(os.path.join(data_rgb_dir, scene_name), exist_ok=True)
            shutil.copy(
                os.path.join(variation_image_root, f),
                os.path.join(data_rgb_dir, scene_name, f),
            )
            break
    if scene is not None and scene_name is not None:
        for view_f in os.listdir(os.path.join(views_6_root, scene)):
            if view_f.startswith("rgb"):
                shutil.copy(
                    os.path.join(views_6_root, scene, view_f),
                    os.path.join(data_rgb_dir, scene_name, view_f),
                )
        if include_normal:
            for view_f in os.listdir(os.path.join(views_6_root, scene, "normals")):
                if view_f.startswith("normals"):
                    shutil.copy(
                        os.path.join(views_6_root, scene, "normals", view_f),
                        os.path.join(data_rgb_dir, scene_name, view_f),
                    )
