# run 3d-to-3d generation pipeline
# before run, make sure:
#   - generate concept images and remove backgroud, under data/concept_images
#   - render images of reference model, under data/reference_models/renderings/elevation_rand


import os
import argparse
import subprocess


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def run_bash(bash_args, logger=None):
    process = subprocess.Popen(
        bash_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
    )

    while True:
        if process.poll() is not None:
            break

        output = process.stdout.readline().decode("utf-8")
        print(output, end="")
        if logger is not None:
            logger.write(output)
            logger.flush()

def run_pipe(gpu_ids, image_name, args):
    if "_clipdrop-background-removal.png" in image_name:
        obj_name = image_name.split("_clipdrop-background-removal.png")[0]
        suffix = "_clipdrop-background-removal"
    else:
        obj_name = image_name.split(".png")[0]
        suffix = ""

    logfile_name = os.path.join(args.log_root, f"{obj_name}.txt")
    os.makedirs(args.log_root, exist_ok=True)
    logfile = open(logfile_name, "w")

    # 1. run wonder3d to generate multi-view images
    if args.get_multi_view:
        print("get multi-view!")
        with cd("./Wonder3D"):
            out = subprocess.check_output("pwd")
            print(f"enter {out}")

            if not os.path.exists("./outputs/cropsize-192-cfg3.0"):
                os.makedirs("./outputs/cropsize-192-cfg3.0")
                
            if image_name.split(".png")[0] in os.listdir(
                "./outputs/cropsize-192-cfg3.0"
            ):
                print(f"multi-view of {obj_name} exits, skip!")
            else:
                image_path = f"../{args.test_img_dir}/{image_name}"
                mvdiffusion_args = [
                    f"sh run_mv.sh {obj_name}{suffix} {gpu_ids[0]} {image_path} {args.seed}"
                ]
                run_bash(mvdiffusion_args, logfile)

                print(f"Multi-view images of {obj_name} generated!")

    # 2. run wonder3d to generate initial 3D model
    if args.get_coarse_3d:
        coarse_3d_dir = f"./Wonder3D/instant-nsr-pl/exp/mesh-ortho-{obj_name}/ckpts/epoch=0-step=5000.ckpt"
        if os.path.exists(coarse_3d_dir):
            print("coarse 3d exist, skip!")
        else:
            print("get coarse 3D!")
            with cd("./Wonder3D/instant-nsr-pl"):
                coarse3d_args = [
                    f"sh run_coarse3d.sh {obj_name}{suffix} {gpu_ids[0]} {args.port}"
                ]
                run_bash(coarse3d_args, logfile)

                print(f"Coarse3D model of {obj_name} generated!")

            # render coarse3D videos
            coarse3d_ckpt_path = f"./Wonder3D/instant-nsr-pl/exp/mesh-ortho-{obj_name}{suffix}/ckpts/epoch=0-step=5000.ckpt"
            config_path = f"configs/theme-station-render.yaml"
            render_coarse3D_images_args = [
                f"sh scripts/run_coarse3d_video.sh {obj_name} {coarse3d_ckpt_path} {gpu_ids[0]} {config_path} {args.input_image_elevation_offset}"
            ]
            run_bash(render_coarse3D_images_args, logfile)

            print(f"video of coarse 3d {obj_name} generated!")

    # 3. learn concept prior and reference prior
    if args.train_dreambooth:
        # train init dreambooth and perform img2img to obtain pseudo multi-view images
        init_db_dir = f"./dreambooth/ckpt/base_model/{obj_name}_text/checkpoint-300/text_encoder/config.json"
        if os.path.exists(init_db_dir):
            print("concept prior has been learned before, skip!")
        else:
            print("train init dreambooth and concept prior!")

            rand_view_dir = (
                f"./outputs/rendered_init_images/{obj_name}/save/it5000-test"
            )
            if os.path.exists(rand_view_dir):
                print("random views exist, skip!")
            else:
                print("get random views of coarse 3d")
                coarse3d_ckpt_path = f"./Wonder3D/instant-nsr-pl/exp/mesh-ortho-{obj_name}{suffix}/ckpts/epoch=0-step=5000.ckpt"
                config_path = f"configs/theme-station-render.yaml"
                render_coarse3D_images_args = [
                    f"sh scripts/run_render_init_images.sh {obj_name} {coarse3d_ckpt_path} {gpu_ids[0]} {config_path} {args.input_image_elevation_offset}"
                ]
                run_bash(render_coarse3D_images_args, logfile)
                print(f"random views of coarse 3d {obj_name} generated!")

            # train init dreambooth
            with cd("./dreambooth"):
                out = subprocess.check_output("pwd")
                print(f"enter {out}")

                obj_name_only = " ".join(
                    [x for x in obj_name.split("_") if not x.isnumeric()]
                )
                prompt = f"a 3d model of sks {obj_name_only}"
                print(f"prompt: {prompt}")
                round_1_rendering_path = (
                    f"../outputs/rendered_init_images/{obj_name}/save/it5000-test"
                )

                learning_rate = "2e-6"
                print(f"init_db learning rate={learning_rate}")
                train_init_dreambooth_args = [
                    f'python run_move_data_here.py --obj_name {obj_name} --test_img_dir {args.test_img_dir}; \
                        sh scripts/run_train_init.sh {obj_name} "{prompt}" {round_1_rendering_path} {",".join(gpu_ids)} {gpu_ids[0]}  {args.port} "{learning_rate}" {args.seed}'
                ]
                run_bash(train_init_dreambooth_args, logfile)

                print(f"Train initial dreambooth of {obj_name} and perform img2img2!")

            # learn concept prior
            with cd("./dreambooth"):
                out = subprocess.check_output("pwd")
                print(f"enter {out}")

                obj_name_only = " ".join(
                    [x for x in obj_name.split("_") if not x.isnumeric()]
                )
                prompt = f"a 3d model of sks {obj_name_only}"
                print(f"prompt: {prompt}")

                learning_rate = "2e-6"
                print(f"concept prior learning rate={learning_rate}")
                train_base_dreambooth_args = [
                    f'sh scripts/run_train_concept.sh {obj_name} "{prompt}" "{",".join(gpu_ids)}" "{gpu_ids[0]}" {args.port} "{learning_rate}" {args.seed}; \
                    python run_remove_optimizer.py --obj_name {obj_name} --dreambooth_type base_model; '
                ]
                run_bash(train_base_dreambooth_args, logfile)

                print(f"learn concept prior of {obj_name}!")

        # learn reference prior
        with cd("./dreambooth"):
            out = subprocess.check_output("pwd")
            print(f"enter {out}")

            obj_name_only = " ".join(
                [x for x in obj_name.split("_") if not x.isnumeric()]
            )
            prompt = f"a 3d model of {obj_name_only}, in the style of sks"
            print(f"prompt: {prompt}")

            obj_name_no_id = "_".join(obj_name.split("_")[:-1])

            # reference dreambooth
            ref_db_dir = f"./ckpt/reference_model_elevation_rand/{obj_name}"
            if os.path.exists(ref_db_dir + "/checkpoint-150/text_encoder/config.json"):
                print(f"reference prior exist: {ref_db_dir}, skip!")
            else:
                print("learn reference prior!")

                variation_image_dir = f"./data/data_rgb/{obj_name}"
                pseudo_image_dir = (
                    f"./data/img2img_20views/{obj_name}_text/checkpoint-150_strength50"
                )
                learning_rate = "2e-6"
                print(f"reference prior learning rate={learning_rate}")
                train_ref_dreambooth_args = [
                    f'sh scripts/run_train_reference.sh {obj_name} {obj_name_no_id} "{prompt}" "{",".join(gpu_ids)}" "{gpu_ids[0]}" {args.port} {variation_image_dir} {pseudo_image_dir} {ref_db_dir} {learning_rate} {args.seed}; \
                    python run_remove_optimizer.py --obj_name {obj_name} --dreambooth_type reference_model; '
                ]
                run_bash(train_ref_dreambooth_args, logfile)

                print(
                    f"learn reference prior of {obj_name}!"
                )
    
    # 4. perfrom reference-informed 3d asset modeling to get optimized 3D model
    if args.get_optim_3d:
        print("get optimized 3D!")
        # 7. start optimization
        coarse3d_ckpt_path = f"./Wonder3D/instant-nsr-pl/exp/mesh-ortho-{obj_name}{suffix}/ckpts/epoch=0-step=5000.ckpt"
        obj_name_only = " ".join([x for x in obj_name.split("_") if not x.isnumeric()])
        prompt = f"a 3d model of sks {obj_name_only}"
        ref_prompt = f"a 3d model of {obj_name_only}, in the style of sks"
        print(f"prompt: {prompt}")
        print(f"ref prompt: {ref_prompt}")

        obj_name_no_id = "_".join(obj_name.split("_")[:-1])

        yaml = f"theme-station-optimization.yaml"

        if not args.resume:
            optimization_args = [
                f'sh scripts/run_optimization.sh {obj_name} {obj_name_no_id} {coarse3d_ckpt_path} "{prompt}" "{ref_prompt}" "{gpu_ids[0]}" "{args.task_name}" {yaml} {args.input_image_elevation_offset};'
            ]

        else:
            coarse3d_ckpt_path = f"./outputs/{args.task_name}/{obj_name}"
            optimization_args = [
                f'sh scripts/run_optimization_resume.sh {obj_name} {obj_name_no_id} {coarse3d_ckpt_path} "{prompt}" "{ref_prompt}" "{gpu_ids[0]}" "{args.task_name}" {yaml} {args.input_image_elevation_offset};'
            ]

        run_bash(optimization_args, logfile)

        # remove dreambooth ckpt to save memory after inference
        if args.remove_dreambooth_ckpt:
            remove_ckpt_state = "--remove_ckpt"

            with cd("./dreambooth"):
                out = subprocess.check_output("pwd")
                print(f"enter {out}")

                remove_ckpt_args = [
                    f"python run_remove_optimizer.py --obj_name {obj_name} --dreambooth_type base_model {remove_ckpt_state}; \
                        python run_remove_optimizer.py --obj_name {obj_name} --dreambooth_type reference_model {remove_ckpt_state};"
                ]
                run_bash(remove_ckpt_args, logfile)

                print(f"Dreambooth ckpt of {obj_name} is deleted!")

    if args.export_mesh:
        print(f"export mesh for {obj_name}!")
        # 8. start exporting mesh
        resume_ckpt_base = f"./outputs/{args.task_name}/{obj_name}"

        if os.path.exists(
            os.path.join(resume_ckpt_base, "save", "it5000-export", "model.obj")
        ) or os.path.exists(
            os.path.join(resume_ckpt_base, "save", "it10000-export", "model.obj")
        ):
            print(f"{obj_name} mesh was already exported! skip!")
            return

        if os.path.exists(
            os.path.join(resume_ckpt_base, "ckpts", "epoch=0-step=10000.ckpt")
        ):
            resume_ckpt_path = os.path.join(
                resume_ckpt_base, "ckpts", "epoch=0-step=10000.ckpt"
            )
        elif os.path.exists(
            os.path.join(resume_ckpt_base, "ckpts", "epoch=0-step=5000.ckpt")
        ):
            resume_ckpt_path = os.path.join(
                resume_ckpt_base, "ckpts", "epoch=0-step=5000.ckpt"
            )
        else:
            print(f"{obj_name} optim 3D is not generated!")
            return

        assert os.path.exists(os.path.join(resume_ckpt_base, "configs", "parsed.yaml"))
        resume_config = os.path.join(resume_ckpt_base, "configs", "parsed.yaml")
        yaml = f"theme-station-optimization.yaml"

        optimization_args = [
            f'sh scripts/run_export_mesh.sh  {obj_name} "{resume_ckpt_path}" "{resume_config}"  "{yaml}" "{args.task_name}" "{gpu_ids[0]}";'
        ]

        run_bash(optimization_args, logfile)

    print("Finished see {} for logs".format(logfile_name))


def main(gpu_ids, obj_name, args):
    image_root = args.test_img_dir
    assert os.path.exists(image_root)
    print(f"{len(sorted(os.listdir(image_root)))} concept images in {image_root}")
    for image_ix, image in enumerate(sorted(os.listdir(image_root))):
        print(f"process case {image_ix}: {image}")
        if not image.endswith(".png"):
            continue

        if args.run_batch:
            assert args.batch_range is not None
            # only process images with index in batch_range
            if args.batch_range == "all":
                start_ix = 0
                print(image_root)
                end_ix = len(os.listdir(image_root))
                print(f"process all {end_ix} cases")
            else:
                start_ix, end_ix = int(args.batch_range.split(",")[0]), int(
                    args.batch_range.split(",")[1]
                )

            if image in sorted(os.listdir(image_root))[start_ix:end_ix]:
                obj_name = image.split(".png")[0]
                print(f"process obj: {obj_name}")
                if (
                    os.path.exists(
                        os.path.join(
                            "./outputs",
                            args.task_name,
                            obj_name,
                            "ckpts",
                            "epoch=0-step=5000.ckpt",
                        )
                    )
                    and not args.resume
                    and not (args.export_mesh and not args.get_optim_3d)
                ):
                    print(f"{obj_name} has been processed before, skip!")
                    continue

                options = ""
                if args.get_multi_view:
                    options += "--get_multi_view "
                if args.get_coarse_3d:
                    options += "--get_coarse_3d "
                if args.train_dreambooth:
                    options += "--train_dreambooth "
                if args.get_optim_3d:
                    options += "--get_optim_3d "
                if args.resume:
                    options += "--resume "
                if args.export_mesh:
                    options += "--export_mesh "

                run_args = [
                    f'python run.py --test_img_dir "{args.test_img_dir}" --obj_name {obj_name} --gpu_ids {args.gpu_ids} {options} --task_name "{args.task_name}" --port {args.port} --remove_dreambooth_ckpt --seed {args.seed};'
                ]
                print(f"run_args:{run_args}")

                run_bash(run_args, logger=None)

                print(f"done obj: {obj_name}")
        else:
            if obj_name is not None and not image.startswith(obj_name):
                continue

            run_pipe(
                gpu_ids,
                image_name=image,
                args=args,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="exp dir name",
    )
    parser.add_argument(
        "--obj_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_img_dir",
        type=str,
        default="data/concept_images/elevation_0",
    )
    parser.add_argument(
        "--gpu_ids", type=str, required=False, default="0", help="0,1,2,3"
    )
    parser.add_argument("--seed", type=str, required=False, default=42, help="random seed")
    parser.add_argument("--log_root", type=str, required=False, default='./logs', help="")
    parser.add_argument(
        "--input_image_elevation_offset",
        type=str,
        required=False,
        default=0,
        help="elevation of the object in the input image",
    )
    parser.add_argument("--port", type=int, required=True, help="port")
    parser.add_argument("--get_multi_view", action="store_true")
    parser.add_argument("--get_coarse_3d", action="store_true")
    parser.add_argument("--train_dreambooth", action="store_true")
    parser.add_argument("--get_optim_3d", action="store_true")
    parser.add_argument("--export_mesh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--remove_dreambooth_ckpt", action="store_true", help="delete ckpt files to save memeory after inference")
    parser.add_argument(
        "--run_batch", action="store_true", help="run all images under test_img_dir"
    )
    parser.add_argument(
        "--batch_range", type=str, required=False, help="start index,end index, such '0,20', use 'all' will run all images", default='all'
    )
    args = parser.parse_args()

    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
    gpu_ids = [gpu_ids[int(args.gpu_ids.split(",")[0])]]

    if args.test_img_dir.split("_")[-1].isdigit():
        args.input_image_elevation_offset = int(args.test_img_dir.split("_")[-1])

    print(f"CUDA_VISIBLE_DEVICES:{gpu_ids}")
    print(f"gpu_ids:{gpu_ids}")
    print(f"seed:{args.seed}")
    print(f"elevation:{args.input_image_elevation_offset}")

    main(gpu_ids, args.obj_name, args)
