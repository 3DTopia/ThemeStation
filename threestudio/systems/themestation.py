import os
from dataclasses import dataclass, field

import torch
from torchmetrics.image import TotalVariation

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.systems.utils import parse_optimizer, parse_scheduler
import cv2
from loss.contextual import ContextualLoss
import numpy as np
import random


@threestudio.register("themestation-system")
class ThemeStation(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        stage: str = "coarse"
        visualize_samples: bool = False
        use_camera_embeddings: bool = False
        use_reference_model_dreambooth: bool = False
        use_base_model_dreambooth: bool = False
        use_prompt_norm: bool = False
        use_contextual_loss: bool = False
        use_rgb_contextual_loss: bool = False
        use_tv_loss: bool = False
        save_init_img_camera: bool = False
        start_mesh_deformation_step: int = -1

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.video_dir = "outputs_video"
        self.base_without_norm = True  # not apply normal loss in base model

        if self.cfg.use_contextual_loss:
            self.rgb_view_store_contextual = dict()
            self.view_ray_dict_contextual = dict()
            self.contextual_loss = ContextualLoss()

        if self.cfg.use_tv_loss:
            self.tv_loss = TotalVariation()

        if self.cfg.use_base_model_dreambooth:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()
        else:
            self.guidance = None

        if self.cfg.use_reference_model_dreambooth:
            self.ref_guidance = threestudio.find(self.cfg.guidance_type)(
                self.cfg.ref_guidance
            )
            self.ref_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.ref_prompt_processor)
            self.ref_prompt_utils = self.ref_prompt_processor()

        else:
            self.ref_guidance = None

        if self.cfg.use_prompt_norm:
            if not self.base_without_norm and self.cfg.use_base_model_dreambooth:
                self.prompt_norm_processor = threestudio.find(
                    self.cfg.prompt_processor_type
                )(self.cfg.prompt_norm_processor)
                self.prompt_norm_utils = self.prompt_norm_processor()

            if self.cfg.use_reference_model_dreambooth:
                self.ref_prompt_norm_processor = threestudio.find(
                    self.cfg.prompt_processor_type
                )(self.cfg.ref_prompt_norm_processor)
                self.ref_prompt_norm_utils = self.ref_prompt_norm_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if "shape_init" in self.cfg.geometry.keys():
            raise Exception("not implemented error")
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        #  change learning rate at certain step
        if self.cfg.optimizer.update_lr_at_certain_step:
            milestone = self.cfg.optimizer.milestone
            new_lrs = self.cfg.optimizer.new_lr
            optim = self.optimizers().optimizer
            if batch_idx == milestone:
                threestudio.info("change old lr to new:")
                for g in optim.param_groups:
                    threestudio.info(
                        f"change {g['name']}'s lr from {g['lr']} to {new_lrs[g['name']]['lr']}."
                    )
                    g["lr"] = new_lrs[g["name"]]["lr"]

        # fixed sdf and use deformation to optimize the mesh
        if batch_idx == self.cfg.start_mesh_deformation_step:
            print(
                f"self.geometry.cfg.optim_mesh_via_deformation:{self.geometry.cfg.optim_mesh_via_deformation}"
            )
            self.geometry.cfg.optim_mesh_via_deformation = True

        out = self(batch)

        if self.cfg.stage == "dsd":
            guidance_inp = dict()
            guidance_out = dict()
            guidance_inp["normal"] = out["comp_normal"]
            guidance_inp["rgb"] = out["comp_rgb"]

            if self.cfg.use_base_model_dreambooth:
                guidance_out["rgb"] = self.guidance(
                    guidance_inp["rgb"],
                    self.prompt_utils,
                    **batch,
                    rgb_as_latents=False,
                )

                if not self.base_without_norm:
                    if self.cfg.use_prompt_norm:
                        guidance_out["normal"] = self.guidance(
                            guidance_inp["normal"],
                            self.prompt_norm_utils,
                            **batch,
                            rgb_as_latents=False,
                        )
                    else:
                        guidance_out["normal"] = self.guidance(
                            guidance_inp["normal"],
                            self.prompt_utils,
                            **batch,
                            rgb_as_latents=False,
                        )
        else:
            raise Exception("not impleted error")

        if self.cfg.use_reference_model_dreambooth:
            assert self.ref_guidance is not None
            assert self.ref_prompt_utils is not None
            if self.cfg.stage == "dsd":
                ref_guidance_out = dict()
                ref_guidance_out["rgb"] = self.ref_guidance(
                    guidance_inp["rgb"],
                    self.ref_prompt_utils,
                    **batch,
                    rgb_as_latents=False,
                )

                if self.cfg.use_prompt_norm:
                    ref_guidance_out["normal"] = self.ref_guidance(
                        guidance_inp["normal"],
                        self.ref_prompt_norm_utils,
                        **batch,
                        rgb_as_latents=False,
                    )
                else:
                    ref_guidance_out["normal"] = self.ref_guidance(
                        guidance_inp["normal"],
                        self.ref_prompt_utils,
                        **batch,
                        rgb_as_latents=False,
                    )
            else:
                raise Exception("not impleted error")

        loss = 0.0

        if self.cfg.use_contextual_loss:
            # contextual loss
            # save reference fixed views with camera into dicts for calculating contextual loss
            if len(self.view_ray_dict_contextual.keys()) == 0:
                # read view ray dict
                obj_name = self.get_save_dir().split("/")[-2]
                view_ray_dict_dir = (
                    f"./outputs/rendered_init_images/{obj_name}/save/it5000-test"
                )
                threestudio.info(f"view_ray_dict_dir:{view_ray_dict_dir}")
                for i in os.listdir(view_ray_dict_dir):
                    if i.endswith("_view_ray_dict.pth"):
                        id = int(i.split("_view_ray_dict.pth")[0])
                        self.view_ray_dict_contextual[id] = torch.load(
                            os.path.join(view_ray_dict_dir, i)
                        )

                img2img_strength = 50  # TODO: turn this hard code into config
                if self.cfg.use_rgb_contextual_loss:
                    # save reference images into view store
                    obj_name = self.get_save_dir().split("/")[-2]
                    ref_img_root = "./dreambooth/data/img2img_20views"
                    reference_view_dir = os.path.join(
                        ref_img_root,
                        f"{obj_name}_text",
                        f"checkpoint-150_strength{img2img_strength}",
                    )
                    for i in os.listdir(reference_view_dir):
                        if i.endswith(".png"):
                            img = cv2.imread(os.path.join(reference_view_dir, i))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                            img = torch.from_numpy(img)[None, :]
                            id = int(i.split(".")[0])
                            self.rgb_view_store_contextual[id] = img.float().to("cuda")

            # sample current views
            ref_view_num = len(self.view_ray_dict_contextual.keys())
            sample_num = 4
            symmetric = False
            rand_view_ix = random.sample(range(ref_view_num), sample_num)
            loss_contextual_rgb = 0.0
            for i in rand_view_ix:
                context_out = self(self.view_ray_dict_contextual[int(i)])
                cur_view_i = context_out["comp_rgb"]

                if self.cfg.use_rgb_contextual_loss:
                    if symmetric:
                        loss_contextual_rgb += (
                            self.contextual_loss(
                                self.rgb_view_store_contextual[i].permute(0, 3, 1, 2),
                                cur_view_i.permute(0, 3, 1, 2),
                            )
                            + self.contextual_loss(
                                cur_view_i.permute(0, 3, 1, 2),
                                self.rgb_view_store_contextual[i].permute(0, 3, 1, 2),
                            )
                        ) / 2
                    else:
                        loss_contextual_rgb += self.contextual_loss(
                            cur_view_i.permute(0, 3, 1, 2),
                            self.rgb_view_store_contextual[i].permute(0, 3, 1, 2),
                        )

            if self.cfg.use_rgb_contextual_loss:
                loss += (
                    loss_contextual_rgb
                    / sample_num
                    * self.C(self.cfg.loss.lambda_contextual_rgb)
                )

        if self.cfg.use_tv_loss:
            loss_tv = self.tv_loss(out["comp_rgb"].permute(0, 3, 1, 2))
            loss += loss_tv * self.C(self.cfg.loss.lambda_tv)

        if self.cfg.stage == "dsd":
            if self.cfg.use_base_model_dreambooth:
                if not self.base_without_norm:
                    for name, value in guidance_out["normal"].items():
                        self.log(f"train/{name}", value)
                        if name.startswith("loss_"):
                            loss += value * self.C(
                                self.cfg.loss[name.replace("loss_", "lambda_")]
                            )
                for name, value in guidance_out["rgb"].items():
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        loss += value * self.C(
                            self.cfg.loss[name.replace("loss_", "lambda_")]
                        )

            if self.cfg.use_reference_model_dreambooth:
                for name, value in ref_guidance_out["normal"].items():
                    self.log(f"train/{name}_ref", value)
                    if name.startswith("loss_"):
                        loss += value * self.C(
                            self.cfg.loss[name.replace("loss_", "lambda_ref_")]
                        )

                for name, value in ref_guidance_out["rgb"].items():
                    self.log(f"train/{name}_ref", value)
                    if name.startswith("loss_"):
                        loss += value * self.C(
                            self.cfg.loss[name.replace("loss_", "lambda_ref_")]
                        )
        else:
            raise Exception("not impleted error")

        if self.cfg.stage == "coarse" or self.cfg.stage == "dsd":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if "z_variance" in out:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            # sdf loss
            if "sdf_grad" in out:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
                self.log("train/inv_std", out["inv_std"], prog_bar=True)

        if self.cfg.stage == "geometry" or self.cfg.stage == "dsd":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )

        if self.cfg.stage == "texture":
            raise Exception("not impleted error")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ],
            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ],
            name="test_step",
            step=self.true_global_step,
        )

        if self.cfg.use_camera_embeddings:
            prompt_v = self.prompt_utils.get_view_dependent_prompt(
                self.prompt_processor.prompt,
                batch["elevation"],
                batch["azimuth"],
                batch["camera_distances"],
            )
            self.save_view_dependent_prompt(
                f"it{self.true_global_step}-test/{batch['index'][0]}_{prompt_v}.txt",
                prompt_v,
            )
            print(prompt_v)

        if self.cfg.save_init_img_camera:
            init_img_camera = batch
            init_img_camera["use_white_background"] = True
            self.save_pth(
                f"it{self.true_global_step}-test/{batch['index'][0]}_view_ray_dict.pth",
                init_img_camera,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
            video_dir=self.video_dir,
        )
