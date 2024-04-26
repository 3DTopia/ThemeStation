import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseImplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *
from threestudio.utils.misc import scale_anything
from threestudio.utils.ops import scale_tensor
from threestudio.models.isosurface import MarchingTetrahedraHelper


@threestudio.register("implicit-sdf")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        feature_dim: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

        # deformation
        optim_mesh_via_deformation: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()
        print(f"isosurface_remove_outliers:{self.cfg.isosurface_remove_outliers}")
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        # from sdf_network to network (as in Wonder3D);
        # here, network is used to compute both sdf (out[..:0]) and features (out)
        self.network = get_mlp(
            self.encoding.n_output_dims,
            self.cfg.feature_dim,
            self.cfg.mlp_network_config,
        )

        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )

        self.finite_difference_normal_eps: Optional[float] = None

        # deformation
        self.mesh = None
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.register_buffer("isosurface_bbox", self.bbox.clone())
        self.sdf: Float[Tensor, "Nv 1"]
        self.deformation: Optional[Float[Tensor, "Nv 3"]]
        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",
        )
        self.register_parameter(
            "sdf",
            nn.Parameter(
                torch.zeros(
                    (self.isosurface_helper.grid_vertices.shape[0], 1),
                    dtype=torch.float32,
                )
            ),
        )
        if self.cfg.isosurface_deformable_grid:
            self.register_parameter(
                "deformation",
                nn.Parameter(torch.zeros_like(self.isosurface_helper.grid_vertices)),
            )
        else:
            self.deformation = None

    def initialize_shape(self) -> None:
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = mesh.vertices.mean(0)
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            from pysdf import SDF

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm

        for _ in tqdm(
            range(1000),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            points_rand = (
                torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
            )
            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = True
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        out = self.network(enc).view(*points.shape[:-1], self.cfg.feature_dim)
        sdf, feature = out[..., 0], out
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {"sdf": sdf.view(*points.shape[:-1], 1)}
        
        if self.cfg.n_feature_dims > 0:
            features = feature
            output.update({"features": features})

        if output_normal:
            if self.cfg.normal_type == "finite_difference":
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                    [
                        [eps, 0.0, 0.0],
                        [-eps, 0.0, 0.0],
                        [0.0, eps, 0.0],
                        [0.0, -eps, 0.0],
                        [0.0, 0.0, eps],
                        [0.0, 0.0, -eps],
                    ]
                ).to(points_unscaled)

                points_d_ = (points_unscaled[..., None, :] + offsets).clamp(
                    -self.cfg.radius, self.cfg.radius
                )
                points_d = scale_anything(
                    points_d_, (-self.cfg.radius, self.cfg.radius), (0, 1)
                )
                points_d_sdf = (
                    self.network(self.encoding(points_d.view(-1, 3)))[..., 0]
                    .view(*points.shape[:-1], 6)
                    .float()
                )
                sdf_grad = (
                    0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps
                )
                normal = F.normalize(sdf_grad, p=2, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")

            output.update(
                {"normal": normal, "shading_normal": normal, "sdf_grad": sdf_grad}
            )
        return output

    def forward_sdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        out = self.network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], self.cfg.feature_dim)
        sdf, feature = out[..., 0], out
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        out = self.network(enc).reshape(
            *points.shape[:-1], self.cfg.feature_dim
        )  # feature_dim=13

        # from wonder3D
        sdf, feature = out[..., 0], out
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)
        return sdf, deformation

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

    def isosurface(self) -> Mesh:
        if not self.cfg.optim_mesh_via_deformation:
            return super().isosurface()
        else:
            if self.mesh is None:
                mesh = super().isosurface()
                self.isosurface_bbox = mesh.extras["bbox"]
                self.sdf.data = mesh.extras["grid_level"].to(mesh.v_pos)

                if len(self.sdf.data.shape) == 1:  # sdf should be [Nv, 1]
                    self.sdf.data = self.sdf.data[:, None]

                if self.cfg.isosurface_deformable_grid:
                    self.deformation.data = mesh.extras["grid_deformation"].to(
                        mesh.v_pos
                    )

            mesh = self.isosurface_helper(self.sdf, self.deformation)
            mesh.v_pos = scale_tensor(
                mesh.v_pos, self.isosurface_helper.points_range, self.isosurface_bbox
            )
            if self.cfg.isosurface_remove_outliers:
                mesh = mesh.remove_outlier(
                    self.cfg.isosurface_outlier_n_faces_threshold
                )
            self.mesh = mesh
            return mesh

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "ImplicitSDF":
        if isinstance(other, ImplicitSDF):
            instance = ImplicitSDF(cfg, **kwargs)
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.network.load_state_dict(
                other.network.state_dict()  # network in Wonder3D = feature_network in 3studio
            )
            if (
                instance.cfg.isosurface_deformable_grid
                and other.cfg.isosurface_deformable_grid
            ):
                assert (
                    instance.cfg.isosurface_method == "mt"
                ), "isosurface_deformable_grid only works with mt"
                instance.deformation_network.load_state_dict(
                    other.deformation_network.state_dict()
                )

            return instance
        else:
            raise TypeError(
                f"Cannot create {ImplicitSDF.__name__} from {other.__class__.__name__}"
            )

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        out = self(points, **kwargs)
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if "global_step_offset" in self.cfg.pos_encoding_config.keys():
            global_step += self.cfg.pos_encoding_config.global_step_offset
        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )
