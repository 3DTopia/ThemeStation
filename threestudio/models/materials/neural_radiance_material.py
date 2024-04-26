import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("neural-radiance-material")
class NeuralRadianceMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        input_feature_dims: int = 8
        color_activation: str = "sigmoid"
        dir_encoding_config: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 3}
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        )
        requires_normal: bool = False
        with_viewdir: bool = False

    cfg: Config

    def configure(self) -> None:
        self.with_viewdir=self.cfg.with_viewdir
        if self.with_viewdir:
            self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
            self.n_input_dims = self.cfg.input_feature_dims + self.encoding.n_output_dims  # type: ignore
        else:
            self.encoding = None
            self.n_input_dims = self.cfg.input_feature_dims
        
        self.network = get_mlp(self.n_input_dims, 3, self.cfg.mlp_network_config)
        self.requires_normal=self.cfg.requires_normal

    def forward(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        **kwargs,
    ) -> Float[Tensor, "*B 3"]:
        if self.with_viewdir:
            # viewdirs and normals must be normalized before passing to this function
            viewdirs = (viewdirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
            viewdirs_embd = self.encoding(viewdirs.view(-1, 3))

            network_inp = torch.cat(
                [features.view(-1, features.shape[-1]), viewdirs_embd] + [v.view(-1, v.shape[-1]) for k, v in kwargs.items()], dim=-1
            )

            color = self.network(network_inp).view(*features.shape[:-1], 3)

        else:
            network_inp = torch.cat(
                [features.view(-1, features.shape[-1])] + [v.view(-1, v.shape[-1]) for k, v in kwargs.items()], dim=-1
            )

            color = self.network(network_inp).view(*features.shape[:-1], 3)

        color = get_activation(self.cfg.color_activation)(color)

        return color
    
    # added
    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseMaterial,
        cfg: Optional[Union[dict, DictConfig]] = None,
        **kwargs,
    ) -> "NeuralRadianceMaterial":
        if isinstance(other, NeuralRadianceMaterial):
            instance = NeuralRadianceMaterial(cfg, **kwargs)
            
            if instance.with_viewdir:
                instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.network.load_state_dict(
                other.network.state_dict()
            )

            return instance
        else:
            raise TypeError(
                f"Cannot create {NeuralRadianceMaterial.__name__} from {other.__class__.__name__}"
            )
        
    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        color = self(features, viewdirs=-kwargs['normal'], normal=kwargs['normal']).clamp(0, 1)
        assert color.shape[-1] >= 3, "Output color must have at least 3 channels"
        if color.shape[-1] > 3:
            threestudio.warn(
                "Output color has >3 channels, treating the first 3 as RGB"
            )
        return {"albedo": color[..., :3]}