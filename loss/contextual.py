import torch
import torch.nn as nn
from loss.vgg import VGG19
import torch.nn.functional as F

LOSS_TYPES = ["cosine"]


def contextual_loss(x=torch.Tensor, y=torch.Tensor, band_width=0.5, loss_type="cosine"):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """
    # print('band_width:',band_width)
    # assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f"select a loss type from {LOSS_TYPES}."

    N, C, H, W = x.size()

    if loss_type == "cosine":
        dist_raw = compute_cosine_distance(x, y)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)

    r_m = torch.max(cx, dim=1, keepdim=True)
    c = torch.gather(torch.exp((1 - dist_raw) / 0.5), 1, r_m[1])
    cx = torch.sum(torch.squeeze(r_m[0] * c, 1), dim=1) / torch.sum(
        torch.squeeze(c, 1), dim=1
    )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.
    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(
        self, band_width=0.5, loss_type="cosine", use_vgg=True, vgg_layer="relu3_4"
    ):
        super(ContextualLoss, self).__init__()

        self.band_width = band_width

        if use_vgg:
            print("use_vgg:", use_vgg)
            self.vgg_model = VGG19()
            self.vgg_layer = vgg_layer
            self.register_buffer(
                name="vgg_mean",
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False
                ),
            )
            self.register_buffer(
                name="vgg_std",
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False
                ),
            )

    def forward(self, x, y):
        if hasattr(self, "vgg_model"):
            assert (
                x.shape[1] == 3 and y.shape[1] == 3
            ), "VGG model takes 3 chennel images."
            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            x = getattr(self.vgg_model(x), self.vgg_layer)
            y = getattr(self.vgg_model(y), self.vgg_layer)

        return contextual_loss(x, y, self.band_width)
