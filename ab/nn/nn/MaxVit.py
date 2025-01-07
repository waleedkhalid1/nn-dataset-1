import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth


def _get_conv_output_shape(input_size: Tuple[int, int], kernel_size: int, stride: int, padding: int) -> Tuple[int, int]:
    return (
        (input_size[0] - kernel_size + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size + 2 * padding) // stride + 1,
    )


def _make_block_input_shapes(input_size: Tuple[int, int], n_blocks: int) -> List[Tuple[int, int]]:
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes


def _get_relative_position_index(height: int, width: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(height), torch.arange(width)]))
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        squeeze_ratio: float,
        stride: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        p_stochastic_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        proj: Sequence[nn.Module]
        self.proj: nn.Module

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")
        else:
            self.stochastic_depth = nn.Identity()

        _layers = OrderedDict()
        _layers["pre_norm"] = norm_layer(in_channels)
        _layers["conv_a"] = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            inplace=None,
        )
        _layers["conv_b"] = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            groups=mid_channels,
            inplace=None,
        )
        _layers["squeeze_excitation"] = SqueezeExcitation(mid_channels, sqz_channels, activation=nn.SiLU)
        _layers["conv_c"] = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, bias=True)

        self.layers = nn.Sequential(_layers)

    def forward(self, x: Tensor) -> Tensor:
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x


class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}")

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim**-0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(((2 * self.size - 1) * (2 * self.size - 1), self.n_heads), dtype=torch.float32),
        )

        self.register_buffer("relative_position_index", _get_relative_position_index(self.size, self.size))
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor:
        bias_index = self.relative_position_index.view(-1)
        relative_bias = self.relative_position_bias_table[bias_index].view(self.max_seq_len, self.max_seq_len, -1)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = F.softmax(dot_prod + pos_bias, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class SwapAxes(nn.Module):
    def __init__(self, a: int, b: int) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.swapaxes(x, self.a, self.b)
        return res


class WindowPartition(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int) -> Tensor:
        B, C, H, W = x.shape
        P = p
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, (H // P) * (W // P), P * P, C)
        return x


class WindowDepartition(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int, h_partitions: int, w_partitions: int) -> Tensor:
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        x = x.reshape(B, HP, WP, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, HP * P, WP * P)
        return x


class PartitionAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        partition_size: int,
        partition_type: str,
        grid_size: Tuple[int, int],
        mlp_ratio: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        attention_dropout: float,
        mlp_dropout: float,
        p_stochastic_dropout: float,
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type not in ["grid", "window"]:
            raise ValueError("partition_type must be either 'grid' or 'window'")

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        self.departition_swap = SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()

        self.attn_layer = nn.Sequential(
            norm_layer(in_channels),
            RelativePositionalMultiHeadAttention(in_channels, head_dim, partition_size**2),
            nn.Dropout(attention_dropout),
        )

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout),
        )

        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")

    def forward(self, x: Tensor) -> Tensor:
        gh, gw = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        torch._assert(
            self.grid_size[0] % self.p == 0 and self.grid_size[1] % self.p == 0,
            "Grid size must be divisible by partition size. Got grid size of {} and partition size of {}".format(
                self.grid_size, self.p
            ),
        )

        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))
        x = self.departition_swap(x)
        x = self.departition_op(x, self.p, gh, gw)

        return x


class MaxVitLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        p_stochastic_dropout: float,
        partition_size: int,
        grid_size: Tuple[int, int],
    ) -> None:
        super().__init__()

        layers: OrderedDict = OrderedDict()

        layers["MBconv"] = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=stride,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        layers["window_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        layers["grid_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class MaxVitBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        partition_size: int,
        input_grid_size: Tuple[int, int],
        n_layers: int,
        p_stochastic: List[float],
    ) -> None:
        super().__init__()
        if not len(p_stochastic) == n_layers:
            raise ValueError(f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}.")

        self.layers = nn.ModuleList()
        self.grid_size = _get_conv_output_shape(input_grid_size, kernel_size=3, stride=2, padding=1)

        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            self.layers += [
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p,
                ),
            ]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

args = [
    (299, 299),
    64,
    1,
    [64, 128, 256, 512],
    [2, 2, 5, 2],
    32,
    0.2,
]


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout', 'attention_dropout', 'stochastic_depth_prob'}


class Net(nn.Module):

    def train_setup(self, device, prm):
        self.device = device
        self.criteria = (nn.CrossEntropyLoss().to(device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict) -> None:
        super().__init__()
        input_size: Tuple[int, int] = in_shape[2:]
        stem_channels: int = 64
        partition_size: int = 1
        block_channels = None
        block_layers = None

        head_dim: int = 32
        stochastic_depth_prob: float = prm['stochastic_depth_prob']
        norm_layer: Optional[Callable[..., nn.Module]] = None
        activation_layer: Callable[..., nn.Module] = nn.GELU
        squeeze_ratio: float = 0.25
        expansion_ratio: float = 4
        mlp_ratio: int = 4
        mlp_dropout: float = prm['dropout']
        attention_dropout: float = prm['attention_dropout']
        num_classes: int = out_shape[0]
        if block_layers is None:
            block_layers = [2, 2, 5, 2]
        if block_channels is None:
            block_channels = [64, 128, 256, 512]
        input_channels = in_shape[1]

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))
        for idx, block_input_size in enumerate(block_input_sizes):
            if block_input_size[0] % partition_size != 0 or block_input_size[1] % partition_size != 0:
                raise ValueError(
                    f"Input size {block_input_size} of block {idx} is not divisible by partition size {partition_size}. "
                    f"Consider changing the partition size or the input size.\n"
                    f"Current configuration yields the following block input sizes: {block_input_sizes}."
                )

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                input_channels,
                stem_channels,
                3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                bias=False,
                inplace=None,
            ),
            Conv2dNormActivation(
                stem_channels, stem_channels, 3, stride=1, norm_layer=None, activation_layer=None, bias=True
            ),
        )

        input_size = _get_conv_output_shape(input_size, kernel_size=3, stride=2, padding=1)
        self.partition_size = partition_size

        self.blocks = nn.ModuleList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels

        p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()

        p_idx = 0
        for in_channel, out_channel, num_layers in zip(in_channels, out_channels, block_layers):
            self.blocks.append(
                MaxVitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_grid_size=input_size,
                    n_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx : p_idx + num_layers],
                ),
            )
            input_size = self.blocks[-1].grid_size
            p_idx += num_layers

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], num_classes, bias=False),
        )

        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
