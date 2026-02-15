from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_2bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 4-bit precision along the last dimension.
    Always quantize group_size value together and store their absolute value first.
    To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
    Return the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
    normalization = 1.55 * sigma
    x_norm = (x.clamp(-normalization, normalization) + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 3).round().to(torch.int8)
    # x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    x_quant_2 = (x_quant_8[:, 0::4] & 0x3) + ((x_quant_8[:, 1::4] & 0x3) << 2) + ((x_quant_8[:, 2::4] & 0x3) << 4) + ((x_quant_8[:, 3::4] & 0x3) << 6)

    return x_quant_2, normalization.to(torch.float16)


def block_dequantize_2bit(x_quant_2: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """
    The reverse operation of block_quantize_4bit.
    """
    assert x_quant_2.dim() == 2

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_2.new_empty(x_quant_2.size(0), x_quant_2.shape[1] * 4)  
    x_quant_8[:, 0::4] = x_quant_2 & 0x3
    x_quant_8[:, 1::4] = (x_quant_2 >> 2) & 0x3
    x_quant_8[:, 2::4] = (x_quant_2 >> 4) & 0x3
    x_quant_8[:, 3::4] = (x_quant_2 >> 6) & 0x3
    x_norm = x_quant_8.to(torch.float32) / 3
    x = (x_norm * 2 * normalization) - normalization

    return x.view(-1)


def _pack_3bit(v: torch.Tensor) -> torch.Tensor:

    byte0 = (v[..., 0] & 0x7) | ((v[..., 1] & 0x7) << 3) | ((v[..., 2] & 0x3) << 6)
    byte1 = ((v[..., 2] >> 2) & 0x1) | ((v[..., 3] & 0x7) << 1) | ((v[..., 4] & 0x7) << 4) | ((v[..., 5] & 0x1) << 7)
    byte2 = ((v[..., 5] >> 1) & 0x3) | ((v[..., 6] & 0x7) << 2) | ((v[..., 7] & 0x7) << 5)
    return torch.stack([byte0, byte1, byte2], dim=-1)


def _unpack_3bit(p: torch.Tensor) -> torch.Tensor:

    val0 = p[..., 0] & 0x7
    val1 = (p[..., 0] >> 3) & 0x7
    val2 = ((p[..., 0] >> 6) & 0x3) | ((p[..., 1] & 0x1) << 2)
    val3 = (p[..., 1] >> 1) & 0x7
    val4 = (p[..., 1] >> 4) & 0x7
    val5 = ((p[..., 1] >> 7) & 0x1) | ((p[..., 2] & 0x3) << 1)
    val6 = (p[..., 2] >> 2) & 0x7
    val7 = (p[..., 2] >> 5) & 0x7
    return torch.stack([val0, val1, val2, val3, val4, val5, val6, val7], dim=-1)


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:

    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    x_norm = (x + normalization) / (2 * normalization)
    x_quant = (x_norm * 7).round().to(torch.int8)

    # Pack: reshape to (-1, group_size/8, 8) then pack each set of 8 into 3 bytes
    x_quant = x_quant.view(-1, group_size // 8, 8)
    packed = _pack_3bit(x_quant).view(-1, group_size * 3 // 8)

    return packed, normalization.to(torch.float16)


def block_dequantize_3bit(packed: torch.Tensor, normalization: torch.Tensor, group_size: int = 32) -> torch.Tensor:

    assert packed.dim() == 2

    normalization = normalization.to(torch.float32)
    # Unpack: reshape to (-1, group_size/8, 3) then unpack each 3 bytes into 8 values
    p = packed.view(-1, group_size // 8, 3)
    x_quant = _unpack_3bit(p).view(-1, group_size)

    x_norm = x_quant.to(torch.float32) / 7
    x = (x_norm * 2 * normalization) - normalization

    return x.view(-1)


class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        # Let's store all the required information to load the weights from a checkpoint
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # self.register_buffer is used to store the weights in the model, but not as parameters
        # This makes sure weights are put on the correct device when calling `model.to(device)`.
        # persistent=False makes sure the buffer is not saved or loaded. The bignet has a parameters
        # called "weight" that we need to quantize when the model is loaded.
        self.register_buffer(
            "weight_q2",
            torch.zeros(out_features * in_features // group_size, group_size // 4, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        # Register a hook to load the weights from a checkpoint. This function reaches deep into
        # PyTorch internals. It makes sure that Linear4Bit._load_state_dict_pre_hook is called
        # every time the model is loaded from a checkpoint. We will quantize the weights in that function.
        self._register_load_state_dict_pre_hook(Linear2Bit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            # Load the original weights and remove them from the state_dict (mark them as loaded)
            weight = state_dict[f"{prefix}weight"]  # noqa: F841
            del state_dict[f"{prefix}weight"]
            # TODO: Quantize the weights and store them in self.weight_q4 and self.weight_norm
            
            x_quant2bit, norm = block_quantize_2bit(weight.flatten(), self._group_size)
            self.weight_q2.copy_(x_quant2bit)
            self.weight_norm.copy_(norm)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # TODO: Dequantize and call the layer
            # Hint: You can use torch.nn.functional.linear
            # raise NotImplementedError()
            weight = block_dequantize_2bit(self.weight_q2, self.weight_norm).view(self._shape)
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet2Bit(torch.nn.Module):
    """
    A BigNet where all weights are in 4bit precision. Use the Linear4Bit module for this.
    It is fine to keep all computation in float32.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        n_groups = out_features * in_features // group_size
        packed_cols = group_size * 3 // 8  # 3 bytes per 8 values

        self.register_buffer(
            "weight_q3",
            torch.zeros(n_groups, packed_cols, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(n_groups, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            packed, norm = block_quantize_3bit(weight.flatten(), self._group_size)
            self.weight_q3.copy_(packed)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_3bit(self.weight_q3, self.weight_norm, self._group_size).view(self._shape)
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet3Bit(torch.nn.Module):
    """
    A BigNet where all weights are in 3-bit precision. Use the Linear3Bit module for this.
    Uses 3-bit packing (8 values -> 3 bytes) and group_size=32 to fit under 9 MB.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet3Bit:
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
