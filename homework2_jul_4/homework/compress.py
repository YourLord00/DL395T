from pathlib import Path
from typing import cast

import struct
import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer

PRECISION = 32
FULL = 1 << PRECISION
HALF = FULL >> 1
QUARTER = FULL >> 2

def ae_encode(symbols, probs_list):
    low = 0
    high = FULL
    pending = 0
    bits = []

    for idx, sym in enumerate(symbols):
        cdf = _probs_to_cdf(probs_list[idx])
        total = cdf[-1]
        rng = high - low

        high = low + (rng * cdf[sym + 1]) // total
        low = low + (rng * cdf[sym]) // total

        while True:
            if high <= HALF:
                bits.append(0)
                bits.extend([1] * pending)
                pending = 0
                high <<= 1
                low <<= 1
            elif low >= HALF:
                bits.append(1)
                bits.extend([0] * pending)
                pending = 0
                high = (high - HALF) << 1
                low = (low - HALF) << 1
            elif low >= QUARTER and high <= 3 * QUARTER:
                high = (high - QUARTER) << 1
                low = (low - QUARTER) << 1
                pending += 1
            else:
                break

    pending += 1
    
    if low < QUARTER:
        bits.append(0)
        bits.extend([1] * pending)
    else:
        bits.append(1)
        bits.extend([0] * pending)

    return bits


def _probs_to_cdf(probs):
    total = 1 << 16  # 65536
    freqs = (probs * total).long().clamp(min=1)
    cdf = torch.zeros(len(probs) + 1, dtype=torch.long)
    cdf[1:] = torch.cumsum(freqs, dim=0)
    return cdf.tolist()

def ae_decode_symbol(state, probs):
    low = state["low"]
    high = state["high"]
    value = state["value"]
    bit_idx = state["bit_idx"]
    bits = state["bits"]

    cdf = _probs_to_cdf(probs)
    total = cdf[-1]

    rng = high - low
    scaled = ((value - low + 1) * total - 1) // rng

    sym = 0
    while cdf[sym + 1] <= scaled:
        sym += 1

    high = low + (rng * cdf[sym + 1]) // total
    low = low + (rng * cdf[sym]) // total

    while True:
        if high <= HALF:
            low <<= 1
            high <<= 1
            value = (value << 1) | (bits[bit_idx] if bit_idx < len(bits) else 0)
            bit_idx += 1
        elif low >= HALF:
            low = (low - HALF) << 1
            high = (high - HALF) << 1
            value = ((value - HALF) << 1) | (bits[bit_idx] if bit_idx < len(bits) else 0)
            bit_idx += 1
        elif low >= QUARTER and high <= 3 * QUARTER:
            low = (low - QUARTER) << 1
            high = (high - QUARTER) << 1
            value = ((value - QUARTER) << 1) | (bits[bit_idx] if bit_idx < len(bits) else 0)
            bit_idx += 1
        else:
            break

    state["low"] = low
    state["high"] = high
    state["value"] = value
    state["bit_idx"] = bit_idx

    return sym


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        # raise NotImplementedError()
        tokens = self.tokenizer.encode_index(x)
        tokens = tokens.unsqueeze(0)
        #prob distribution
        logits, _ = self.autoregressive(tokens)
        probs = torch.softmax(logits, dim=-1)[0]  # (20, 30, 1024)

        tokens_flattend = tokens[0].reshape(-1).cpu().tolist()
        probs_flattend = probs.reshape(-1,probs.shape[-1]).cpu()

        bits = ae_encode(tokens_flattend, probs_flattend)

        h, w = tokens[0].shape
        num_bits = len(bits)
        header = struct.pack(">HHI", h, w,num_bits)

        padded = bits + [0] * (-len(bits) % 8)
        byte_data = int(''.join(map(str, padded)), 2).to_bytes(len(padded) // 8, 'big')

        return header + bytes(byte_data)

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        h, w, num_bits = struct.unpack(">HHI", x[:8])
        byte_data = x[8:]

        # byte to bits
        bit_str = bin(int.from_bytes(byte_data, 'big'))[2:].zfill(len(byte_data) * 8)
        bits = [int(b) for b in bit_str[:num_bits]]

        device = next(self.autoregressive.parameters()).device
        value = 0
        
        for i in range( PRECISION):
            value = (value << 1) | (bits[i] if i < len(bits) else 0)
        state = {"low": 0, "high": FULL, "value": value, "bit_idx": PRECISION, "bits": bits}

        tokens = torch.zeros(1, h, w, dtype=torch.long, device=device)
        for pos in range(h * w):
            i, j = pos //  w, pos % w

            with torch.inference_mode():
                logits, _ = self.autoregressive.forward(tokens)
                probs = torch.softmax(logits[0, i, j], dim=-1).cpu()

            symol = ae_decode_symbol(state, probs)
            tokens[0, i, j] = symol

        with torch.inference_mode():
            image = self.tokenizer.decode_index(tokens[0])

        return image


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
