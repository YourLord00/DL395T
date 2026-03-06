from pathlib import Path

import torch
from PIL import Image

DATASET_PATH = Path(__file__).parent.parent / "data"


class ImageDataset:
    image_paths: list[Path]
    _image_cache: list[torch.Tensor | None]
    _cache_images: bool

    def __init__(self, split: str, cache_images: bool = True):
        self.image_paths = list((DATASET_PATH / split).rglob("*.jpg"))
        self._image_cache = [None] * len(self.image_paths)
        self._cache_images = cache_images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import numpy as np

        cached_image = self._image_cache[idx]
        if cached_image is not None:
            return cached_image

        img = torch.tensor(np.array(Image.open(self.image_paths[idx])), dtype=torch.uint8)
        if self._cache_images:
            self._image_cache[idx] = img
        return img


class TokenDataset(torch.utils.data.TensorDataset):
    def __init__(self, split: str):
        tensor_path = DATASET_PATH / f"tokenized_{split}.pth"
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found at {tensor_path}. Create it following the assignment instructions."
            )
        self.data = torch.load(tensor_path, weights_only=False)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    # Debug ImageDataset Testing
    print("=== ImageDataset ===")
    print(f"DATASET_PATH: {DATASET_PATH}")
    print(f"DATASET_PATH exists: {DATASET_PATH.exists()}")

    ds = ImageDataset("train")
    print(f"Number of images: {len(ds)}")
    print(f"First 5 paths: {ds.image_paths[:5]}")

    img = ds[0]
    print(f"Image type: {type(img)}")
    print(f"Image shape: {img.shape}")  # (H, W, C)
    print(f"Image dtype: {img.dtype}")
    print(f"Min: {img.min()}, Max: {img.max()}")
    print(f"First pixel: {img[0, 0]}")

    # Debug TokenDataset Testing
    print("\n=== TokenDataset ===")
    try:
        tds = TokenDataset("train")
        print(f"Number of samples: {len(tds)}")
        print(f"Raw data type: {type(tds.data)}")
        print(f"Raw data shape: {tds.data.shape}")
        print(f"Raw data dtype: {tds.data.dtype}")

        token = tds[0]  # __getitem__
        print(f"Token type: {type(token)}")
        print(f"Token shape: {token.shape}")  # (h, w) token grid per image
        print(f"Token dtype: {token.dtype}")
        print(f"Min: {token.min()}, Max: {token.max()}")
        print(f"First 10 tokens: {token.flatten()[:10]}")
    except FileNotFoundError as e:
        print(f"TokenDataset not available: {e}")
