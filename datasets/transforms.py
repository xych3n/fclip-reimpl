from typing import Any, Callable, Iterable, List, Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms: Iterable[Callable[..., Any]]):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor:
    def __call__(self, img, ann):
        img = F.to_tensor(img)
        if ann is not None:
            keys = tuple(ann.keys())
            for k in keys:
                try:
                    ann[k] = torch.as_tensor(ann[k])
                except TypeError:
                    pass
        return img, ann


class Resize:
    def __init__(self, size: int | Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img, ann = None, /):
        w1, h1 = F.get_image_size(img)
        img = F.resize(img, self.size, antialias=True)
        w2, h2 = F.get_image_size(img)
        if ann is not None:
            ann["points"][:, 0] *= w2 / w1
            ann["points"][:, 1] *= h2 / h1
        return img, ann


class Normalize:
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, img_tensor: Tensor, *args):
        img_tensor = F.normalize(img_tensor, self.mean, self.std)
        return img_tensor, *args


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Tensor, ann):
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            w, _ = F.get_image_size(img)
            ann["points"][:, 0] = w - ann["points"][:, 0]
        return img, ann


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Tensor, ann):
        if torch.rand(1) < self.p:
            img = F.vflip(img)
            _, h = F.get_image_size(img)
            ann["points"][:, 1] = h - ann["points"][:, 1]
        return img, ann
