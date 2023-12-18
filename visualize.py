from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from datasets import transforms as T
from datasets.wireframe import WireframeDataset
from models.fclip import encode_clam


def build_target(img: Tensor, ann: Dict[str, Tensor]):
    _, h1, w1 = img.shape
    points = ann["points"]
    lines = ann["lines"]

    points[:, 0] *= 128 / w1
    points[:, 1] *= 128 / h1
    lpos = points[lines].reshape(-1, 4)
    cloc, coff, llen, lang = encode_clam(lpos)

    tgt = {
        "cloc": cloc, "coff": coff,
        "llen": llen, "lang": lang,
    }
    return img, tgt


if __name__ == "__main__":
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize([512, 512]),
        build_target,
    ])
    dataset = WireframeDataset("data/wireframe", split="train", transforms=transforms)
    img, tgt = dataset[0]

    image_arr = img.permute(1, 2, 0).numpy()
    image_arr = (image_arr * 255).astype(np.uint8)
    image = Image.fromarray(image_arr)
    draw = ImageDraw.Draw(image)

    cloc: Tensor = tgt["cloc"]
    coff: Tensor = tgt["coff"]
    llen: Tensor = tgt["llen"]
    lang: Tensor = tgt["lang"]

    yi, xi = cloc.nonzero(as_tuple=True)
    centers = torch.stack((xi, yi)) + coff[:, yi, xi]
    radii = llen[yi, xi] * 64
    angles = lang[yi, xi] * torch.pi
    disp = torch.stack((angles.cos(), -angles.sin())) * radii
    lines = torch.cat((centers + disp, centers - disp))
    lines = lines.t() * 4
    for xy in lines.tolist():
        draw.line(xy, fill=(0, 255, 0))
    image.save("a.png")
