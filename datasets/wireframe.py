import contextlib
import copy
import os
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F


class WireframeDataset(VisionDataset):
    def __init__(self, root: str, split: Literal["train", "test"],
                 transforms: Optional[Callable] = None):
        super().__init__(root, transforms)
        self.split = split
        with contextlib.redirect_stdout(None):
            self.coco = COCO(self.annotation_file)
        self.ids = sorted(self.coco.imgs.keys())

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.image_dir, path)).convert("RGB")

    def _load_annotation(self, id: int) -> Dict[str, Any]:
        ann = self.coco.loadAnns(self.coco.getAnnIds(id))[0]
        ann = copy.deepcopy(ann)
        ann["points"] = np.array(ann["points"], dtype=np.float32)
        ann["lines"] = np.array(ann["lines"], dtype=int)
        if self.split == "test":
            ann.update({"file_name": self.coco.loadImgs(id)[0]["file_name"]})
        return ann

    def __getitem__(self, index):
        if self.split == "train":
            index, t = divmod(index, 4)
        id = self.ids[index]
        img = self._load_image(id)
        ann = self._load_annotation(id)

        if self.split == "train":
            w, h = F.get_image_size(img)
            if t == 1:
                img = F.hflip(img)
                ann["points"][:, 0] = w - ann["points"][:, 0]
            elif t == 2:
                img = F.vflip(img)
                ann["points"][:, 1] = h - ann["points"][:, 1]
            elif t == 3:
                img = F.hflip(img)
                img = F.vflip(img)
                ann["points"][:, 0] = w - ann["points"][:, 0]
                ann["points"][:, 1] = h - ann["points"][:, 1]

        if self.transforms is not None:
            img, ann = self.transforms(img, ann)

        return img, ann

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.ids) * 4
        else:
            return len(self.ids)

    @property
    def image_dir(self) -> str:
        return os.path.join(self.root, "images", self.split)

    @property
    def annotation_file(self) -> str:
        return os.path.join(self.root, self.split + ".json")
