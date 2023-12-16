from functools import partial
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils import clip_lines
from .backbone.hourglass import Bottleneck, StackedHourglass


def encode_clam(lines: Tensor) -> Tuple[Tensor]:
    """represent a line segment by its center, length, and angle"""
    lines = torch.tensor(clip_lines(lines)).reshape(-1, 2, 2)
    p1, p2 = torch.unbind(lines, dim=1)
    center = (p1 + p2) / 2
    x, y = torch.unbind(center, dim=1)
    centeri = center.clamp(0, 128 - 1e-4).floor().long()
    xi, yi = torch.unbind(centeri, dim=1)
    cloc = torch.zeros((128, 128), dtype=torch.int64)
    cloc[yi, xi] = 1
    coff = torch.zeros((2, 128, 128), dtype=torch.float32)
    coff[:, yi, xi] = torch.stack((x - xi, y - yi))
    lines_vec = p2 - p1
    disp = torch.norm(lines_vec) / 2
    llen = torch.zeros((128, 128), dtype=torch.float32)
    llen[yi, xi] = disp.clamp(0, 64) / 64
    lang = torch.zeros((128, 128), dtype=torch.float32)
    lang[yi, xi] = (torch.atan2(lines_vec[:, 1], lines_vec[:, 0]) % torch.pi) / torch.pi
    return cloc, coff, llen, lang


class LineBlock(nn.Module):
    def __init__(self, planes: int) -> None:
        super().__init__()

        self.conv_h = nn.Conv2d(planes, planes, kernel_size=(7, 1), padding=(3, 0))
        self.conv_v = nn.Conv2d(planes, planes, kernel_size=(1, 7), padding=(0, 3))

    def forward(self, x: Tensor):
        x = torch.maximum(self.conv_h(x), self.conv_v(x))
        return x


class BottleneckLine(Bottleneck):
    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__(inplanes, planes, stride)

        self.conv2 = LineBlock(planes)


class StackedHourglassLine(StackedHourglass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for hourglass in self.hourglasses:
            hourglass += nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )


class MultitaskHead(nn.Module):
    def __init__(self, in_channels: int, task_sizes: Sequence[int]) -> None:
        super().__init__()

        inter_channels = in_channels // 4
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            ) for out_channels in task_sizes
        ])

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=1)


def build_backbone(task_sizes: Sequence[int]):
    return StackedHourglassLine(
        BottleneckLine,
        partial(MultitaskHead, task_sizes=task_sizes),
        num_classes=sum(task_sizes),
    )


class FClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone = build_backbone([2, 2, 1, 1])

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        _, heatmaps_list = self.backbone(images)
        if self.training:
            return {"heatmaps": heatmaps_list}

        heatmaps = heatmaps_list[0]
        cloc, = heatmaps[:, 0:2].softmax(dim=1)[:, [1]]
        coff, = heatmaps[:, 2:4].sigmoid()[:, [1, 0]]   # assure [dx, dy] format
        llen = heatmaps[:, 4:5].sigmoid()
        lang = heatmaps[:, 5:6].sigmoid()

        centers, scores, indices = propose_junctions(cloc, coff,
            k=1000, soft=0.8, return_indices=True)
        radii = llen.flatten()[indices] * 64
        angles = lang.flatten()[indices] * torch.pi
        displs = torch.stack((angles.cos(), -angles.sin().abs())) * radii
        lines = torch.cat((centers + displs.t(), centers - displs.t()), dim=1)
        lines, scores = structrual_nms(lines, scores)
        return lines, scores


def nms(x: Tensor, kernel_size: int = 3, soft: float = 0) -> Tensor:
    mask = x == F.max_pool2d(x, kernel_size=kernel_size,
                             stride=1, padding=kernel_size//2)
    return x * (mask + ~mask * soft)


def propose_junctions(jloc: Tensor, joff: Tensor, k: int = 0, threshold: float = 0,
                      kernel_size: int = 3, soft: float = 0, return_indices: bool = False):
    assert jloc.ndim == 3 and jloc.size(0) == 1
    assert joff.ndim == 3 and joff.size(0) == 2
    _, H, W = jloc.shape
    jloc = nms(jloc, kernel_size, soft)
    jloc = jloc.flatten()
    joff = joff.flatten(start_dim=1)
    if k > 0:
        scores, indices = torch.topk(jloc, k)
    else:
        indices = (jloc > threshold).nonzero()
        scores = jloc[indices]
    y = indices // W + joff[1, indices]
    x = indices % W + joff[0, indices]
    coords = torch.stack((x, y), dim=-1)
    coords = coords[scores > threshold]
    if return_indices:
        return coords, scores, indices
    return coords, scores


def structrual_nms(lines: Tensor, scores: Tensor, threshold: float = 2):
    lines = lines.reshape(-1, 2, 2)
    euid = lambda x, y: ((x - y) ** 2).sum(axis=-1)
    dist = torch.minimum(
        euid(lines[:, None,  0], lines[None, :, 0]) + euid(lines[:, None, 1], lines[None, :, 1]),
        euid(lines[:, None,  1], lines[None, :, 0]) + euid(lines[:, None, 0], lines[None, :, 1])
    )
    indices = dist <= threshold
    diagonal = torch.eye(len(lines), dtype=bool, device=lines.device)
    indices[diagonal] = False
    drop = indices[0]
    for i in range(1, len(lines) - 2):
        if drop[i]:
            continue
        drop[i+1:] |= indices[i, i+1:]
    lines, scores = lines[~drop], scores[~drop]
    lines = lines.reshape(-1, 4)
    return lines, scores
