import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from datasets import transforms as T
from datasets.wireframe import WireframeDataset
from models.fclip import FClip, encode_clam


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


def l1_loss(input: Tensor, target: Tensor):
    assert input.shape == target.shape
    loss = F.l1_loss(input, target, reduction="none")
    if input.dim() == 4:
        loss = loss.mean(dim=1)
    return loss


def criterion(outputs: Dict[str, List[Tensor]], targets: Dict[str, Tensor]):
    loss_dict = defaultdict(float)

    b, y, x = targets["cloc"].nonzero(as_tuple=True)
    for heatmaps in outputs["heatmaps"]:
        cloc = heatmaps[:, 0:2]
        coff = heatmaps[:, 2:4].sigmoid()[:, [1, 0]]
        llen = heatmaps[:, 4].sigmoid()
        lang = heatmaps[:, 5].sigmoid()

        loss_dict["cloc"] += F.cross_entropy(cloc, targets["cloc"])

        loss_coff = l1_loss(coff, targets["coff"])
        loss_dict["coff"] += loss_coff[b, y, x].mean()

        loss_llen = l1_loss(llen, targets["llen"])
        loss_dict["llen"] += loss_llen[b, y, x].mean()

        loss_lang = l1_loss(lang, targets["lang"])
        loss_dict["lang"] += loss_lang[b, y, x].mean()

    loss_dict = dict(loss_dict)
    loss_weights = {
        "cloc": 1, "coff": 0.25,
        "llen": 3, "lang": 1,
    }
    for k, w in loss_weights.items():
        loss_dict[k] *= w
    return loss_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--logdir", default="logs/")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = T.Compose([
        T.ToTensor(),
        T.Resize([512, 512]),
        T.Normalize([.430, .407, .387], [.087, .087, .091]),
        build_target,
    ])
    dataset = WireframeDataset("data/wireframe", split="train", transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.batch_size)

    model = FClip()
    model.to(args.device)
    model.train()

    optimizer = Adam(model.parameters(), lr=4e-4, weight_decay=1e-4, amsgrad=True)
    scheduler = MultiStepLR(optimizer, milestones=[240, 280])

    logger = SummaryWriter(log_dir=args.logdir)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch:03}", dynamic_ncols=True):
            images = images.to(args.device)
            for k, v in targets.items():
                targets[k] = v.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            optimizer.step()
            logger.add_scalar("loss", total_loss, global_step)
            for k, v in loss_dict.items():
                logger.add_scalar(f"loss_dict/{k}", v, global_step)
            global_step += 1
        scheduler.step()

        if epoch % 10 == 0:
            checkpoint = {"model": model.state_dict()}
            torch.save(checkpoint, os.path.join(args.logdir, f"checkpoint{epoch:03}.pth"))
