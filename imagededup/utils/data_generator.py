import os
from pathlib import PurePath
from typing import Callable, Dict, List, Optional, Tuple

import clip
import numpy as np
import timm
import torch
import torch.nn as nn
from clip.clip import _MODELS, _download
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from imagededup.utils.general_utils import generate_files
from imagededup.utils.image_utils import load_image
from torchvision import transforms

class ImgDataset(Dataset):
    def __init__(
        self,
        image_dir: PurePath,
        basenet_preprocess: Callable[[np.array], torch.tensor],
        recursive: Optional[bool],
    ) -> None:
        self.image_dir = image_dir
        self.basenet_preprocess = basenet_preprocess
        self.recursive = recursive
        self.image_files = sorted(
            generate_files(self.image_dir, self.recursive)
        )  # ignore hidden files

    def __len__(self) -> int:
        """Number of images."""
        return len(self.image_files)

    def __getitem__(self, item) -> Dict:
        im_arr = load_image(self.image_files[item], target_size=None, grayscale=None)
        if im_arr is not None:
            img = self.basenet_preprocess(im_arr)
            return {"image": img, "filename": self.image_files[item]}
        else:
            return {"image": None, "filename": self.image_files[item]}


def _collate_fn(batch: List[Dict]) -> Tuple[torch.tensor, str, str]:
    ims, filenames, bad_images = [], [], []

    for b in batch:
        im = b["image"]
        if im is not None:
            ims.append(im)
            filenames.append(b["filename"])
        else:
            bad_images.append(b["filename"])
    return torch.stack(ims), filenames, bad_images


def img_dataloader(
    image_dir: PurePath,
    batch_size: int,
    basenet_preprocess: Callable[[np.array], torch.tensor],
    recursive: Optional[bool],
) -> DataLoader:
    img_dataset = ImgDataset(
        image_dir=image_dir, basenet_preprocess=basenet_preprocess, recursive=recursive
    )
    return DataLoader(
        dataset=img_dataset, batch_size=batch_size, collate_fn=_collate_fn
    )


class CNNModel(torch.nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.name == "mobilenet_v3_small":
            mobilenet = (
                models.mobilenet_v3_small(pretrained=True).eval().to(self.device)
            )
            self.mobilenet_gap_op = torch.nn.Sequential(
                mobilenet.features, mobilenet.avgpool
            )
        elif self.name == "ViT-L/14@336px":
            self.clip_norm = nn.Sequential(
                transforms.Resize(size=[336, 336]),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            )
            model_path = _download(
                _MODELS[self.name], os.path.expanduser("~/.cache/clip")
            )
            with open(model_path, "rb") as opened_file:
                self.clip_vit = torch.jit.load(opened_file).visual.eval().to(self.device)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            self.timm_model = (
                timm.create_model(self.name, pretrained=True).eval().cuda()
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x) -> torch.tensor:
        if self.name == "mobilenet_v3_small":
            return self.mobilenet_gap_op(x.to(self.device)).detach().cpu()
        elif self.name == "ViT-L/14@336px":
            x = self.clip_norm(x / 255.0)
            return self.avgpool(self.clip_vit(x.half().to(self.device))).detach().cpu()
        else:
            return (
                self.avgpool(self.timm_model.forward_features(x.to(self.device)))
                .detach()
                .cpu()
            )
