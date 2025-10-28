from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, random, os


class DressCodeDataset(Dataset):
    def __init__(
        self,
        data_root,
        train=True,  # True / False / None
        size=(512, 384),
        center_crop=False,
        instance_prompt="",
        scale=1.0,
        use_different=False,  # 支持和 VITONDataset 一样的 flag
    ):
        self.data_root = Path(data_root)
        self.categories = ("upper_body", "lower_body", "dresses")

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.cond_size = [int(x // scale) for x in size]
        self.center_crop = center_crop
        self.size = size
        self.train = train
        self.use_different = use_different

        self.samples = []

        if not self.use_different:
            # 原始逻辑：读取配对文件
            if train is True:
                pair_file = "train_pairs.txt"
            elif train is False:
                pair_file = "test_pairs_paired.txt"
            elif train is None:
                pair_files = ["train_pairs.txt", "test_pairs_paired.txt"]
            else:
                raise ValueError("train must be True/False/None")

            if train is None:
                for pf in pair_files:
                    pair_path = self.data_root / pf
                    if not pair_path.exists():
                        raise FileNotFoundError(f"Pair file not found: {pair_path}")
                    with open(pair_path, "r") as f:
                        for line in f:
                            img1, img2, subset_id = line.strip().split()
                            subset_id = int(subset_id)
                            if subset_id < 0 or subset_id >= len(self.categories):
                                raise ValueError(
                                    f"Invalid subset id {subset_id} in {pf}"
                                )
                            category = self.categories[subset_id]
                            self.samples.append((category, img1, img2))
            else:
                pair_path = self.data_root / pair_file
                if not pair_path.exists():
                    raise FileNotFoundError(f"Pair file not found: {pair_path}")
                with open(pair_path, "r") as f:
                    for line in f:
                        img1, img2, subset_id = line.strip().split()
                        subset_id = int(subset_id)
                        if subset_id < 0 or subset_id >= len(self.categories):
                            raise ValueError(
                                f"Invalid subset id {subset_id} in {pair_file}"
                            )
                        category = self.categories[subset_id]
                        self.samples.append((category, img1, img2))

        else:
            # 新逻辑：读取非配对文件
            if train is True:
                pair_files = ["train_pairs_unpaired.txt"]
            elif train is False:
                pair_files = ["test_pairs_unpaired.txt"]
            elif train is None:
                pair_files = ["train_pairs_unpaired.txt", "test_pairs_unpaired.txt"]
            else:
                raise ValueError("train must be True/False/None")

            for pf in pair_files:
                pair_path = self.data_root / pf
                if not pair_path.exists():
                    raise FileNotFoundError(f"Pair file not found: {pair_path}")
                with open(pair_path, "r") as f:
                    for line in f:
                        img1, img2, subset_id = line.strip().split()
                        subset_id = int(subset_id)
                        if subset_id < 0 or subset_id >= len(self.categories):
                            raise ValueError(f"Invalid subset id {subset_id} in {pf}")
                        category = self.categories[subset_id]
                        self.samples.append((category, img1, img2))

        # 定义 transform
        crop_op = (
            transforms.CenterCrop(self.size)
            if center_crop
            else transforms.RandomCrop(self.size)
        )
        crop_cond_op = (
            transforms.CenterCrop(self.cond_size)
            if center_crop
            else transforms.RandomCrop(self.cond_size)
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                crop_op,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.transform_cond = transforms.Compose(
            [
                transforms.Resize(
                    self.cond_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                crop_cond_op,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def load_image(self, directory, name, transform):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return transform(img)

    def __getitem__(self, idx):
        category, name1, name2 = self.samples[idx]
        base_dir = self.data_root / category

        image_dir = base_dir / "images"
        image_ref_dir = base_dir / "image_ref"
        dense_dir = base_dir / "dense"
        person_dir = base_dir / "person"

        agnostic_dir = base_dir / "agnostic_catvton"
        agnostic_mask_dir = base_dir / "agnostic_masks_catvton"
        
        example = {
            "image": self.load_image(image_dir, name1, self.transform),
            "agnostic": self.load_image(
                agnostic_dir, name1.replace("_0.jpg", "_6.jpg"), self.transform
            ),
            "cloth": self.load_image(image_dir, name2, self.transform),
            "image_ref": self.load_image(
                image_ref_dir, name2[:-5] + "0.jpg", self.transform_cond
            ),
            "agnostic_mask": self.load_image(
                agnostic_mask_dir,
                name1.replace("_0.jpg", "_3.png"),
                self.transform_cond,
            ),
            "dense": self.load_image(
                dense_dir, name1.replace("_0.jpg", "_5.png"), self.transform_cond
            ),
            "instance_prompt": self.instance_prompt,
            "category": category,
            "index": str(name1),
        }
        if self.train and os.path.exists(person_dir):
            example["person"] = self.load_image(person_dir, name1, self.transform)
        else:
            example["person"] = self.load_image(image_dir, name1, self.transform)
        return example
