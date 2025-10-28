from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, random, os


class VITONDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=(512, 384),
        train=True,  # True / False / None
        center_crop=False,
        instance_prompt="",
        scale=1.0,
        use_different=False,
    ):
        self.data_root = Path(data_root)
        self.train = train
        self.use_different = use_different
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.center_crop = center_crop
        self.size = size
        self.cond_size = [int(x // scale) for x in size]

        self.samples = []

        subsets = []
        if train is True:
            subsets = ["train"]
        elif train is False:
            subsets = ["test"]
        elif train is None:
            subsets = ["train", "test"]

        for subset in subsets:
            subset_root = self.data_root / subset
            image_dir = subset_root / "image"
            cloth_dir = subset_root / "cloth"

            if self.use_different:
                pairs_file = subset_root / (
                    "train_pairs.txt" if subset == "train" else "test_pairs.txt"
                )
                if not pairs_file.exists():
                    raise FileNotFoundError(f"Missing pairs file: {pairs_file}")
                with open(pairs_file, "r") as f:
                    for line in f:
                        person_name, cloth_name = line.strip().split()
                        self.samples.append((subset, person_name, cloth_name))
            else:
                names = sorted(
                    [
                        f.name
                        for f in image_dir.iterdir()
                        if f.suffix in [".jpg", ".png"]
                    ]
                )
                for name in names:
                    self.samples.append((subset, name, name))

        # transforms
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
        subset, person_name, cloth_name = self.samples[idx]
        subset_root = self.data_root / subset

        image_dir = subset_root / "image"
        image_ref_dir = subset_root / "image_ref"
        cloth_dir = subset_root / "cloth"
        person_dir = subset_root / "person"
        dense_dir = subset_root / "dense"

        agnostic_mask_dir = subset_root / "agnostic_mask"
        agnostic_dir = subset_root / "agnostic"

        reference_name = cloth_name

        # print(agnostic_mask_dir, agnostic_dir)
        example = {
            "image": self.load_image(image_dir, person_name, self.transform),
            "agnostic": self.load_image(agnostic_dir, person_name, self.transform),
            "cloth": self.load_image(cloth_dir, cloth_name, self.transform),
            "image_ref": self.load_image(
                image_ref_dir, reference_name, self.transform_cond
            ),
            "agnostic_mask": self.load_image(
                agnostic_mask_dir, person_name[:-4] + "_mask.png", self.transform_cond
            ),
            "dense": self.load_image(
                dense_dir, person_name[:-4] + ".png", self.transform_cond
            ),
            "instance_prompt": self.instance_prompt,
            "index": str(person_name),
        }
        if self.train and os.path.exists(person_dir):
            example["person"] = self.load_image(person_dir, person_name, self.transform)
        else:
            example["person"] = self.load_image(image_dir, person_name, self.transform)
        return example
