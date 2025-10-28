from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, random, os


def find_model_image(dir_path, cloth_name):

    prefix = os.path.splitext(cloth_name)[0].split("_")[0]  
    for fname in os.listdir(dir_path):
        if fname.startswith(prefix + "_"):
            return fname
    return None


class ViViDDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=(512, 384),
        center_crop=False,
        instance_prompt="",
        scale=1.0,
        use_different=False,  # True=unpaired, False=paired
    ):
        self.data_root = Path(data_root)
        self.categories = {0: "upper_body", 1: "lower_body", 2: "dresses"}

        self.instance_prompt = instance_prompt
        self.cond_size = [int(x // scale) for x in size]
        self.center_crop = center_crop
        self.size = size
        self.use_different = use_different

        self.samples = []

        pair_files = ["unpairs.txt"] if use_different else ["pairs.txt"]

        for pf in pair_files:
            pair_path = self.data_root / pf
            if not pair_path.exists():
                continue
            with open(pair_path, "r") as f:
                for line in f:
                    img1, img2, subset_id = line.strip().split()
                    subset_id = int(subset_id)
                    if subset_id < 0 or subset_id >= len(self.categories):
                        raise ValueError(f"Invalid subset id {subset_id} in {pf}")
                    category = self.categories[subset_id]
                    self.samples.append((category, img1, img2))

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
        cloth_dir = base_dir / "cloth"
        image_ref_dir = base_dir / "image_ref"
        dense_dir = base_dir / "dense"
        agnostic_dir = base_dir / "agnostic"
        person_dir = base_dir / "person"

        agnostic_dir = base_dir / "agnostic"
        agnostic_mask_dir = base_dir / "agnostic_mask"

        ref_name = find_model_image(image_ref_dir, name2)
        example = {
            "image": self.load_image(image_dir, name1, self.transform),
            "agnostic": self.load_image(agnostic_dir, name1, self.transform),
            "cloth": self.load_image(cloth_dir, name2, self.transform),
            "image_ref": self.load_image(image_ref_dir, ref_name, self.transform),
            "person": self.load_image(image_dir, name1, self.transform_cond),
            "instance_prompt": self.instance_prompt,
            "category": category,
            "index": str(name1),
        }
        return example


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset = MyDataset(
        data_root="../../datasets/mydata",  
        size=(512, 384),
        center_crop=False,
        scale=1.0,
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {v}")
        if i >= 1:  
            break
