from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, random, os


from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=(512, 384),
        instance_prompt = "",
        center_crop=False,
        scale=1.0,
    ):
        self.data_root = Path(data_root)
        self.size = size
        self.cond_size = [int(x // scale) for x in size]

        image_dir = self.data_root / "images"
        cloth_dir = self.data_root / "cloth"
        agnostic_dir = self.data_root / "agnostic"
        image_ref_dir = self.data_root / "image_ref"
        person_dir = self.data_root / "person"
        

        self.samples = sorted(
            [
                f.name
                for f in cloth_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".png"]
            ]
        )

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

        self.image_dir = image_dir
        self.cloth_dir = cloth_dir
        self.agnostic_dir = agnostic_dir
        self.image_ref_dir = image_ref_dir
        self.person_dir = person_dir
        self.instance_prompt = instance_prompt

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
        name = self.samples[idx]

        example = {
            "cloth": self.load_image(self.cloth_dir, name, self.transform),
            "instance_prompt": self.instance_prompt,
            "index": str(name),
        }
        if os.path.exists(self.image_dir):
            example["image"] = self.load_image(self.image_dir, name, self.transform)
        if os.path.exists(self.agnostic_dir):
            example["agnostic"] = self.load_image(self.agnostic_dir, name, self.transform)
        if os.path.exists(self.image_ref_dir):
            example["image_ref"] = self.load_image(self.image_ref_dir, name, self.transform_cond)
        if os.path.exists(self.person_dir):
            example["person"] = self.load_image(self.person_dir, name, self.transform)
        return example




if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = MyDataset(
        data_root="test_image",  
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

