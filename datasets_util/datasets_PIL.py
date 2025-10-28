from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, random


from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image


class ViViDDataset_PIL(Dataset):
    def __init__(
        self,
        data_root,
        train=False,
        categories=("dresses", "upper_body", "lower_body"),  # 可单个或多个类别
    ):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.samples = []

        self.data_root = Path(data_root)
        if isinstance(categories, str):
            categories = [categories]
        self.categories = categories

        # 遍历每个类别的 images 文件夹
        for category in categories:
            image_dir = self.data_root / category / "new_image"
            image_names = sorted(
                [
                    f.name
                    for f in image_dir.iterdir()
                    if f.suffix.lower() in [".jpg", ".png"]
                ]
            )
            # 保存 (类别, 文件名)

            self.samples.extend([(category, name) for name in image_names])

    def __len__(self):
        return len(self.samples)

    def load_image(self, directory, name):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        category, name = self.samples[idx]
        img = self.load_image(self.data_root / category / "new_image", name)
        return img, name, category


class DressCodeDataset_PIL(Dataset):
    def __init__(
        self,
        data_root,
        categories=("dresses", "upper_body", "lower_body"),  # 可以单个或多个类别
    ):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.samples = []

        self.data_root = Path(data_root)
        if isinstance(categories, str):
            categories = [categories]
        self.categories = categories

        for category in categories:
            image_dir = self.data_root / category / "agnostic"
            image_names = sorted(
                [
                    f.name
                    for f in image_dir.iterdir()
                    if f.suffix.lower() in [".jpg", ".png"]
                ]
            )
            print(image_names)
            # 保存 (category, 文件名)
            self.samples.extend(
                [(category, name[:-6] + "_0.jpg") for name in image_names]
            )

    def __len__(self):
        return len(self.samples)

    def load_image(self, directory, name):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        category, name = self.samples[idx]
        img = self.load_image(self.data_root / category / "image", name)
        return img, name, category


class VITONDataset_PIL(Dataset):
    def __init__(
        self,
        data_root,
        train=True,  # True -> train, False -> test, None -> both
    ):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.samples = []

        if train is True:
            subsets = ["train"]
        elif train is False:
            subsets = ["test"]
        else:  # train=None -> both
            subsets = ["train", "test"]

        for subset in subsets:
            image_dir = Path(data_root) / subset / "image"
            image_names = sorted(
                [f.name for f in image_dir.iterdir() if f.suffix in [".jpg", ".png"]]
            )
            # 保存 (subset, 文件名) 方便后续知道属于 train 还是 test
            self.samples.extend([(subset, name) for name in image_names])

        self.data_root = Path(data_root)

    def __len__(self):
        return len(self.samples)

    def load_image(self, directory, name):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        subset, name = self.samples[idx]
        img = self.load_image(self.data_root / subset / "image", name)
        # 返回图像、文件名、子集名
        return img, name, subset


if __name__ == "__main__":
    dataset = ViViDDataset_PIL(
        data_root="../../datasets/IGPair_processed",
        categories=("upper_body"),  # 只加载上衣
    )

    # img, name, category = dataset[0]
    # print(len(dataset))
    # print(img.shape, name, category)
