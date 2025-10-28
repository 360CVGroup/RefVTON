from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class IGpairsDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=(512, 384),
        center_crop=False,
        instance_prompt="",
        scale=1.0,
        use_different=True,  # True=unpaired, False=paired
    ):
        self.data_root = Path(data_root)
        self.categories = ("upper_body", "lower_body", "dresses")

        self.instance_prompt = instance_prompt
        self.cond_size = [int(x // scale) for x in size]
        self.center_crop = center_crop
        self.size = size
        self.use_different = use_different

        self.samples = []

        pair_file = "unpairs.txt" if use_different else "pairs.txt"
        pair_path = self.data_root / pair_file
        if pair_path.exists():
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

        self.samples = self.samples[8000:]
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
        agnostic_mask_dir = base_dir / "agnostic_mask"

        ref_name = name1

        example = {
            "image": self.load_image(image_dir, name1, self.transform),
            "agnostic": self.load_image(agnostic_dir, name1, self.transform),
            "cloth": self.load_image(cloth_dir, name2, self.transform),
            "image_ref": self.load_image(image_ref_dir, ref_name, self.transform),
            #  "person": self.load_image(person_dir, name1, self.transform),
            "instance_prompt": self.instance_prompt,
            "category": category,
            "index": str(name1),
        }
        return example


import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = IGpairsDataset(
        data_root="../../datasets/IGPair_processed",
        size=(512, 384),
        center_crop=False,
        instance_prompt="test prompt",
        scale=1.0,
        use_different=False,  # False -> pair.txt, True -> unpair.txt
    )

    print("Dataset length:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}")
        print("image:", batch["image"].shape)
        print("agnostic:", batch["agnostic"].shape)
        print("cloth:", batch["cloth"].shape)
        print("image_ref:", batch["image_ref"].shape)
        print("person:", batch["person"].shape)
        print("index:", batch["index"])

        if i == 1:  
            break
