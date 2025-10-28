from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import random, os


class CombinedDataset(Dataset):
    def __init__(
        self,
        viton_root=None,
        dresscode_root=None,
        vivid_root=None,
        fashiontryon_root=None,
        igpairs_root=None,
        train=True,  # True / False / None
        size=(1024, 768),
        center_crop=False,
        instance_prompt="",
        scale=1.0,
        vivid_use_different=False,
        fashion_use_different=False,
        igpairs_use_different=False,  
    ):
        self.custom_instance_prompts = None
        self.viton_root = Path(viton_root) if viton_root else None
        self.dresscode_root = Path(dresscode_root) if dresscode_root else None
        self.vivid_root = Path(vivid_root) if vivid_root else None
        self.fashiontryon_root = Path(fashiontryon_root) if fashiontryon_root else None
        self.igpairs_root = Path(igpairs_root) if igpairs_root else None

        self.train = train
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.cond_size = [int(x // scale) for x in size]

        self.vivid_use_different = vivid_use_different
        self.fashion_use_different = fashion_use_different
        self.igpairs_use_different = igpairs_use_different

        # --- transforms ---
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

        self.samples = []

        # VITON
        if self.viton_root:
            subsets = (
                ["train"]
                if self.train is True
                else ["test"] if self.train is False else ["train", "test"]
            )
            for subset in subsets:
                subset_root = self.viton_root / subset
                image_dir = subset_root / "image"
                if not image_dir.exists():
                    continue
                names = sorted(
                    [
                        f.name
                        for f in image_dir.iterdir()
                        if f.suffix in [".jpg", ".png"]
                    ]
                )
                for name in names:
                    self.samples.append(
                        {"dataset": "viton", "subset": subset, "name": name}
                    )

        # DressCode
        if self.dresscode_root:
            categories = ("upper_body", "lower_body", "dresses")
            pair_files = (
                ["train_pairs.txt"]
                if self.train is True
                else (
                    ["test_pairs_paired.txt"]
                    if self.train is False
                    else ["train_pairs.txt", "test_pairs_paired.txt"]
                )
            )
            for pf in pair_files:
                pair_path = self.dresscode_root / pf
                if not pair_path.exists():
                    continue
                with open(pair_path, "r") as f:
                    for line in f:
                        img1, img2, subset_id = line.strip().split()
                        subset_id = int(subset_id)
                        category = categories[subset_id]
                        self.samples.append(
                            {
                                "dataset": "dresscode",
                                "category": category,
                                "name1": img1,
                                "name2": img2,
                            }
                        )

        # ViViD
        if self.vivid_root:
            categories = ("upper_body", "lower_body", "dresses")
            pair_files = ["unpairs.txt"] if self.vivid_use_different else ["pairs.txt"]
            for pf in pair_files:
                pair_path = self.vivid_root / pf
                if not pair_path.exists():
                    continue
                with open(pair_path, "r") as f:
                    for line in f:
                        img1, img2, subset_id = line.strip().split()
                        subset_id = int(subset_id)
                        category = categories[subset_id]
                        self.samples.append(
                            {
                                "dataset": "vivid",
                                "category": category,
                                "name1": img1,
                                "name2": img2,
                            }
                        )

        # FashionTryOn
        if self.fashiontryon_root:
            subsets = (
                ["train"]
                if self.train is True
                else ["test"] if self.train is False else ["train", "test"]
            )
            for subset in subsets:
                subset_root = self.fashiontryon_root / subset
                image_dir = subset_root / "images"
                if not image_dir.exists():
                    continue

                if self.fashion_use_different:
                    pairs_file = subset_root / "unpairs.txt"
                    if not pairs_file.exists():
                        continue
                    with open(pairs_file, "r") as f:
                        for line in f:
                            person_name, cloth_name = line.strip().split()
                            self.samples.append(
                                {
                                    "dataset": "fashiontryon",
                                    "subset": subset,
                                    "image_name": person_name,
                                    "cloth_name": cloth_name,
                                }
                            )
                else:
                    names = sorted(
                        [
                            f.name
                            for f in image_dir.iterdir()
                            if f.suffix in [".jpg", ".png"]
                        ]
                    )
                    for name in names:
                        self.samples.append(
                            {
                                "dataset": "fashiontryon",
                                "subset": subset,
                                "image_name": name,
                                "cloth_name": name,
                            }
                        )

        # IGpairs
        if self.igpairs_root:
            categories = ("upper_body", "lower_body", "dresses")
            pair_file = "unpairs.txt" if self.igpairs_use_different else "pairs.txt"
            pair_path = self.igpairs_root / pair_file
            if pair_path.exists():
                with open(pair_path, "r") as f:
                    for line in f:
                        img1, img2, subset_id = line.strip().split()
                        subset_id = int(subset_id)
                        if subset_id < 0 or subset_id >= len(categories):
                            raise ValueError(
                                f"Invalid subset id {subset_id} in {pair_file}"
                            )
                        category = categories[subset_id]
                        self.samples.append(
                            {
                                "dataset": "igpairs",
                                "category": category,
                                "name1": img1,
                                "name2": img2,
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def load_image(self, path, cond=False):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform_cond(img) if cond else self.transform(img)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # VITON
        if sample["dataset"] == "viton":
            subset_root = self.viton_root / sample["subset"]
            name = sample["name"]

            if self.train:
                agnostic_dir = subset_root / (
                    "agnostic" if random.random() < 0.5 else "agnostic_refined"
                )
            else:
                agnostic_dir = subset_root / "agnostic"

            example = {
                "image": self.load_image(subset_root / "image" / name),
                "agnostic": self.load_image(agnostic_dir / name),
                "person": self.load_image(subset_root / "person" / name),
                "cloth": self.load_image(subset_root / "cloth" / name),
                "image_ref": self.load_image(
                    subset_root / "image_ref" / name, cond=True
                ),
                "instance_prompt": self.instance_prompt,
                "dataset": "viton",
                "index": str(name),
            }

        # DressCode
        elif sample["dataset"] == "dresscode":
            cat, name1, name2 = sample["category"], sample["name1"], sample["name2"]
            base_dir = self.dresscode_root / cat
            prob_ft = random.random()
            agnostic_dir = base_dir / (
                "agnostic_catvton" if prob_ft < 0.75 else "agnostic_enhanced"
            )

            example = {
                "image": self.load_image(base_dir / "images" / name1),
                "agnostic": self.load_image(
                    agnostic_dir / name1.replace("_0.jpg", "_6.jpg")
                ),
                "cloth": self.load_image(base_dir / "images" / name2),
                "image_ref": self.load_image(
                    base_dir / "image_ref" / name2.replace("_1.jpg", "_0.jpg"),
                    cond=True,
                ),
                "person": self.load_image(base_dir / "person" / name1),
                "instance_prompt": self.instance_prompt,
                "dataset": "dresscode",
                "index": str(name1),
            }

        # ViViD
        elif sample["dataset"] == "vivid":
            cat, name1, name2 = sample["category"], sample["name1"], sample["name2"]
            base_dir = self.vivid_root / cat

            example = {
                "image": self.load_image(base_dir / "images" / name1),
                "agnostic": self.load_image(base_dir / "agnostic" / name1),
                "cloth": self.load_image(base_dir / "cloth" / name2),
                "image_ref": self.load_image(base_dir / "image_ref" / name1, cond=True),
                "person": self.load_image(base_dir / "person" / name1),
                "instance_prompt": self.instance_prompt,
                "dataset": "vivid",
                "index": str(name1),
            }

        # FashionTryOn
        elif sample["dataset"] == "fashiontryon":
            subset_root = self.fashiontryon_root / sample["subset"]
            image_name, cloth_name = sample["image_name"], sample["cloth_name"]
            prob_ft = random.random()

            if self.train:
                agnostic_dir = subset_root / (
                    "agnostic" if prob_ft < 0.6 else "agnostic_enhanced"
                )
            else:
                agnostic_dir = subset_root / "agnostic"

            example = {
                "image": self.load_image(subset_root / "images" / image_name),
                "agnostic": self.load_image(agnostic_dir / image_name),
                "person": self.load_image(subset_root / "person" / image_name),
                "cloth": self.load_image(
                    subset_root / "cloth" / (cloth_name[:5] + ".jpg")
                ),
                "image_ref": self.load_image(
                    subset_root / "image_ref" / image_name, cond=True
                ),
                "instance_prompt": self.instance_prompt,
                "dataset": "fashiontryon",
                "index": str(image_name),
            }

        # IGpairs
        elif sample["dataset"] == "igpairs":
            cat, name1, name2 = sample["category"], sample["name1"], sample["name2"]
            base_dir = self.igpairs_root / cat

            example = {
                "image": self.load_image(base_dir / "images" / name1),
                "agnostic": self.load_image(base_dir / "agnostic" / name1),
                "cloth": self.load_image(base_dir / "cloth" / name2),
                "image_ref": self.load_image(base_dir / "image_ref" / name1, cond=True),
                "person": self.load_image(base_dir / "person" / name1),
                "instance_prompt": self.instance_prompt,
                "dataset": "igpairs",
                "index": str(name1),
            }

        else:
            raise ValueError(f"Unsupported dataset type: {sample['dataset']}")

        return example


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    import os

    # 路径根据你本地的数据修改
    viton_root = "../../datasets/viton"
    dresscode_root = "../../datasets/DressCode"
    vivid_root = "../../datasets/vivid_processed"
    fashion_root = "../../datasets/FashionTryOn_processed"
    igpairs_root = "../../datasets/IGPair_processed"

    # 输出文件夹
    out_dir = "folder"
    os.makedirs(out_dir, exist_ok=True)
    for sub in ["cloth", "image", "agnostic", "image_ref", "person"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # 初始化数据集

    dataset = CombinedDataset(
        viton_root=viton_root,
        dresscode_root=dresscode_root,
        vivid_root=vivid_root,
        fashiontryon_root=fashion_root,
        igpairs_root=igpairs_root,
        size=(512, 384),
        train=True,
        center_crop=False,
        scale=2.0,
        vivid_use_different=False,  # ViViD: False=paired, True=unpaired
        fashion_use_different=False,  # FashionTryOn: False=paired, True=unpaired
    )

    def denormalize(tensor):
        return tensor * 0.5 + 0.5

    print(f"Total samples in CombinedDataset: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    for batch_idx, batch in enumerate(dataloader):
        print(f"\n=== Batch {batch_idx} ===")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
            else:
                print(f"  {key}: {value}")

        # 保存可视化
        for i in range(len(batch["image"])):

            index = batch["index"][i]
            dataset_name = batch["dataset"][i]

            save_image(
                denormalize(batch["image"][i]),
                os.path.join(out_dir, "image", f"{dataset_name}_{index}"),
            )
            save_image(
                denormalize(batch["cloth"][i]),
                os.path.join(out_dir, "cloth", f"{dataset_name}_{index}"),
            )
            save_image(
                denormalize(batch["agnostic"][i]),
                os.path.join(out_dir, "agnostic", f"{dataset_name}_{index}"),
            )
            save_image(
                denormalize(batch["image_ref"][i]),
                os.path.join(out_dir, "image_ref", f"{dataset_name}_{index}"),
            )
            save_image(
                denormalize(batch["person"][i]),
                os.path.join(out_dir, "person", f"{dataset_name}_{index}"),
            )

        if batch_idx == 1:  # 只看前两个 batch
            break
