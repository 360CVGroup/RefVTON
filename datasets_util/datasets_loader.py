from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, random


def viton_collate_fn(examples):
    batch = {}
    if "image" in examples[0]:
        pixel_values = [example["image"] for example in examples]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        batch["pixel_values"] = pixel_values

    if "image_ref" in examples[0]:
        pixel_values_ref = [example["image_ref"] for example in examples]
        pixel_values_ref = torch.stack(pixel_values_ref)
        pixel_values_ref = pixel_values_ref.to(
            memory_format=torch.contiguous_format
        ).float()
        batch["pixel_values_ref"] = pixel_values_ref

    if "agnostic" in examples[0]:
        cond_pixel_values_agnostic = [example["agnostic"] for example in examples]
        cond_pixel_values_agnostic = torch.stack(cond_pixel_values_agnostic)
        cond_pixel_values_agnostic = cond_pixel_values_agnostic.to(
            memory_format=torch.contiguous_format
        ).float()
        batch["cond_pixel_values_agnostic"] = cond_pixel_values_agnostic

    if "cloth" in examples[0]:
        cond_pixel_values_cloth = [example["cloth"] for example in examples]
        cond_pixel_values_cloth = torch.stack(cond_pixel_values_cloth)
        cond_pixel_values_cloth = cond_pixel_values_cloth.to(
            memory_format=torch.contiguous_format
        ).float()
        batch["cond_pixel_values_cloth"] = cond_pixel_values_cloth

    if "dense" in examples[0]:
        cond_pixel_values_dense = [example["dense"] for example in examples]
        cond_pixel_values_dense = torch.stack(cond_pixel_values_dense)
        cond_pixel_values_dense = cond_pixel_values_dense.to(
            memory_format=torch.contiguous_format
        ).float()
        batch["cond_pixel_values_dense"] = cond_pixel_values_dense

    if "agnostic_mask" in examples[0]:
        cond_pixel_values_agnostic_mask = [
            example["agnostic_mask"] for example in examples
        ]
        cond_pixel_values_agnostic_mask = torch.stack(cond_pixel_values_agnostic_mask)
        cond_pixel_values_agnostic_mask = cond_pixel_values_agnostic_mask.to(
            memory_format=torch.contiguous_format
        ).float()
        batch["cond_pixel_values_agnostic_mask"] = cond_pixel_values_agnostic_mask

    if "person" in examples[0]:
        cond_pixel_values_person = [example["person"] for example in examples]
        cond_pixel_values_person = torch.stack(cond_pixel_values_person)
        cond_pixel_values_person = cond_pixel_values_person.to(
            memory_format=torch.contiguous_format
        ).float()
        batch["cond_pixel_values_person"] = cond_pixel_values_person

    if "category" in examples[0]:
        category = [example["category"] for example in examples]
        batch["category"] = category

    if "instance_prompt" in examples[0]:
        prompts = [example["instance_prompt"] for example in examples]
        batch["prompts"] = prompts

    if "index" in examples[0]:
        index = [example["index"] for example in examples]
        batch["index"] = index

    return batch


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

        # train 时动态选择 agnostic 路径
        agnostic_mask_dir = subset_root / "agnostic_mask"
        agnostic_dir = subset_root / "agnostic"

        # 参考 cloth
        if random.random() > 0.5:
            reference_name = cloth_name
        else:
            reference_name = cloth_name[:-5] + "1.jpg"

        # print(agnostic_mask_dir, agnostic_dir)
        example = {
            "image": self.load_image(image_dir, person_name, self.transform),
            "agnostic": self.load_image(agnostic_dir, person_name, self.transform),
            "person": self.load_image(person_dir, person_name, self.transform),
            "cloth": self.load_image(cloth_dir, cloth_name, self.transform),
            "image_ref": self.load_image(image_ref_dir, reference_name, self.transform),
            "agnostic_mask": self.load_image(
                agnostic_mask_dir, person_name[:-4] + "_mask.png", self.transform_cond
            ),
            "dense": self.load_image(
                dense_dir, person_name[:-4] + ".png", self.transform_cond
            ),
            "instance_prompt": self.instance_prompt,
            "index": str(person_name),
        }
        return example


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
                transforms.ToTensor(),
                transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                crop_op,
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

        if random.random() > 0.5:
            agnostic_dir = base_dir / "agnostic"
            agnostic_mask_dir = base_dir / "agnostic_mask"
        else:
            agnostic_dir = base_dir / "agnostic_enhanced"
            agnostic_mask_dir = base_dir / "agnostic_mask_enhanced"

        example = {
            "image": self.load_image(image_dir, name1, self.transform),
            "agnostic": self.load_image(
                agnostic_dir, name1.replace("_0.jpg", "_6.jpg"), self.transform
            ),
            "cloth": self.load_image(image_dir, name2, self.transform),
            "image_ref": self.load_image(
                image_ref_dir, name2[:-5] + "0.jpg", self.transform
            ),
            "agnostic_mask": self.load_image(
                agnostic_mask_dir,
                name1.replace("_0.jpg", "_3.png"),
                self.transform_cond,
            ),
            "dense": self.load_image(
                dense_dir, name1.replace("_0.jpg", "_5.png"), self.transform_cond
            ),
            "person": self.load_image(person_dir, name1, self.transform_cond),
            "instance_prompt": self.instance_prompt,
            "category": category,
            "index": str(name1),
        }
        return example



class ViViDDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=(512, 384),
        center_crop=False,
        instance_prompt="",
        scale=1.0,
        use_different=False,  # 支持 unpaired 文件
    ):
        self.data_root = Path(data_root)
        self.categories = ("upper_body", "lower_body", "dresses")

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.cond_size = [int(x // scale) for x in size]
        self.center_crop = center_crop
        self.size = size
        self.use_different = use_different

        self.samples = []

        # ----------------------------------------------------
        # 读取 pair 文件
        # ----------------------------------------------------
        if not self.use_different:
            pair_file = self.data_root / "pairs.txt"
        else:
            pair_file = self.data_root / "pairs_unpaired.txt"

        if not pair_file.exists():
            raise FileNotFoundError(f"Pair file not found: {pair_file}")

        with open(pair_file, "r") as f:
            for line in f:
                img1, img2, subset_id = line.strip().split()
                subset_id = int(subset_id)
                if subset_id < 0 or subset_id >= len(self.categories):
                    raise ValueError(f"Invalid subset id {subset_id} in {pair_file}")
                category = self.categories[subset_id]
                self.samples.append((category, img1, img2))

        # ----------------------------------------------------
        # 定义 transform
        # ----------------------------------------------------
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
        agnostic_dir = base_dir / "agnostic"
        agnostic_mask_dir = base_dir / "agnostic_mask"
        densepose_dir = base_dir / "densepose"

        example = {
            "image": self.load_image(image_dir, name1, self.transform),
            "cloth": self.load_image(image_dir, name2, self.transform),
            "agnostic": self.load_image(agnostic_dir, name1, self.transform),
            "agnostic_mask": self.load_image(
                agnostic_mask_dir, name1, self.transform_cond
            ),
            "densepose": self.load_image(densepose_dir, name1, self.transform_cond),
            "instance_prompt": self.instance_prompt,
            "category": category,
            "index": str(name1),
        }
        return example


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from inference import save_tensor_as_png

    train = True
    # 数据集路径
    viton_root = "../../datasets/viton_test"
    dresscode_root = "../../datasets/DressCode"

    dataset_viton = VITONDataset(
        data_root=viton_root, size=(512, 384), train=train, center_crop=False, scale=2.0
    )
    dataset_dc = DressCodeDataset(
        data_root=dresscode_root,
        size=(512, 384),
        train=train,
        center_crop=False,
        scale=2.0,
    )

    dataset = dataset_dc
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # print(f"Total samples: {len(dataset)}")
    # print(f"Total batches: {len(dataloader)}")

    for batch in dataloader:
        # print("Loaded batch:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
            else:
                print(f"  {key}: {value}")

        for i in range(len(batch["cloth"])):
            name = batch["index"][i]
            save_tensor_as_png(batch["cloth"][i], f"temp/{name}")

        break  # 只看第一个 batch
