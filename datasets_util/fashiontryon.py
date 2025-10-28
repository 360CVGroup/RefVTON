import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


def find_model_image(dir_path, cloth_name):

    prefix = os.path.splitext(cloth_name)[0]  # 去掉后缀，得到 "xxxx"
    for fname in os.listdir(dir_path):
        if fname.startswith(prefix[:5] + "_"):
            return fname
    return None


class FashTryOn(Dataset):
    def __init__(
        self,
        data_root,
        size=(512, 384),
        train=True,  # True / False / None
        center_crop=False,
        instance_prompt="",
        use_different=False,
    ):
        self.data_root = Path(data_root)
        self.train = train
        self.use_different = use_different
        self.instance_prompt = instance_prompt
        self.center_crop = center_crop
        self.size = size

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
            image_dir = subset_root / "images"
            cloth_dir = subset_root / "cloth"

            if self.use_different:
                pairs_file = subset_root / "unpairs.txt"
                if not pairs_file.exists():
                    raise FileNotFoundError(f"Missing pairs file: {pairs_file}")
                with open(pairs_file, "r") as f:
                    for line in f:
                        person_name, cloth_name = line.strip().split()
                        # reference_name 和 person_name 一样，保证 __getitem__ 不报错
                        self.samples.append(
                            (subset, person_name, cloth_name, person_name)
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
                    # reference_name 和 person_name 一样
                    self.samples.append((subset, name, name, name))

        # transforms
        crop_op = (
            transforms.CenterCrop(self.size)
            if center_crop
            else transforms.RandomCrop(self.size)
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
        subset, image_name, cloth_name, reference_name = self.samples[idx]
        # person_name = image_name
        # print(subset, person_name, cloth_name, reference_name)
        # asfd
        subset_root = self.data_root / subset

        image_dir = subset_root / "images"
        person_dir = subset_root / "person"
        image_ref_dir = subset_root / "image_ref"
        cloth_dir = subset_root / "cloth"

        # train 时动态选择 agnostic 路径
        if self.train:
            if random.random() < 0.25:
                agnostic_dir = subset_root / "agnostic"
            else:
                agnostic_dir = subset_root / "agnostic_enhanced"
        else:
            agnostic_dir = subset_root / "agnostic"
        ref_name = image_name

        example = {
            "images": self.load_image(image_dir, image_name, self.transform),
            "agnostic": self.load_image(agnostic_dir, image_name, self.transform),
            "person": self.load_image(person_dir, image_name, self.transform),
            "cloth": self.load_image(
                cloth_dir, cloth_name[:5] + ".jpg", self.transform
            ),
            "image_ref": self.load_image(image_ref_dir, ref_name, self.transform),
            "instance_prompt": self.instance_prompt,
            "index": str(person_name),
        }
        return example


if __name__ == "__main__":
    dataset = FashTryOn(
        data_root="../../datasets/FasionTryOn_processed",
        train=True,
        use_different=False,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print("Dataset length:", len(dataset))
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Image batch shape:", batch["images"].shape)
        print("Person batch shape:", batch["person"].shape)
        print("Agnostic batch shape:", batch["agnostic"].shape)
        print("Cloth batch shape:", batch["cloth"].shape)
        print("Reference image batch shape:", batch["image_ref"].shape)
        print("Instance prompt:", batch["instance_prompt"])
        print("Index list:", batch["index"])
        break
