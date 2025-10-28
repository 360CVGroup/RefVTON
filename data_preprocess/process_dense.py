import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Mapping table from RGB colors to label IDs
label_rgb_vivid = {
    0: [0, 0, 0],
    2: [71, 30, 109],
    3: [70, 43, 120],
    4: [66, 65, 127],
    5: [61, 69, 132],
    6: [58, 80, 137],
    9: [42, 112, 138],
    10: [37, 123, 140],
    13: [30, 152, 137],
    14: [28, 161, 130],
    15: [36, 172, 124],
    16: [52, 182, 120],
    18: [96, 197, 101],
    17: [64, 192, 107],
    19: [107, 205, 82],
    20: [142, 211, 66],
    21: [167, 218, 53],
    22: [193, 223, 37],
    23: [222, 227, 23],
    24: [250, 231, 33],
}

label_rgb_viton = {
    0: [0, 0, 0],
    2: [20, 80, 194],
    3: [4, 98, 224],
    4: [8, 110, 221],
    5: [13, 122, 215],
    6: [20, 133, 211],
    9: [6, 166, 198],
    10: [22, 173, 184],
    13: [88, 186, 145],
    14: [114, 189, 130],
    15: [145, 191, 116],
    16: [170, 190, 105],
    17: [193, 188, 97],
    18: [216, 187, 87],
    19: [228, 191, 74],
    20: [240, 198, 60],
    21: [252, 207, 46],
    22: [250, 220, 36],
    23: [251, 235, 25],
    24: [248, 251, 14],
}

# Precompute RGB and label arrays
rgb_values_viton = np.array(list(label_rgb_viton.values()), dtype=np.float32)  # (N, 3)
label_values_viton = np.array(list(label_rgb_viton.keys()), dtype=np.uint8)  # (N,)

rgb_values_vivid = np.array(list(label_rgb_vivid.values()), dtype=np.float32)  # (N, 3)
label_values_vivid = np.array(list(label_rgb_vivid.keys()), dtype=np.uint8)  # (N,)

label_values = label_values_vivid
rgb_values = rgb_values_vivid

# Build KDTree for fast nearest-neighbor color lookup
kdtree = cKDTree(rgb_values)


def densepose_rgb_to_label_nearest(rgb_path, save_path):
    """
    Convert an RGB DensePose image into a single-channel label map
    using KDTree nearest color matching.
    """
    img = Image.open(rgb_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32)  # (H, W, 3)

    H, W, _ = img_np.shape
    img_flat = img_np.reshape(-1, 3)  # (H*W, 3)

    # Query KDTree for nearest color index
    _, nearest_idx = kdtree.query(img_flat)  # return indices
    label_map = label_values[nearest_idx].reshape(H, W)

    # Save label map
    Image.fromarray(label_map).save(save_path)


def batch_convert_densepose(input_dir, output_dir, exts=(".png", ".jpg", ".jpeg")):
    """
    Batch convert all DensePose RGB images in a folder to label maps.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(files)} DensePose RGB images.")

    for file in tqdm(files, desc="Converting..."):
        rel_path = file.relative_to(input_dir)
        save_path = output_dir / rel_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        densepose_rgb_to_label_nearest(file, save_path)


def visualize_rgb_and_label(rgb_dir, label_dir):
    """
    Sequentially visualize RGB DensePose images and their corresponding label maps.

    Args:
        rgb_dir (str or Path): Directory containing original RGB DensePose images (.jpg)
        label_dir (str or Path): Directory containing corresponding label maps (.png)
    """
    rgb_dir = Path(rgb_dir)
    label_dir = Path(label_dir)

    rgb_files = sorted([p for p in rgb_dir.glob("*.jpg")])
    if not rgb_files:
        print(f"No JPG files found in {rgb_dir}")
        return

    for rgb_path in rgb_files:
        img1 = np.array(Image.open(rgb_path).convert("RGB"))

        # Find the corresponding label map (.png)
        label_path = label_dir / rgb_path.with_suffix(".png").name
        if not label_path.exists():
            print(f"Missing label: {label_path}")
            continue
        img2 = np.array(Image.open(label_path))

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.title("RGB DensePose")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.title("Label Map")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Modify these paths according to your dataset location
    input_folder = "../../datasets/ViViD_processed/dresses/densepose"
    output_folder = "../../datasets/ViViD_processed/dresses/dense"

    batch_convert_densepose(input_folder, output_folder)
    # visualize_rgb_and_label(input_folder, output_folder)
