import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2  # fast dilation

MASK_RGB_DRESSCODE_all = {
    "dresses": [
        (128, 128, 128),
        (64, 128, 128),
        (192, 128, 128),
        (192, 0, 128),
        (64, 0, 128),
    ],
    "upper_body": [(0, 0, 128), (64, 128, 128), (192, 128, 128)],
    "lower_body": [(0, 128, 128), (192, 0, 128), (64, 0, 128)],
}

MASK_RGB_DRESSCODE_body = {
    "dresses": [
        (64, 128, 128),
        (192, 128, 128),
        (192, 0, 128),
        (64, 0, 128),
    ],
    "upper_body": [(64, 128, 128), (192, 128, 128)],
    "lower_body": [(192, 0, 128), (64, 0, 128)],
}

MASK_RGB_DRESSCODE = {
    "dresses": (128, 128, 128),
    "upper_body": (0, 0, 128),
    "lower_body": (0, 128, 128),
}


MASK_RGB_VITON = [[0, 0, 85], [254, 85, 0], [0, 119, 220]]


def dilate_mask(binary_mask, r):
    if r <= 0:
        return binary_mask.copy()
    k = 2 * r + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(binary_mask, kernel, iterations=1)


def generate_agnostic_and_mask(
    image_path,
    parse_path,
    dense_path,
    agnostic_out_path,
    mask_out_path,
    subset="dresses",
    dilation_radius1=20,  # d1
    dilation_radius2=5,   # d2
    dilation_radius0=10,  # d3
):
    mask_rgbs_all = np.array(MASK_RGB_DRESSCODE_all[subset], dtype=np.uint8)  # (K,3)
    mask_rgbs_body = np.array(MASK_RGB_DRESSCODE_body[subset], dtype=np.uint8)  # (K,3)
    mask_rgb = np.array(MASK_RGB_DRESSCODE[subset], dtype=np.uint8)  # (3,)

    image = Image.open(image_path).convert("RGB")
    parse = Image.open(parse_path).convert("RGB")
    dense = Image.open(dense_path).convert("L")

    image_np = np.array(image)
    parse_np = np.array(parse)
    dense_np = np.array(dense)

    H, W, _ = parse_np.shape

    # -------- cloth_mask_all (may contain multiple RGB values) --------
    parse_flat = parse_np.reshape(-1, 3)[:, None, :]  # (H*W,1,3)
    mask_rgbs_all = mask_rgbs_all[None, :, :]  # (1,K,3)
    matches_all = np.all(parse_flat == mask_rgbs_all, axis=-1)  # (H*W,K)
    cloth_mask_all = matches_all.any(axis=1).reshape(H, W).astype(np.uint8) * 255

    # -------- cloth_mask (single RGB) --------
    cloth_mask = (np.all(parse_np == mask_rgb, axis=-1)).astype(np.uint8) * 255

    # -------- keep region --------
    keep_labels = [3, 4, 5, 6, 23, 24]
    keep_mask = np.isin(dense_np, keep_labels).astype(np.uint8) * 255  # 0/255

    # -------- first dilation logic --------
    cloth_mask_minus_keep = cloth_mask_all.copy()
    cloth_mask_minus_keep[keep_mask > 0] = 0
    mask_d1 = dilate_mask(cloth_mask_minus_keep, dilation_radius1)
    mask_d1[keep_mask > 0] = 0

    # -------- Added: third dilation logic (for body region) --------
    parse_flat = parse_np.reshape(-1, 3)[:, None, :]  # (H*W,1,3)
    mask_rgbs_body = mask_rgbs_body[None, :, :]       # (1,K,3)
    matches_body = np.all(parse_flat == mask_rgbs_body, axis=-1)  # (H*W,K)
    body_mask = matches_body.any(axis=1).reshape(H, W).astype(np.uint8) * 255

    # Dilate only within the body region
    body_mask_dilated = dilate_mask(body_mask, dilation_radius0)
    mask_d3 = body_mask_dilated.copy()
    mask_d3[keep_mask > 0] = 0  # remove keep region

    # -------- second dilation: single RGB mask --------
    mask_d2 = dilate_mask(cloth_mask, dilation_radius2)
    # Note: keep_mask is not excluded here

    # -------- generate agnostic image --------
    agnostic_np = image_np.copy()
    agnostic_np[mask_d1 > 0] = 128
    agnostic_np[mask_d3 > 0] = 128   # newly added step
    agnostic_np[mask_d2 > 0] = 128

    # -------- save agnostic image --------
    agnostic_img = Image.fromarray(agnostic_np)
    agnostic_img.save(agnostic_out_path)

    # -------- final mask: based on pixel differences --------
    changed = np.any(agnostic_np != image_np, axis=-1)
    mask_applied = changed.astype(np.uint8) * 255
    Image.fromarray(mask_applied).save(mask_out_path)



def batch_generate_agnostic_and_mask_dresscode(
    data_root,
    subset="dresses",
    dilation_radius1=25,
    dilation_radius2=5,
    dilation_radius0=30,
):
    image_dir = os.path.join(data_root, "images")
    parse_dir = os.path.join(data_root, "label_maps")
    dense_dir = os.path.join(data_root, "dense")

    agnostic_dir = os.path.join(data_root, f"agnostic_enhanced")
    mask_dir = os.path.join(data_root, f"agnostic_mask_enhanced")
    os.makedirs(agnostic_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_names = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png")) and "_0" in f]
    )

    for i, img_name in enumerate(tqdm(image_names, desc=f"{subset}")):
        image_path = os.path.join(image_dir, img_name)
        parse_path = os.path.join(parse_dir, img_name.replace("_0.jpg", "_4.png"))
        dense_path = os.path.join(dense_dir, img_name.replace("_0.jpg", "_5.png"))

        agnostic_out_path = os.path.join(
            agnostic_dir, img_name.replace("_0.jpg", "_6.jpg")
        )
        mask_out_path = os.path.join(mask_dir, img_name.replace("_0.jpg", "_3.png"))

        generate_agnostic_and_mask(
            image_path,
            parse_path,
            dense_path,
            agnostic_out_path,
            mask_out_path,
            subset=subset,
            dilation_radius1=dilation_radius1,
            dilation_radius2=dilation_radius2,
            dilation_radius0=dilation_radius0,
        )


def dilate_mask(binary_mask, r):
    """binary_mask: uint8 0/255"""
    if r <= 0:
        return binary_mask.copy()
    k = 2 * r + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(binary_mask, kernel, iterations=1)


def generate_viton_agnostic_and_mask(
    image_path,
    mask1_path,
    parse_path,
    dense_path,
    agnostic_out_path,
    mask_out_path,
    dilation_radius1=25,  # used to first dilate mask2 (parse cloth region)
    dilation_radius2=5,  # used to second dilate the union (do not exclude hands/face)
    fill_value=128,  # fill color
):
    # Load inputs
    image = Image.open(image_path).convert("RGB")
    mask1 = Image.open(mask1_path).convert("L")  # initial mask1 (single-channel)
    parse = Image.open(parse_path).convert("RGB")  # parse color image
    dense = Image.open(dense_path).convert("L")  # densepose single-channel

    image_np = np.array(image)
    mask1_np = np.array(mask1)
    parse_np = np.array(parse)
    dense_np = np.array(dense)

    # parse -> mask2 (cloth region, multiple RGB values)
    cloth_mask2 = np.zeros(parse_np.shape[:2], dtype=np.uint8)
    for rgb in MASK_RGB_VITON:
        cloth_mask2 |= np.all(parse_np == rgb, axis=-1).astype(np.uint8)
    cloth_mask2 = cloth_mask2 * 255  # 0/255

    # Binarize mask1 to 0/255
    mask1_bin = (mask1_np > 0).astype(np.uint8) * 255

    # ========== Step2: first dilate mask2, then union with mask1 ==========
    mask2_dil = dilate_mask(cloth_mask2, dilation_radius1)
    # Union (mask1 ∪ mask2_dil)
    mask_union = np.maximum(mask1_bin, mask2_dil)  # 0/255

    # ========== Step3: first application: remove hands/face (based on mask_union) ==========
    keep_labels = [3, 4, 23, 24]  # hands, face, etc.
    keep_mask = np.isin(dense_np, keep_labels)  # bool

    mask_first_apply = mask_union.copy()
    mask_first_apply[keep_mask] = 0  # clear hands/face positions

    agnostic_np = image_np.copy()
    agnostic_np[mask_first_apply > 0] = fill_value

    # ========== Step4: second dilation (based on mask_union, do not remove hands/face) ==========
    mask_second = dilate_mask(cloth_mask2, dilation_radius2)
    agnostic_np[mask_second > 0] = fill_value

    # Save agnostic image
    agnostic_img = Image.fromarray(agnostic_np)
    agnostic_img.save(agnostic_out_path)

    # Save final mask: pixels actually covered (based on pixel differences)
    changed = np.any(agnostic_np != image_np, axis=-1)
    mask_final = changed.astype(np.uint8) * 255
    Image.fromarray(mask_final).save(mask_out_path)


def batch_generate_viton_agnostic_and_mask(
    data_root,
    dilation_radius1=25,
    dilation_radius2=5,
):
    image_dir = os.path.join(data_root, "image")
    mask1_dir = os.path.join(data_root, "agnostic_mask")
    parse_dir = os.path.join(data_root, "image-parse-v3")
    dense_dir = os.path.join(data_root, "dense")

    agnostic_dir = os.path.join(data_root, "agnostic_refined")
    mask_dir = os.path.join(data_root, "agnostic_mask_refined")
    os.makedirs(agnostic_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_names = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
    )

    for i, img_name in enumerate(tqdm(image_names, desc="VITON")):
        image_path = os.path.join(image_dir, img_name)
        # The following replacement rules are based on your dataset naming; adjust if necessary:
        mask1_path = os.path.join(mask1_dir, img_name.replace(".jpg", "_mask.png"))
        parse_path = os.path.join(parse_dir, img_name.replace(".jpg", ".png"))
        dense_path = os.path.join(dense_dir, img_name.replace(".jpg", ".png"))

        if not (
            os.path.exists(mask1_path)
            and os.path.exists(parse_path)
            and os.path.exists(dense_path)
        ):
            # skip samples where corresponding files are missing
            continue

        agnostic_out_path = os.path.join(agnostic_dir, img_name)
        mask_out_path = os.path.join(
            mask_dir, img_name.replace("_mask.jpg", "_mask.png")
        )

        generate_viton_agnostic_and_mask(
            image_path,
            mask1_path,
            parse_path,
            dense_path,
            agnostic_out_path,
            mask_out_path,
            dilation_radius1=dilation_radius1,
            dilation_radius2=dilation_radius2,
        )


if __name__ == "__main__":
    dir_path = "../../datasets/DressCode"
    "dresses"
    "upper_body"
    "lower_body"
    for subset in ["lower_body"]:
        batch_generate_agnostic_and_mask_dresscode(
            os.path.join(dir_path, subset),
            subset=subset,
            dilation_radius1=15,
            dilation_radius2=5,
            dilation_radius0=66,
        )
