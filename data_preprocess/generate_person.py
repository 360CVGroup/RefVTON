import torch, os
from diffusers import FluxKontextPipelineI2I, FluxKontextPipeline
from diffusers.utils import load_image
from train_lora_flux_kontext_1st_stage import viton_collate_fn
from datasets_util.datasets_loader import (
    VITONDataset,
    DressCodeDataset,
    CombinedDataset,
)
from PIL import Image
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.distributed as dist


def save_tensor_as_png(tensor, filename="visualize"):
    tensor = tensor.detach().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    array = tensor.float().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(filename)


def main(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # -------------------- Dataset --------------------

    if args.split == "train":
        train = True
    elif args.split == "test":
        train = False
    elif args.split == "all":
        train = None

    if "viton" in args.instance_data_dir:
        dataset = VITONDataset(
            args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            scale=args.cond_scale,
            size=(args.resolution, int(0.75 * args.resolution)),
            train=train
        )
    elif "DressCode" in args.instance_data_dir:
        dataset = DressCodeDataset(
            args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=(args.resolution, int(0.75 * args.resolution)),
            scale=args.cond_scale,
            train=train,
            use_different=False,
        )
    else:
        dataset = CombinedDataset(
            viton_root=os.path.join(args.instance_data_dir, "viton"),
            dresscode_root=os.path.join(args.instance_data_dir, "DressCode"),
            size=(args.resolution, int(0.75 * args.resolution)),
            instance_prompt=args.instance_prompt,
            scale=args.cond_scale,
            train=train,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,  
        collate_fn=lambda examples: viton_collate_fn(examples),
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )

    # -------------------- Pipeline --------------------
    pipe = FluxKontextPipelineI2I.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(args.output_dir)

    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    pipe.to(accelerator.device)

    if accelerator.is_main_process:
        print(f"Device: {pipe.device}, Total batches: {len(dataloader)}")

    total_generated_images = 0
    save_dir = "visualize"
    result_dir = "samples"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{args.instance_data_dir}/reference", exist_ok=True)

    if args.use_person:
        key_to_index_scale = {
            "cond_pixel_values_person": [1, 1],
            "cond_pixel_values_cloth": [2, 1],
            "cond_pixel_values_agnostic_mask": [3, args.cond_scale],
            "cond_pixel_values_dense": [4, args.cond_scale],
            "pixel_values_ref": [5, args.cond_scale],
        }
    else:
        key_to_index_scale = {
            "cond_pixel_values_agnostic": [1, 1],
            "cond_pixel_values_cloth": [2, 1],
            "cond_pixel_values_agnostic_mask": [3, args.cond_scale],
            "cond_pixel_values_dense": [4, args.cond_scale],
            "pixel_values_ref": [5, args.cond_scale],
        }
    accelerator.wait_for_everyone()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f"generating...")):

            images = pipe(
                image=batch,
                batch_size=len(batch["cond_pixel_values_cloth"]),
                prompt=args.instance_prompt,
                num_images_per_prompt=1,
                guidance_scale=2.5,
                generator=torch.Generator().manual_seed(42),
                height=args.resolution,
                width=int(0.75 * args.resolution),
                cond_scale=args.cond_scale,
                key_to_index_scale=key_to_index_scale,
            ).images
            accelerator.wait_for_everyone()

            gt_image = accelerator.gather(batch["pixel_values"]).to("cpu")
            cloth_image = accelerator.gather(batch["cond_pixel_values_cloth"]).to("cpu")
            image_ref = accelerator.gather(batch["pixel_values_ref"]).to("cpu")
            images = accelerator.gather(images).to("cpu")

            index = batch["index"]
            gathered_index = index
            category = batch["category"]
            gathered_category = category
            if torch.distributed.is_initialized():
                gathered_index = [None for _ in range(dist.get_world_size())]
                gathered_category = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_index, index)
                dist.all_gather_object(gathered_category, category)

                gathered_index = [
                    item for sublist in gathered_index for item in sublist
                ]
                gathered_category = [
                    item for sublist in gathered_category for item in sublist
                ]

            for i in range(len(images)):
                if not os.path.exists(f"samples/{gathered_category[i]}"):
                    os.makedirs(
                        f"samples/{gathered_category[i]}",
                        exist_ok=True,
                    )

                save_tensor_as_png(
                    images[i],
                    f"samples/{gathered_category[i]}/{gathered_index[i]}",
                )
                total_generated_images = total_generated_images + 1

            if accelerator.is_main_process:
                print(
                    f"----------------Generated {total_generated_images} images with shape of {list(images.shape)}----------------"
                )


if __name__ == "__main__":
    from argparser import parse_args

    args = parse_args()
    main(args)
