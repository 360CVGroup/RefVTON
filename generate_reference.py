from pathlib import Path
import torch, json, random, os
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import FluxKontextPipeline
from tqdm.auto import tqdm
from transformers import pipeline
from datasets_util.datasets_PIL import (
    VITONDataset_PIL,
    DressCodeDataset_PIL,
    ViViDDataset_PIL,
)

TEXT_PROMPT = {
    "viton": [
        'Describe the appearance of this model in one sentence (start with Positive:) focusing only on race, skin tone, hair color, length or style, facial expression, and eye color. Only describe physical traits, never describe clothing, outfits, or accessories, start with "The model". Then provide an opposite appearance in one sentence (start with Negative:) by changing race, skin tone, hair color, length or style, facial expression, and eye color, without using "not". The answer must be short and avoid speculative words such as "could" or "might".',
        'Describe the appearance of this model in one sentence (start with Positive:) focusing only on hair color, length or style, facial expression, and eye color. Only describe physical traits, never describe clothing, outfits, or accessories, start with "The model". Then provide an opposite appearance in one sentence (start with Negative:) by changing, hair color, length or style, facial expression, and eye color, without using "not". The answer must be short and avoid speculative words such as "could" or "might".',
    ],
    "DressCode": [
        "Describe the gender (start with Gender:) as man or woman, and then describe the appearance of the man/woman in one sentence (start with Positive:) focusing only on the looking directions(looking to the left or right), hair color and style(if gender is man, do not descirbe the hair is long or short!), race and skin tone. Only describe physical traits, never describe clothing, outfits, or accessories. Then provide an opposite appearance in one sentence (start with Negative:) by changing the looking direction(looking to the left or right), hair color or style(if gender is man, do not descirbe the hair is long or short!), race and skin tone, while keeping the same gender as in Positive. Do not use 'not'. The answer must be short and avoid speculative words such as 'could' or 'might'.",
        "Describe the gender (start with Gender:) as man or woman, and then describe the appearance of the man/woman in one sentence (start with Positive:) focusing only on the looking directions(looking to the left or right), hair color and style(if gender is man, do not descirbe the hair is long or short!). Only describe physical traits, never describe clothing, outfits, or accessories. Then provide an opposite appearance in one sentence (start with Negative:) by changing the looking direction(looking to the left or right), hair color or style(if gender is man, do not descirbe the hair is long or short!), while keeping the same gender as in Positive. Do not use 'not'. The answer must be short and avoid speculative words such as 'could' or 'might'.",
    ],
}


def split_positive_negative(text: str):
    positive, negative = "", ""
    if "Positive:" in text and "Negative:" in text:
        parts = text.split("Negative:")
        pos_part = parts[0].replace("Positive:", "").strip()
        neg_part = parts[1].strip()
        positive = pos_part
        negative = neg_part
    return positive, negative


def get_prompt(pipe_qwen, image_path, desc_list, accelerator, dataset_name, rng):

    descriptions = rng.choice(desc_list)
    prompts_2 = descriptions["action"] + descriptions["outfit"]
    text_prompt = rng.choices(TEXT_PROMPT[dataset_name], weights=[0.75, 0.25])[0]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "path": str(image_path),
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    x = pipe_qwen(text=messages, max_new_tokens=100, return_full_text=False)
    negative_prompt, positive_prompt = split_positive_negative(x[-1]["generated_text"])

    prompts_2 = positive_prompt + prompts_2
    prompts = positive_prompt

    return prompts, prompts_2, negative_prompt


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    pipe_qwen = pipeline(
        task="image-text-to-text",
        model="../../pretrained_models/Qwen2.5-VL-32B-Instruct/",
        device_map={"": accelerator.device.index},
        torch_dtype=torch.bfloat16,
    )
    dtype = torch.bfloat16
    pipe = FluxKontextPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    ).to(device)

    with open("desc.json", "r") as f:
        desc_dict = json.load(f)

    dataset_root = Path(args.instance_data_dir)

    # -------- Dataset & Dataloader --------
    batch_size = args.inference_batch_size
    if "viton" in args.instance_data_dir or "FasionTryOn" in args.instance_data_dir:
        if args.split == "test":
            train = False
        elif args.split == "train":
            train = True
        elif args.split == "all":
            train = None
        dataset = VITONDataset_PIL(dataset_root, train=train)
        desc_list = desc_dict["viton"]
        dataset_name = "viton"

    elif "DressCode" in args.instance_data_dir:
        dataset = DressCodeDataset_PIL(dataset_root, categories=("upper_body"))
        desc_list = desc_dict["DressCode"]
        dataset_name = "DressCode"

    elif "ViViD" in args.instance_data_dir:
        dataset = ViViDDataset_PIL(dataset_root, categories=("upper_body"))
        desc_list = desc_dict["DressCode"]
        dataset_name = "DressCode"
    elif "FasionTryOn" in args.instance_data_dir:
        dataset = VITONDataset_PIL(dataset_root, train=train)
        desc_list = desc_dict["FasionTryOn"]
        dataset_name = "viton"
    elif "IGPair" in args.instance_data_dir:
        dataset = ViViDDataset_PIL(dataset_root, categories=("upper_body"))
        desc_list = desc_dict["DressCode"]
        dataset_name = "DressCode"

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    pipe_qwen, pipe, dataloader = accelerator.prepare(pipe_qwen, pipe, dataloader)

    if accelerator.is_main_process:
        pbar = tqdm(total=len(dataloader), desc="Generating images")
    else:
        pbar = None

    accelerator.wait_for_everyone()
    rng = random.Random(1234 + accelerator.process_index)
    for batch, names, subsets in dataloader:
        height, width = list(batch.shape)[-2:]
        name = names[0]
        subset = subsets[0]
        descriptions = desc_list
        if dataset_name == "DressCode":
            descriptions = desc_list[subset]

        prompt, prompt_2, negativa_prompt = get_prompt(
            pipe_qwen,
            dataset_root / subset / "new_image" / name,
            descriptions,
            accelerator,
            dataset_name=dataset_name,
            rng=rng,
        )

        imgs = pipe(
            image=batch,
            prompt_2=prompt_2
            + "And translucent clothing needs to maintain a consistent skin color.",
            prompt=prompt
            + "And translucent clothing needs to maintain a consistent skin color.",
            negative_prompt=negativa_prompt,
            guidance_scale=2.5,
            num_inference_steps=28,
            height=height,
            width=width,
        ).images

        accelerator.wait_for_everyone()

        os.makedirs(
            os.path.join(args.instance_data_dir, subset, "image_ref"),
            exist_ok=True,
        )
        out_path = dataset_root / subset / "image_ref" / name
        img = imgs[0].resize([width, height])
        img.save(out_path)

        if pbar is not None:
            pbar.update(1)


if __name__ == "__main__":
    from argparser import parse_args

    args = parse_args()
    main(args)
