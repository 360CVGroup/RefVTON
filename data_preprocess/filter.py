import torch, json, os, random
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from torchvision import transforms
from transformers import pipeline
from tqdm import tqdm

"Is the person in the picture too close to the camera, resulting in only upper body being visible? Must answer with 'Yes' or 'No'"
"Is the person in the picture facing away from the camera? Must answer with 'Yes' or 'No'"
"Look at the image and determine: Is the person wearing layered clothing, where an outer garment (such as a coat, cardigan, or jacket) is open so that the inner clothing is clearly visible?"


def main(args):
    pipe_qwen = pipeline(
        task="image-text-to-text",
        model="../../pretrained_models/Qwen2.5-VL-32B-Instruct/",
        torch_dtype=torch.bfloat16,
        device=0,
    )

    path = args.instance_data_dir
    path_list = sorted(os.listdir(f"{path}/images"))[:5000]

    for i, name in enumerate(tqdm(path_list)):
        # print(name, path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": f"{path}/images/{str(name)}",
                    },
                    {
                        "type": "text",
                        "text": "Is the model in the photo in profile? Answer the question with only Yes or No.",
                    },
                ],
            }
        ]

        x = pipe_qwen(text=messages, max_new_tokens=128, return_full_text=False)
        result = x[-1]["generated_text"]
        if "Yes" in x[-1]["generated_text"]:
            if os.path.exists(f"{path}/images/{str(name)}"):
                os.remove(f"{path}/images/{str(name)}")
            print(
                f"--------------------------remove {path}/cloth/{str(name)}---------------------------"
            )
        print(x[-1]["generated_text"], name, "Yes" in x[-1]["generated_text"])


if __name__ == "__main__":
    from argparser import parse_args

    args = parse_args()
    main(args)
