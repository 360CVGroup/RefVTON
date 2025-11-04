
  

# End2End Virtual Tryon with Visual Reference

  

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-EVTAR-ff9900?style=flat)](https://huggingface.co/qihoo360/EVTAR) [![arXiv](https://img.shields.io/badge/arXiv-2101.00001-B31B1B?style=flat)](https://arxiv.org/abs/2511.00956)

  

![examples](assets/examples.png)

  

  

We propose **EVTAR**, an End-to-End Virtual Try-on model with Additional Visual Reference, that directly fits the target garment onto the person image while incorporating reference images to enhance the model's ability to preserve and accurately depict clothing details.

  

  

  

## 💡 Update

  

  

- [x] [2025.10.11] Release the virtual try-on inference code and LoRA weights.

  

  

- [x] [2025.10.13] Release the technical report on Arxiv.

  

  

  

## 💪 Highlight Feature

  

  

  

-  **And End-To-End virtual try-on model:** Can function either as an inpainting model for placing the target clothing into masked areas, or as a direct garment transfer onto the human body.

  

  

-  **Using Reference Image To Enhance the Try-on Performance:** To emulate human attention on the overall wearing effect rather than the garment itself when shopping online, our model allows using images of a model wearing the target clothing as input, thereby better preserving its material texture and design details.

  

  

-  **Improved Performance** Our model achieves state-of-the-art performance on public benchmarks and demonstrates strong generalization ability to in-the-wild inputs.

  

  

  

## 🧩 Environment Setup

  

  

  

```
conda create -n EVTAR python=3.12 -y
conda activate EVTAR
pip install -r requirements.txt
```

  

  

  

## 📂 Preparation of Dataset and Pretrained Models

  

  

  

### Dataset

  

  

  

Currently, we provide a small test set with reference images for trying our model. We plan to release the reference data generation code, along with our proposed full dataset containing model reference images, in the future.

  

  

Nevertheless, inference can still be performed in a reference-free setting on public benchmarks, including [VITON-HD](https://github.com/shadow2496/VITON-HD) and [DressCode](https://github.com/aimagelab/dress-code).

  

  

### Reference Data Preparation

  

  

One key feature of our method is the use of _reference data_, where an image of a different person wearing the target garment is provided to help the model imagine how the target person would look in that garment. In most online shopping applications, such reference images are commonly used by customers to better visualize the clothing. However, publicly available datasets such as VITON-HD and DressCode do not include such reference data, so we generate them ourselves.

  

  

  

Please prepare the pretrained weights of the Flux-Kontext model and the Qwen2.5-VL-32B model. And you can generate the reference image using the following commands:

  

  

```
accelerate launch --num_processes 8 --main_process_port 29500 generate_reference.py \
--instance_data_dir "path_to_your_datasets" \
--inference_batch_size 1 \
--split "train" \
--desc_path "desc.json"
```

  

  

  

### Pretrained Models

  

  

We provide pretrained backbone networks and LoRA weights for testing and deployment. Please download the `.safetensors` files from [here] and place them in the `checkpoints` directory.

  

  

  

## ⏳ Inference Pipeline

  

  

  

Here we provide the inference code for our EVTAR.

  

  

```
accelerate launch --num_processes 8 --main_process_port 29500 inference.py \

--pretrained_model_name_or_path="[path_to_your_Flux_model]" \
--instance_data_dir="[your_data_directory]" \
--output_dir="[Path_to_LoRA_weights]" \
--mixed_precision="bf16" \
--split="test" \
--height=1024 \
--width=768 \
--inference_batch_size=1 \
--cond_scale=2 \
--seed="0" \
--use_reference \
--use_different \
--use_person
```

  

  

-  `pretrained_model_name_or_path`: Path to the downloaded Flux-Kontext model weights.

  

  

-  `instance_data_dir`: Path to your dataset. For inference on VITON-HD or DressCode, ensure that the words "viton" or "DressCode" appear in the path.

  

  

-  `output_dir`: Path to the downloaded or trained LoRA weights.

  

  

-  `cond_scale`: Resize scale of the reference image during training. Defaults to `1.0` for $512\times384$ and `2.0` for $1024\times768$ resolution.

  

  

-  `use_reference`: Whether to use a reference model image.

  

  

-  `use_different`: **Only applicable for VITON/DressCode inference.** Whether to use different cloth-person pairs.

  

  

-  `use_person`: **Only applicable for VITON/DressCode inference.** Whether to use the unmasked person image instead of the agnostic masked image as input for the virtual try-on task.

  

  

## 📊 Evaluation

  

  

We quantitatively evaluate the quality of virtual try-on results using the FID, KID, SSIM, and LPIPS. Here, we provide the evaluation code for the VITON-HD and DressCode datasets.

  

```
# Evaluation on VITON-HD dataset

CUDA_VISIBLE_DEVICES=0 python eval_dresscode.py \

--gt_folder_base [path_to_your_ground_truth_image_folder] \
--pred_folder_base [[path_to_your_generated_image_folder]]\
--paired
```

  

  

  

```
# Evaluation on DressCode dataset

CUDA_VISIBLE_DEVICES=0 python eval.py \

--gt_folder_base [path_to_your_ground_truth_image_folder] \
--pred_folder_base [[path_to_your_generated_image_folder]]\
```

  

  

-  `paired`: If you perform unpaired generation, where different garments are fitted onto the target person, you should enable this flag during evaluation.

  
  

Evaluation result on VITON-HD dataset:

![examples](assets/VITON_results.png)

  
  

Evaluation result on DressCode dataset:

![examples](assets/DressCode_results.png)

  

## 🌸 Acknowledgement

  

  

This code is mainly built upon [Diffusers](https://github.com/huggingface/diffusers/tree/main), [Flux](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/flux), and [CatVTON](https://github.com/Zheng-Chong/CatVTON/) repositories. Thanks so much for their solid work!

  

  

  

## 💖 Citation


If you find this repository useful, please consider citing our paper:
```
@misc{li2025evtarendtoendtryadditional,
title={EVTAR: End-to-End Try on with Additional Unpaired Visual Reference},
author={Liuzhuozheng Li and Yue Gong and Shanyuan Liu and Bo Cheng and Yuhang Ma and Liebucha Wu and Dengyang Jiang and Zanyi Wang and Dawei Leng and Yuhui Yin},
year={2025},
eprint={2511.00956},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2511.00956},
}
```