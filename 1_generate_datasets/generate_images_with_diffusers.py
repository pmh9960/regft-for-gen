import argparse
import json
import os
import random

import torch
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import diffusers
from diffusers.pipelines import AutoPipelineForText2Image


class PromptDataset(Dataset):
    def __init__(self, dataset_name, n_iter, prompt_style_file='data/prompt_styles.json'):
        super().__init__()
        with open(prompt_style_file, 'r') as f:
            prompt_styles = json.load(f)
        self.classnames = prompt_styles[dataset_name]['classnames']
        self.prompt = prompt_styles[dataset_name]['template']
        self.save_subdir = prompt_styles[dataset_name]['save_subdir'] if dataset_name == 'imagenet' else prompt_styles[dataset_name]['classnames']
        self.n_iter = n_iter

    def __len__(self):
        return len(self.classnames) * self.n_iter

    def __getitem__(self, idx):
        class_id = idx % len(self.classnames)
        prompt = self.prompt.format(self.classnames[class_id])
        return dict(
            name=self.classnames[class_id],
            save_path=self.save_subdir[class_id],
            prompt=prompt,
            class_id=class_id,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--n_iter', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='outputs/imagenet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--prompt_style_file', type=str, default='data/prompt_styles.json')
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.overwrite and os.path.exists(os.path.join(args.output_dir, f'{args.seed:04d}.yaml')):
        raise FileExistsError(f'File {os.path.join(args.output_dir, f"{args.seed:04d}.yaml")} already exists')
    os.makedirs(args.output_dir, exist_ok=True)
    OmegaConf.save(OmegaConf.create(vars(args)), os.path.join(args.output_dir, f'{args.seed:04d}.yaml'))

    seed_everything(args.seed)
    dataset = PromptDataset(args.dataset, args.n_iter, args.prompt_style_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    pipe = AutoPipelineForText2Image.from_pretrained(
        args.ckpt,
        revision=args.revision,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        add_watermarker=False,
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)

    pbar = tqdm(total=len(dataloader), desc='Generating images')
    for i, batch in enumerate(dataloader):
        # name = batch['name']
        save_paths = batch['save_path']
        prompt = batch['prompt']
        # class_id = batch['class_id']

        images = pipe(
            prompt,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
        ).images

        for idx, img in enumerate(images):
            os.makedirs(os.path.join(args.output_dir, save_paths[idx]), exist_ok=True)
            idx_cur = len([name for name in os.listdir(os.path.join(args.output_dir, save_paths[idx])) if name.startswith(f'{args.seed:04d}_')])
            save_path = os.path.join(args.output_dir, save_paths[idx], f'{args.seed:04d}_{idx_cur:04d}.png')
            img = img.resize((224, 224), Image.BICUBIC)
            img.save(save_path)
        pbar.update(1)


if __name__ == '__main__':
    main()
