import sys

sys.path.append('wise_ft')

import os

import jsonlines
import pandas as pd
import torch

from src.models.modeling import ImageClassifier


def _merge(alpha, theta_0, theta_1):
    return {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }


def find_best_alpha(results_jsonl):
    with jsonlines.open(results_jsonl, 'r') as f:
        results = list(f.iter())
    results = [r for r in results if isinstance(r['alpha'], float)]
    top1_key = [key for key in results[0].keys() if 'top1' in key][0]
    results.sort(key=lambda x: x[top1_key], reverse=True)
    return results[0]['alpha']


def mixing_weights(zeroshot_checkpoint, finetuned_checkpoint, alpha_mixing='best'):
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)

    if alpha_mixing == 'best':
        alpha = find_best_alpha(results_jsonl=os.path.join(os.path.dirname(finetuned_checkpoint), 'results.jsonl'))
        alpha_disp = f'best_alpha={alpha:.3f}'
    elif isinstance(alpha_mixing, float):
        alpha = alpha_mixing
        alpha_disp = f'alpha={alpha:.3f}'
    else:
        raise ValueError(f'alpha_mixing={alpha_mixing} is not supported')

    if os.path.isfile(os.path.join(os.path.dirname(finetuned_checkpoint), f'wise_ft_{alpha_disp}.pt')):
        print(f'File exists: {os.path.join(os.path.dirname(finetuned_checkpoint), f"wise_ft_{alpha_disp}.pt")}')
        return

    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    theta = _merge(alpha, theta_0, theta_1)

    finetuned.load_state_dict(theta)
    torch.save({
        'clip': finetuned.image_encoder.model.state_dict(),
        'head': (finetuned.classification_head.weight.data, finetuned.classification_head.bias.data)
    },
        os.path.join(os.path.dirname(finetuned_checkpoint), f'wise_ft_{alpha_disp}.pt'))
    print(os.path.join(os.path.dirname(finetuned_checkpoint), f'wise_ft_{alpha_disp}.pt'))
