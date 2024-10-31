import json
import os

import _clip.clip as clip
import numpy as np
import src.datasets as datasets
import src.templates as templates
import torch
from src.args import parse_arguments
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageClassifier, ImageEncoder
from tqdm import tqdm


def gpt_clip_classifier(args, clip_model):
    assert args.template is not None
    assert args.train_dataset is not None
    # template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        classnames=args.classnames,
        num_workers=args.num_workers,
    )
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with open(args.gpt_prompt_file) as f:
        gpt_prompts = json.load(f)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames, desc='generate clip weights', ncols=80):
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        zeroshot_weights *= logit_scale.exp()

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_zeroshot_classifier(args, clip_model):
    assert args.template is not None
    assert args.train_dataset is not None
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        classnames=args.classnames,
        num_workers=args.num_workers,
    )
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.classnames, ncols=80):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device)  # tokenize
            embeddings = clip_model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def eval(args):
    args.freeze_encoder = True
    if args.load is not None:
        classifier = ImageClassifier.load(args.load)
    else:
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)

    evaluate(classifier, args)

    if args.save is not None:
        classifier.save(args.save)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)
