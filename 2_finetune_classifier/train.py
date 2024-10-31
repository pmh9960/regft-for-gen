import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from src.args import parse_arguments
from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageClassifier, ImageEncoder
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier, gpt_clip_classifier


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


def wise_ft(args):
    ### fix seed ###
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    assert args.save is not None, 'Please provide a path to store models'

    if args.load is None:
        # Build and save zero-shot model
        image_encoder = ImageEncoder(args, keep_lang=True)
        if args.gpt_prompt_file == 'none':
            classification_head = get_zeroshot_classifier(args, image_encoder.model)
        else:
            classification_head = gpt_clip_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)

        if args.zeroshot_checkpoint is None:
            zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
            classifier.save(zeroshot_checkpoint)
            args.zeroshot_checkpoint = zeroshot_checkpoint
        else:
            zeroshot_checkpoint = args.zeroshot_checkpoint
            if not os.path.exists(zeroshot_checkpoint):
                classifier.save(zeroshot_checkpoint)

        # Standard fine-tuning
        args.load = zeroshot_checkpoint
        finetuned_checkpoint = finetune(args)
    else:
        # No need to compute things from stratch
        assert len(args.load) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    zeroshot.process_images = True
    finetuned.process_images = True
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    if args.fisher is None:
        fishers = None
    else:
        fisher_0_file, fisher_1_file = args.fisher
        fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
        fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
        fishers = fisher_0, fisher_1

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # # save model
        # finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    cfg = OmegaConf.to_yaml(vars(args))
    if os.path.exists(os.path.join(args.save, 'config.yml')) and not args.overwrite:
        raise FileExistsError('Config file already exists\n' + os.path.join(args.save, 'config.yml'))
    os.makedirs(args.save, exist_ok=True)
    with open(os.path.join(args.save, 'config.yml'), 'w') as f:
        f.write(cfg)
    wise_ft(args)
