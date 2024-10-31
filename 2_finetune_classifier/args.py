import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
             " Note that same model used for all datasets, so much have same classnames"
             "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=(
            'Interpolation coefficient for ensembling. '
            'Users should specify N-1 values, where N is the number of '
            'models being ensembled. The specified numbers should sum to '
            'less than 1. Note that the order of these values matter, and '
            'should be the same as the order of the classifiers being ensembled.'
        )
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )
    parser.add_argument(
        "--x-dataset",
        type=str,
        default=None,
        help="x-axis dataset for scatter plot",
    )
    parser.add_argument(
        '--zeroshot-checkpoint',
        type=str,
        default=None,
        help='Path to a checkpoint to use for zero-shot classification.',
    )
    parser.add_argument(
        '--save-freq',
        type=int,
        default=1,
        help='How often to save checkpoints.',
    )
    parser.add_argument('--gpt_prompt_file', type=str, default='none', help='Path to a file containing GPT prompts.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vc_reg', type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument('--lambda-ce', type=float, default=1.0)
    parser.add_argument('--freeze-bn', action='store_true')
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert len(parsed_args.reset_layers) in (0, 2), \
        f"Please specify init and end reset layers or nothing: {len(parsed_args.reset_layers)}"
    assert len(parsed_args.freeze_layers) in (0, 2), \
        f"Please specify init and end freeze layers or nothing: {len(parsed_args.freeze_layers)}"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args