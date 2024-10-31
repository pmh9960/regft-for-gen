import copy
import os
import time

import _clip.clip as clip
import src.datasets as datasets
import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, ImageClassifier, ImageEncoder
from src.models.utils import LabelSmoothing, cosine_lr, get_model_weights, torch_load, update_model_weights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def freeze_BN(model):
    for module in model.modules():
        # print(module)
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    return model


def finetune(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."

    writer = SummaryWriter(log_dir=os.path.join(args.save, 'runs'))

    image_classifier = ImageClassifier.load(args.load)
    for param in image_classifier.parameters():
        param.requires_grad = True

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        # print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        # print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_batches = len(dataset.train_loader)

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)

    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', num_params)

    if args.scheduler == 'cosine':
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    elif args.scheduler == 'none':
        def scheduler(step): return None
    else:
        raise NotImplementedError(args.scheduler)

    if args.freeze_encoder:
        image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
    else:
        image_classifier = model.module

    # Evaluate
    eval_results = evaluate(image_classifier, args)
    for eval_dataset in args.eval_datasets:
        writer.add_scalar(f'{eval_dataset}:top1', eval_results[f'{eval_dataset}:top1'], 0)

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_bn:
            model = freeze_BN(model)  # ! freeze batch norm
        data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=image_enc)

        pbar = tqdm(total=len(data_loader), desc=f'Epoch {epoch}/{args.epochs}', ncols=80)

        for i, batch in enumerate(data_loader):
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()

            logits = model(inputs)

            loss = 0.

            ce_loss = loss_fn(logits, labels)
            writer.add_scalar('ce_loss', ce_loss.item(), step)
            loss += ce_loss * args.lambda_ce

            if sum(args.vc_reg) > 0:
                vc_loss = image_classifier.vc_reg(*args.vc_reg)
                loss += vc_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            pbar.update(1)
            pbar.set_postfix({'loss': loss.item()})

            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

        pbar.close()

        if args.freeze_encoder:
            image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
        else:
            image_classifier = model.module

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            print('Saving model to', os.path.join(args.save, 'checkpoint_latest.pt'))
            image_classifier.save(os.path.join(args.save, 'checkpoint_latest.pt'))
            # torch.save(image_classifier.state_dict(), os.path.join(args.save, 'checkpoint_latest.pt'))
            # torch.save(optimizer.state_dict(), os.path.join(args.save, 'optim_latest.pt'))
            if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
                print('Saving model to', os.path.join(args.save, f'checkpoint_{epoch+1}.pt'))
                image_classifier.save(os.path.join(args.save, f'checkpoint_{epoch+1}.pt'))
                # torch.save(image_classifier.state_dict(), os.path.join(args.save, f'checkpoint_{epoch+1}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.save, f'optim_{epoch+1}.pt'))

        # Evaluate
        args.current_epoch = epoch
        eval_results = evaluate(image_classifier, args)
        for eval_dataset in args.eval_datasets:
            writer.add_scalar(f'{eval_dataset}:top1', eval_results[f'{eval_dataset}:top1'], step)

    writer.close()

    # Saving last model
    if args.save is not None:
        return os.path.join(args.save, 'checkpoint_latest.pt')


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
