import logging
import re

import torch
from tqdm import tqdm
import os

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.

        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def camel_to_words(name: str) -> str:
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    spaced = spaced.replace('_', ' ').replace('-', ' ')
    return spaced

def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if ('imagenet-val' not in data) and ('imagenet-v2' not in data) and ('zero-shot-val' not in data):
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot evaluation.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier for available datasets')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    imagenet_classifier = None
    with autocast():
        if ('imagenet-val' in data) or ('imagenet-v2' in data):
            imagenet_classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )

    logging.info('Using classifier(s)')
    results = {}
    if 'imagenet-val' in data and imagenet_classifier is not None:
        top1, top5 = run(model, imagenet_classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data and imagenet_classifier is not None:
        top1, top5 = run(model, imagenet_classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    # Generic zero-shot from ImageFolder dataset
    if 'zero-shot-val' in data:
        zs_dataset = getattr(data['zero-shot-val'].dataloader, 'dataset', None)
        zs_classnames = None
        dataset_name = None
        if zs_dataset is not None and hasattr(zs_dataset, 'root'):
            dataset_name = os.path.basename(os.path.normpath(zs_dataset.root))

        if dataset_name in ['AID', 'RESISC45', 'EuroSAT_RGB', 'RS2800', 'RSI-CB128', 'RSI-CB256']: 
            if zs_dataset is not None and hasattr(zs_dataset, 'classes'):
                zs_classnames = [camel_to_words(c) for c in list(zs_dataset.classes)]
        # elif dataset_name in ['RSI-CB128', 'RSI-CB256']:   
        #     root = getattr(zs_dataset, 'root', None)
        #     if root is not None:
        #         subfolders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        #         second_level = []
        #         for sf in subfolders:
        #             second_level += [d for d in os.listdir(sf) if os.path.isdir(os.path.join(sf, d))]
        #         zs_classnames = [camel_to_words(c) for c in sorted(set(second_level))]

        # print(f"zs_classnames={zs_classnames}")
        
        if zs_classnames:
            with autocast():
                zs_classifier = build_zero_shot_classifier(
                    model,
                    tokenizer=tokenizer,
                    classnames=zs_classnames,
                    templates=OPENAI_IMAGENET_TEMPLATES,
                    num_classes_per_batch=10,
                    device=device,
                    use_tqdm=True,
                )
            print(f"zs_classifier={zs_classifier.shape}")

            top1, top5 = run(model, zs_classifier, data['zero-shot-val'].dataloader, args)
            results['zeroshot-val-top1'] = top1
            results['zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot evaluation.')

    return results
