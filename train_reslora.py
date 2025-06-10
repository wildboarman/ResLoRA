import argparse
import datetime
import time
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler

from timm.utils import NativeScaler
from lib.datasets import build_dataset
from engine import *
from lib.samplers import RASampler

from model.vision_transformer_timm import VisionTransformerSepQKV

import model as models
from timm.models import load_checkpoint

try:
    #SLUMRM
    from mmcv.runner import init_dist
except ModuleNotFoundError as e:
    print(f'{e}. Cannot use multiple-node training...')

import os
from timm.utils.clip_grad import dispatch_clip_grad
from collections import OrderedDict
from lib import utils
import time
from lib.utils import save_to_csv


def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--model_name', required=True, type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    # AutoFormer config
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'vp','retrain','search'], help='mode of AutoFormer')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='path to the pre-trained model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--val_interval', default=1, type=int, help='validataion interval')
    parser.add_argument('--inception',action='store_true')
    parser.add_argument('--direct_resize',action='store_true')

    # reslora params
    parser.add_argument('--freeze_stage', action='store_true')
    parser.add_argument('--sensitivity_path', default='', type=str,)
    parser.add_argument('--scaler', default='naive', type=str,)
    parser.add_argument('--low_rank_dim', default=8, type=int, help='The rank of Adapter or LoRA')
    parser.add_argument('--alpha', default=10., type=float, help='hyper-parameter, the easiness level for a matrix to be structurally tuned.')
    parser.add_argument('--beta', default=5., type=float, help='hyper-parameter, the easiness level for a vector to be structurally tuned.')

    parser.add_argument('--test', action='store_true', help='using test-split or validation split')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--block', type=str, default='BlockSPTParallel')
    parser.add_argument('--get_sensitivity', action='store_true')
    parser.add_argument('--structured_vector', action='store_true', help='trick to also tune vectors structually, directly tune all parameters of these vectors')
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--freeze_kwd', default='patch_embed', type=str, help='freeze patch embedding helps')
    parser.add_argument('--structured_type', type=str, default='lora', help="structured tuning module to use")

    parser.add_argument('--no_structured_drop_out', action='store_true')
    parser.add_argument('--no_structured_drop_path', action='store_true')
    parser.add_argument('--structured_only', action='store_true', help="structured tuning module to use")
    parser.add_argument('--sensitivity_batch_num', default=16, type=int,)
    parser.add_argument('--local_rank', default=0, type=int,)

    return parser


def main(args):

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        init_dist(launcher=args.launcher)
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args,)
    dataset_val, _ = build_dataset(is_train=False, args=args,)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    print(f"{args.data_set} dataset, train: {len(dataset_train)}, evaluation: {len(dataset_val)}")

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    print('mixup_active',mixup_active)
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.get_sensitivity:

        # Always getting sensitivity from the training split
        dataset_sensitivity, _ = build_dataset(is_train=True, args=args, )
        sampler_init = torch.utils.data.SequentialSampler(dataset_sensitivity)

        data_loader_sensitivity = torch.utils.data.DataLoader(
            dataset_sensitivity, sampler=sampler_init,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        model = models.__dict__[args.model_name](img_size=args.input_size,
                                                drop_rate=args.drop,
                                                drop_path_rate=args.drop_path,
                                                freeze_backbone=args.freeze_stage,
                                                num_classes=args.nb_classes
                                                )
    else:
        param_info = torch.load(args.sensitivity_path, map_location='cpu')
        tuned_vectors = param_info['tuned_vectors']
        tuned_matrices = param_info['tuned_matrices']
        tuned_matrices_rank = param_info['tuned_matrices_rank']

        print('Both structured and unstructured tuning', )

        fully_fine_tuned_keys = []
        fully_fine_tuned_keys.extend(tuned_vectors)
        fully_fine_tuned_keys.extend(['head.weight', 'head.bias', 'cls_token'])

        # Setting up unstructured tuning
        unstructured_name_shapes = param_info['unstructured_name_shapes']
        unstructured_indexes = param_info['unstructured_indexes']
        unstructured_params = param_info['unstructured_params']

        #structured_low_ranks = param_info['structured_low_ranks']

        if unstructured_params == 0:
            grad_mask = None
        else:

            grad_mask = torch.cat(
                [torch.zeros(unstructured_name_shapes[key]).flatten() for key in unstructured_name_shapes.keys()])
            grad_mask[unstructured_indexes] = 1.
            grad_mask = grad_mask.split([np.cumprod(list(shape))[-1] for shape in unstructured_name_shapes.values()])
            grad_mask = {k: (mask.view(v) != 0).nonzero() for mask, (k, v) in
                         zip(grad_mask, unstructured_name_shapes.items())}

        model = models.__dict__[args.model_name](img_size=args.input_size,
                                                 drop_rate=args.drop,
                                                 drop_path_rate=args.drop_path,
                                                 freeze_backbone=args.freeze_stage,
                                                 structured_list=tuned_matrices,
                                                 structured_list_rank=tuned_matrices_rank,
                                                 tuned_vectors=tuned_vectors,
                                                 low_rank_dim=args.low_rank_dim,
                                                 block=args.block,
                                                 num_classes=args.nb_classes,
                                                 structured_type=args.structured_type,
                                                 structured_bias=args.structured_vector,
                                                 unstructured_indexes=grad_mask,
                                                 unstructured_shapes=unstructured_name_shapes,
                                                 fully_fine_tuned_keys=fully_fine_tuned_keys,
                                                 no_structured_drop_out=args.no_structured_drop_out,
                                                 no_structured_drop_path=args.no_structured_drop_path,
                                                 )

    train_engine = train_one_epoch
    test_engine = evaluate

    if args.resume:
        # Hard-coded pre-trained model name
        if '.pth' in args.resume:

            if args.resume.endswith('mae_pretrain_vit_base.pth'):
                state_dict = torch.load(args.resume, map_location='cpu')['model']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q')] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k')] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v')] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    else:
                        new_dict[name] = state_dict[name]

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from MAE model: ', msg)

            elif args.resume.endswith('linear-vit-b-300ep.pth'):
                state_dict = torch.load(args.resume, map_location='cpu')['state_dict']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q').split('module.')[1]] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k').split('module.')[1]] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v').split('module.')[1]] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    elif 'head.' in name:
                        continue
                    else:
                        new_dict[name.split('module.')[1]] = state_dict[name]

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from MoCo model: ', msg)

            elif args.resume.endswith('swin_base_patch4_window7_224_22k.pth'):

                state_dict = torch.load(args.resume, map_location='cpu')['model']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q')] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k')] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v')] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    elif 'head.' in name:
                        continue
                    else:
                        new_dict[name] = state_dict[name]

                if args.nb_classes != model.head.weight.shape[0]:
                    model.reset_classifier(args.nb_classes)

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from Swin model: ', msg)

            else:
                raise NotImplementedError

        else:
            load_checkpoint(model, args.resume)
            if args.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(args.nb_classes)

    model.to(device)
    model_ema = None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.get_sensitivity:
        start_time = time.time()
        get_sensitivity(
            model, criterion, data_loader_sensitivity, device,
            amp=args.amp, dataset=args.data_set,
            structured_vector=args.structured_vector, low_rank_dim=args.low_rank_dim,
            exp_name=args.exp_name, structured_type=args.structured_type,
            alpha=args.alpha, beta=args.beta, structured_only=args.structured_only,
            sensitivity_batch_num=args.sensitivity_batch_num
        )
        end_time = time.time()
        print(f"Function took {end_time - start_time} seconds to run.")
        return

    optimizer = utils.build_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # save config for later experiments
    with open(output_dir / "config.yaml", 'w') as f:
        f.write(args_text)

    if args.eval:
        test_stats = test_engine(data_loader_val, model, device, amp=args.amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print("Start training")

    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_engine(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            amp=args.amp, scaler=args.scaler
        )

        lr_scheduler.step(epoch)

        if epoch % args.val_interval == 0 or epoch >= args.epochs-10:  # Evaluate more in the last a few epochs
            test_stats = test_engine(data_loader_val, model, device, amp=args.amp)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(
                f"[{args.exp_name}] Max accuracy on the {args.data_set} dataset {len(dataset_val)} with ({args.opt}, {args.lr}, {args.weight_decay}), {max_accuracy:.2f}%")

            # Save to csv
            save_to_csv('csvs/' + args.exp_name, args.data_set, "%.2f" % round(max_accuracy,2))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
