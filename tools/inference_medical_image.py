import argparse
import os
import numpy as np
import mmcv
import torch
import nibabel as nib
from mmcv.parallel import MMDataParallel
from mmcv.runner import (init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from tqdm import tqdm
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--work-dir', default=None, type=str, help='inferece result save path')
    parser.add_argument('--segmentation', action='store_true', help='save segmentation output')
    parser.add_argument('--threshold', default=0.5, type=float, help='')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if cfg.model.get('test_cfg', None) is not None:
        cfg.model['test_cfg']['threshold'] = args.threshold
    else:
        if cfg.get('test_cfg') is not None:
            cfg.test_cfg['threshold'] = args.threshold
        else:
            cfg.test_cfg = dict(threshold= args.threshold)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()
    results = []
    eval_options = {} if args.eval_options is None else args.eval_options
    eval_options.update({} if cfg.eval_options is None else cfg.eval_options)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    for d in tqdm(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **d)[0]
            result = result.astype(np.float64)
            # result = result.astype(np.float64)
            results.append(result)
            num_classes = result.shape[0]
            for i in range(num_classes):
                image_nii = nib.Nifti1Image(result[i, ...], d['img_metas'].data[0][0]['img_affine_matrix'][0])
                if not eval_options.get('background_as_first_channel', True):
                    save_path = os.path.join(args.work_dir,
                                             d['img_metas'].data[0][0]['image_id'] + '_{}.nii.gz'.format(dataset.CLASSES[i + 1]))
                else:
                    save_path = os.path.join(args.work_dir,
                                             d['img_metas'].data[0][0]['image_id'] + '_{}.nii.gz'.format(
                                                 model.CLASSES[i]))
                nib.save(image_nii, save_path)
    data_loader.dataset.evaluate(results, 'mDice', **eval_options)

if __name__ == '__main__':
    main()