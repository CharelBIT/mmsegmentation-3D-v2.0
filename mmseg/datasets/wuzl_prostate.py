
import os
import torch
import numpy as np
from glob import glob
import os.path as osp
from .builder import DATASETS
from .pipelines import Compose
from .custom import CustomDataset
from mmseg.utils import get_root_logger
from mmcv.utils import print_log
import nibabel as nib

import mmcv
from prettytable import PrettyTable
from mmseg.models.losses.medical_dice_loss import compute_per_channel_dice, expand_as_one_hot

from functools import reduce
@DATASETS.register_module()
class WUHANZL_ProstateDataset(CustomDataset):
    CLASSES = ("background", "1")
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffixes=['image.nii.gz'],
                 mode=None,
                 ann_dir=None,
                 seg_map_suffix="mask.nii.gz",
                 split=None,
                 data_root=None,
                 test_mode=False,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 ignore_index=255):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffixes = img_suffixes
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.mode = mode
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.ignore_index = ignore_index
        # self.datalist = None
        # if datalist_file is not None and osp.exists(datalist_file):
        #     self.datalist = [line.strip() for line in open(datalist_file, 'r').readlines()]

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.mode, self.img_suffixes,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def load_annotations(self, img_dir, mode, img_suffixes, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_id = line.strip()
                    img_info = dict(filename=[])
                    for img_suffix in img_suffixes:
                        img_name = osp.join(img_dir, mode, img_id, img_suffix)
                        img_info['filename'].append(img_name)

                    if ann_dir is not None:
                        seg_map = osp.join(img_dir, mode, img_id, seg_map_suffix)
                        img_info["ann"] = dict(seg_map=seg_map)
                    img_info['image_id'] = img_id
                    img_infos.append(img_info)
        else:
            img_ids = os.listdir(img_dir)
            for img_id in img_ids:
                img_info = dict(filename=[])
                for img_suffix in img_suffixes:
                    img_name =  osp.join(img_dir, mode, img_id, img_suffix)
                    img_info['filename'].append(img_name)
                if ann_dir is not None:
                    seg_map = osp.join(img_dir, mode, img_id, seg_map_suffix)
                    img_info["ann"] = dict(seg_map=seg_map)
                img_info['image_id'] = img_id
                img_infos.append(img_info)
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)


    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                # gt_seg_map = mmcv.imread(
                #     seg_map, flag='unchanged', backend='pillow')
                gt_seg_map = nib.load(seg_map)
                gt_seg_map = np.squeeze(gt_seg_map.get_fdata(dtype=np.float32))
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric='mDice',
                 logger=None,
                 efficient_test=False,
                 **kwargs):

        eval_results = {}
        if isinstance(metric, list) or isinstance(metric, tuple):
            assert (len(metric) == 1 and metric[0] == 'mDice'), "Only Support mDice metric"
        else:
            assert metric == 'mDice', "Only Support mDice metric"
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        for i in range(len(gt_seg_maps)):
            gt_seg_maps[i] = torch.from_numpy(gt_seg_maps[i][np.newaxis, ...].astype(np.int64))
        # gt_seg_maps = torch.from_numpy(np.ascontiguousarray(np.concatenate(gt_seg_maps, axis=0)))
        if mmcv.is_list_of(results, np.ndarray):
            if len(results[0].shape) == 4:
                for i in range(len(results)):
                    results[i] = torch.from_numpy(results[i][np.newaxis, ...])
        elif  mmcv.is_list_of(results, torch.Tensor):
            if len(results[0].shape) == 4:
                for i in range(len(results)):
                    results[i] = results[i][np.newaxis, ...]
        else:
            raise NotImplementedError

        per_channel_dices = []
        assert len(results) == len(gt_seg_maps)
        for i in range(len(results)):
            gt_seg_map = expand_as_one_hot(gt_seg_maps[i], results[i].size(1),
                                            background_as_first_channel=kwargs.get("background_as_first_channel", True))
            per_channel_dice = compute_per_channel_dice(results[i], gt_seg_map)
            per_channel_dices.append(per_channel_dice.detach().cpu().numpy())
        per_channel_dice = np.asarray(per_channel_dices).mean(axis=0)
        class_table_data = PrettyTable()
        return_val = {}
        for i in range(per_channel_dice.shape[0]):
            class_table_data.add_column("{}_Channels".format(i), [per_channel_dice[i].item()])
            return_val["{}_Channels".format(i)] = per_channel_dice[i].item()
        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        return return_val