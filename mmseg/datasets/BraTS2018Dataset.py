
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
import torch.nn.functional as F
import mmcv
from prettytable import PrettyTable
from mmseg.models.losses.medical_dice_loss import compute_per_channel_dice, expand_as_one_hot

from functools import reduce
@DATASETS.register_module()
class BraTS2018Dataset(CustomDataset):
    CLASSES = ("background", "1", "2", "3", "4")
    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffixes=['flair.nii.gz', 't1ce.nii.gz', 't1.nii.gz', 't2.nii.gz'],
                 ann_dir=None,
                 seg_map_suffix="seg.nii.gz",
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
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffixes,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def load_annotations(self, img_dir, img_suffixes, ann_dir, seg_map_suffix,
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
                        img_name = glob(osp.join(img_dir, img_id, '*{}'.format(img_suffix)))
                        assert len(img_name) == 1, "Only support one image per mode"
                        img_info['filename'].append(img_name[0])
                    if ann_dir is not None:
                        seg_map = glob(osp.join(img_dir, img_id, '*{}'.format(seg_map_suffix)))
                        if len(seg_map) == 0:
                            img_info["ann"] = None
                        else:
                            assert len(seg_map) == 1, 'Only support one seg info'
                            img_info["ann"] = dict(seg_map=seg_map[0])
                    img_info['image_id'] = img_id
                    img_infos.append(img_info)
        else:
            img_ids = os.listdir(img_dir)
            for img_id in img_ids:
                img_info = dict(filename=[])
                for img_suffix in img_suffixes:
                    img_name =  glob(osp.join(img_dir, img_id, '*{}'.format(img_suffix)))
                    assert len(img_name) == 1, "Only support one image per mode"
                    img_info['filename'].append(img_name[0])
                if ann_dir is not None:
                    seg_map = glob(osp.join(img_dir, img_id, '*{}'.format(seg_map_suffix)))
                    assert len(seg_map) == 1, "Only support one seg info"
                    img_info["ann"] = dict(seg_map=seg_map[0])
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
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):

        eval_results = {}
        dice_per_patient = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        for i in range(len(gt_seg_maps)):
            gt_seg_maps[i] = gt_seg_maps[i][np.newaxis, ...].astype(np.int64)
        assert len(gt_seg_maps) == len(results), '[ERROR] Miss Match ground truth and prediction'
        patient_infos = [info['filename'] for info in self.img_infos]
        for i, patient_info in enumerate(patient_infos):
            gt_seg_map = torch.from_numpy(gt_seg_maps[i])
            result = torch.from_numpy(results[i])
            result = result[None, ...]
            gt_seg_map = expand_as_one_hot(gt_seg_map, result.size(1),
                                            background_as_first_channel=kwargs.get("background_as_first_channel", True))
            per_channel_dice = compute_per_channel_dice(result, gt_seg_map)
            per_channel_dice = per_channel_dice.detach().cpu().numpy()
            dice_per_patient[patient_info[0]] = per_channel_dice
        dices = np.asarray([dice_per_patient[key] for key in dice_per_patient])
        mean_dices = dices.mean(axis=0)
        # gt_seg_maps = torch.from_numpy(np.ascontiguousarray(np.concatenate(gt_seg_maps, axis=0)))
        # if mmcv.is_list_of(results, np.ndarray):
        #     if len(results[0].shape) == 5:
        #         results = torch.from_numpy(np.ascontiguousarray(np.concatenate(results, axis=0)))
        #     elif len(results[0].shape) == 4:
        #         for i in range(len(results)):
        #             results[i] = results[i][np.newaxis, ...]
        #         results = torch.from_numpy(np.ascontiguousarray(np.concatenate(results, axis=0)))
        # elif  mmcv.is_list_of(results, torch.Tensor):
        #     if len(results[0].shape) == 5:
        #         results = torch.cat(results, dim=0)
        #     elif len(results[0].shape) == 4:
        #         for i in range(len(results)):
        #             results[i] = results[i][np.newaxis, ...]
        #         results = torch.cat(results, dim=0)
        # else:
        #     raise NotImplementedError
        # gt_seg_maps = expand_as_one_hot(gt_seg_maps, results.size(1),
        #                                 background_as_first_channel=kwargs.get("background_as_first_channel", True))
        # per_channel_dice = compute_per_channel_dice(results, gt_seg_maps)
        # per_channel_dice = per_channel_dice.detach().cpu().numpy()
        class_table_data = PrettyTable()
        return_val = {}
        for i in range(mean_dices.shape[0]):
            class_table_data.add_column("{}_Channels".format(i), [mean_dices[i].item()])
            return_val["{}_Channels".format(i)] = mean_dices[i].item()
        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        return return_val