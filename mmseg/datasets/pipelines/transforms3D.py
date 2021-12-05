import numbers

from ..builder import PIPELINES
import numpy as np
import mmcv
from scipy.ndimage import binary_fill_holes
import nibabel as nib
from mmcv.utils import deprecated_api_warning
import torch

#TODO to be eval
@PIPELINES.register_module()
class ResizeMeical(object):
    """Resize images & seg.

        This transform resizes the input image to some scale. If the input dict
        contains the key "scale", then the scale in the input dict is used,
        otherwise the specified scale in the init method is used.

        ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
        (multi-scale). There are 4 multiscale modes:

        - ``ratio_range is not None``:
        1. When img_scale is None, img_scale is the shape of image in results
            (img_scale = results['img'].shape[:2]) and the image is resized based
            on the original size. (mode 1)
        2. When img_scale is a tuple (single-scale), randomly sample a ratio from
            the ratio range and multiply it with the image scale. (mode 2)

        - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
        scale from the a range. (mode 3)

        - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
        scale from multiple scales. (mode 4)

        Args:
            img_scale (tuple or list[tuple]): Images scales for resizing.
            multiscale_mode (str): Either "range" or "value".
            ratio_range (tuple[float]): (min_ratio, max_ratio)
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the
                image.
        """
    def __init__(self, img_scale=None, multiscale_mode='range',
                 ratio_range=None, keep_ratio=True, interpolation='trilinear'):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)


        if ratio_range is None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode.lower() in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.interpolation=interpolation

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 3
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio), int(img_scale[2] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                if not isinstance(results['img'], list):
                    c, h, w = results['img'].shape
                else:
                    c, h, w = results['img'][0].shape
                scale, scale_idx = self.random_sample_ratio((c, h, w),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale) # ??
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def rescale_size(self, old_size, scale, return_scale=False):
        def _scale_size(size, scale):
            """Rescale a size by a ratio.

                Args:
                    size (tuple[int]): (w, h).
                    scale (float): Scaling factor.

                Returns:
                    tuple[int]: scaled size.
                """
            c, h, w = size
            return int(c * float(scale) + 0.5), int(h * float(scale) + 0.5), int(w * float(scale) + 0.5)
        c, h, w = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max([h, w, c]),
                               max_short_edge / min([h, w, c]))
        else:
            raise TypeError(
                f'Scale must be a number or tuple of int, but got {type(scale)}')
        new_size = _scale_size((c, h, w), scale_factor)
        if return_scale:
            return new_size, scale_factor
        else:
            return new_size


    def imrescale(self, img,
                  scale,
                  return_scale=False,
                  interpolateion='bilinear'):
        c, h, w = img.shape
        new_size, scale_factor = self.rescale_size((c, h, w), scale, return_scale=True)
        if interpolateion in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            rescaled_img = torch.nn.functional.interpolate(torch.from_numpy(img[None, None, ...]),
                                                           size=new_size, align_corners=True,
                                                           mode=interpolateion).squeeze().detach().numpy()
        else:
            rescaled_img = torch.nn.functional.interpolate(torch.from_numpy(img[None, None, ...]),
                                                           size=new_size,
                                                           mode=interpolateion).squeeze().detach().numpy()

        if not return_scale:
            return rescaled_img
        return rescaled_img, scale_factor

    def resize(self, img, size, return_scale=False, interpolateion='bilinear'):
        c, h, w = img.shape
        if interpolateion in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            resized_img = torch.nn.functional.interpolate(torch.from_numpy(img[None, None, ...]),
                                                          size=size, align_corners=True,
                                                          mode=interpolateion).squeeze().detach().numpy()
        else:
            resized_img = torch.nn.functional.interpolate(torch.from_numpy(img[None, None, ...]),
                                                          size=size,
                                                          mode=interpolateion).squeeze().detach().numpy()
        if not return_scale:
            return resized_img
        c_scale = size[0] / c
        h_scale = size[1] / h
        w_scale = size[2] / w
        return resized_img, c_scale, h_scale, w_scale


    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            if not isinstance(results['img'], list):
                img, scale_factor = self.imrescale(results['img'],
                                                   results['scale'],
                                                   return_scale=True,
                                                   interpolateion=self.interpolation)
                new_c, new_h, new_w = img.shape
                c, h, w = results['img'].shape
                c_scale = new_c / c
                h_scale = new_h / h
                w_scale = new_w / w
                scale_factor = np.asarray([c_scale, h_scale, w_scale], dtype=np.float32)
                results['img'] = img
                results['img_shape'] = img.shape
                results['pad_shape'] = img.shape  # in case that there is no padding
                results['scale_factor'] = scale_factor
                results['keep_ratio'] = self.keep_ratio
            else:
                for i in range(len(results['img'])):
                    img, scale_factor = self.imrescale(results['img'][i],
                                                   results['scale'],
                                                   return_scale=True,
                                                   interpolateion=self.interpolation)
                    if i == 0:
                        new_c, new_h, new_w = img.shape
                        c, h, w = results['img'][i].shape
                        c_scale = new_c / c
                        h_scale = new_h / h
                        w_scale = new_w / w
                        scale_factor = np.asarray([c_scale, h_scale, w_scale], dtype=np.float32)
                        results['img'][i] = img
                        results['img_shape'] = img.shape
                        results['pad_shape'] = img.shape  # in case that there is no padding
                        results['scale_factor'] = scale_factor
                        results['keep_ratio'] = self.keep_ratio
                    else:
                        results['img'][i] = img
        else:
            if not isinstance(results['img'], list):
                img, c_scale, h_scale, w_scale = self.resize(results['img'],
                                                             results['scale'],
                                                             return_scale=True,
                                                             interpolateion=self.interpolation)
                scale_factor = np.asarray([c_scale, h_scale, w_scale],
                                          dtype=np.float32)
                results['img'] = img
                results['img_shape'] = img.shape
                results['pad_shape'] = img.shape  # in case that there is no padding
                results['scale_factor'] = scale_factor
                results['keep_ratio'] = self.keep_ratio
            else:
                for i in range(len(results['img'])):
                    img, c_scale, h_scale, w_scale = self.resize(results['img'][i],
                                                                 results['scale'],
                                                                 return_scale=True,
                                                                 interpolateion=self.interpolation)
                    if i == 0:
                        scale_factor = np.asarray([c_scale, h_scale, w_scale], dtype=np.float32)
                        results['img'][i] = img
                        results['img_shape'] = img.shape
                        results['pad_shape'] = img.shape  # in case that there is no padding
                        results['scale_factor'] = scale_factor
                        results['keep_ratio'] = self.keep_ratio
                    else:
                        results['img'][i] = img

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                # gt_seg = mmcv.imrescale(
                #     results[key], results['scale'], interpolation='nearest')
                gt_seg = self.imrescale(results[key], results['scale'], interpolateion='nearest')
            else:
                gt_seg = self.resize(results[key], results['scale'], interpolateion='nearest')
                # gt_seg = mmcv.imresize(
                #     results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

@PIPELINES.register_module()
class RandomCropMedical(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_region(self, img):
        """Randomly get a crop bounding box."""
        margin_x = max(img.shape[0] - self.crop_size[0], 0)
        margin_y = max(img.shape[1] - self.crop_size[1], 0)
        margin_z = max(img.shape[2] - self.crop_size[2], 0)
        offset_x = np.random.randint(0, margin_x + 1)
        offset_y = np.random.randint(0, margin_y + 1)
        offset_z = np.random.randint(0, margin_z + 1)
        crop_x1, crop_x2 = offset_x, offset_x + self.crop_size[0]
        crop_y1, crop_y2 = offset_y, offset_y + self.crop_size[1]
        crop_z1, crop_z2 = offset_z, offset_z + self.crop_size[2]
        return crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2

    def crop(self, img, crop_region):
        """Crop from ``img``"""
        # crop_y1, crop_y2, crop_x1, crop_x2 = crop_region
        # img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2 = crop_region
        img = img[crop_x1: crop_x2, crop_y1: crop_y2, crop_z1: crop_z2]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        if isinstance(results["img"], list):
            img = results["img"][0]
        else:
            img = results["img"]
        assert isinstance(img, np.ndarray) and len(img.shape) == len(self.crop_size)
        crop_region = self.get_crop_region(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(20):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_region)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_region = self.get_crop_region(img)
        if isinstance(results["img"], list):
            for i in range(len(results["img"])):
                results["img"][i] = self.crop(results["img"][i], crop_region)
            img_shape = results["img"][0].shape
        else:
            results["img"] = self.crop(results["img"], crop_region)
            img_shape = results["img"].shape
        # img = self.crop(img, crop_region)
        # img_shape = img.shape
        # results['img'] = img
        results['img_shape'] = img_shape
        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_region)
            # print("[DEBUG] foreground {}: {}".format(key, np.where((results[key] > 0) * (results[key] < 5))[0].shape))
            # print("[DEBUG] ignore {}: {}".format(key, np.where(results[key]== 255)[0].shape))
        return results

@PIPELINES.register_module()
class ExtractDataFromObj(object):
    def __init__(self):
        pass
    def __call__(self, results):
        # if isinstance(results["gt_semantic_seg"], nib.Nifti1Image):
        #     results["gt_semantic_seg"] = np.squeeze(results["gt_semantic_seg"].get_fdata(dtype=np.float32))
        if mmcv.is_list_of(results["img"], nib.Nifti1Image) or isinstance(results["img"], nib.Nifti1Image):
            for key in results.get('seg_fields', []):
                results[key] = np.squeeze(results[key].get_fdata(dtype=np.float32))
            if mmcv.is_list_of(results["img"], nib.Nifti1Image):
                for i, img_nii in enumerate(results["img"]):
                    results["img"][i] = np.squeeze(img_nii.get_fdata(dtype=np.float32))
                results['ori_shape'] = results['img'][0].shape
                results['img_shape'] = results['img'][0].shape
            else:
                if not isinstance(results["img"], nib.Nifti1Image):
                    print("[ERROR] Unsupported image type: {}!".format(type(results["img"])))
                    raise ValueError
                results["img"] = np.squeeze(results["img"].get_fdata(dtype=np.float32))
                results['ori_shape'] = results['img'].shape
                results['img_shape'] = results['img'].shape
        self._check_image_seg(results)
        return results
    def _check_image_seg(self, results):
        if isinstance(results['img'], list):
            image_shape = results['img'][0].shape
            for i in range(1, len(results['img'])):
                assert results['img'][i].shape == image_shape
        else:
            image_shape = results['img'].shape
        for key in results.get('seg_fields', []):
            assert results[key].shape == image_shape, results['img_info']

@PIPELINES.register_module()
class RandomCropMedicalWithForeground(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, fore_cat_ratio=1., ignore_index=255):
        self.crop_size = crop_size
        self.fore_cat_ratio = fore_cat_ratio
        self.ignore_index = ignore_index

    def get_crop_region(self, img):
        """Randomly get a crop bounding box."""
        margin_x = max(img.shape[0] - self.crop_size[0], 0)
        margin_y = max(img.shape[1] - self.crop_size[1], 0)
        margin_z = max(img.shape[2] - self.crop_size[2], 0)
        offset_x = np.random.randint(0, margin_x + 1)
        offset_y = np.random.randint(0, margin_y + 1)
        offset_z = np.random.randint(0, margin_z + 1)
        crop_x1, crop_x2 = offset_x, offset_x + self.crop_size[0]
        crop_y1, crop_y2 = offset_y, offset_y + self.crop_size[1]
        crop_z1, crop_z2 = offset_z, offset_z + self.crop_size[2]
        return crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2

    def crop(self, img, crop_region):
        """Crop from ``img``"""
        # crop_y1, crop_y2, crop_x1, crop_x2 = crop_region
        # img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2 = crop_region
        img = img[crop_x1: crop_x2, crop_y1: crop_y2, crop_z1: crop_z2]
        return img

    def find_non_zero_labels_mask(self, segmentation_map, th_percent, crop_region):
        # d1, d2, d3 = segmentation_map.shape
        # segmentation_map[segmentation_map > 0] = 1
        # total_voxel_labels = segmentation_map.sum()
        total_voxel_labels = (segmentation_map > 0).sum()
        cropped_segm_map = self.crop(segmentation_map, crop_region)
        crop_voxel_labels = (cropped_segm_map > 0).sum()

        label_percentage = crop_voxel_labels / total_voxel_labels
        # print(label_percentage,total_voxel_labels,crop_voxel_labels)
        if label_percentage >= th_percent:
            return True
        else:
            return False

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        if isinstance(results["img"], list):
            img = results["img"][0]
        else:
            img = results["img"]
        assert isinstance(img, np.ndarray) and len(img.shape) == len(self.crop_size)
        marge_shape = []
        for i in range(len(img.shape)):
            marge = img.shape[i] - self.crop_size[i]
            if marge >= 0:
                marge_shape.append(0)
            else:
                marge_shape.append(-1 * marge)
        if np.any(np.asarray(marge_shape) > 0):
            if isinstance(results["img"], list):
                org_shape = results['img'][0].shape
                new_shape = (org_shape[0] + marge_shape[0],
                             org_shape[1] + marge_shape[1],
                             org_shape[2] + marge_shape[2])
                for i in range(len(results['img'])):
                    image = np.zeros(new_shape, dtype=results['img'][i].dtype)
                    image[0: org_shape[0], 0: org_shape[1], 0: org_shape[2]] = results['img'][i]
                    results['img'][i] = image
            else:
                org_shape = results['img'].shape
                new_shape = (org_shape[0] + marge_shape[0],
                             org_shape[1] + marge_shape[1],
                             org_shape[2] + marge_shape[2])
                image = np.zeros(new_shape, dtype=results['img'].dtype)
                image[0: org_shape[0], 0: org_shape[1], 0: org_shape[2]] = results['img']
                results['img'] = image
            for key in results.get('seg_fields', []):
                image = np.zeros(new_shape, dtype=results[key].dtype)
                image[0: org_shape[0], 0: org_shape[1], 0: org_shape[2]] = results[key]
                results[key] = image
        crop_region = self.get_crop_region(img)
        if self.fore_cat_ratio < 1.:
            # Repeat 10 times
            for _ in range(20):
                # seg_temp = self.crop(results['gt_semantic_seg'], crop_region)
                # labels, cnt = np.unique(seg_temp, return_counts=True)
                # cnt = cnt[labels != self.ignore_index]
                if self.find_non_zero_labels_mask(results['gt_semantic_seg'], self.fore_cat_ratio,
                                                  crop_region):
                    break
                crop_region = self.get_crop_region(img)
        if isinstance(results["img"], list):
            for i in range(len(results["img"])):
                results["img"][i] = self.crop(results["img"][i], crop_region)
            img_shape = results["img"][0].shape
        else:
            results["img"] = self.crop(results["img"], crop_region)
            img_shape = results["img"].shape
        # img = self.crop(img, crop_region)
        # img_shape = img.shape
        # results['img'] = img
        results['img_shape'] = img_shape
        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_region)
            # print("[DEBUG] foreground {}: {}".format(key, np.where((results[key] > 0) * (results[key] < 5))[0].shape))
            # print("[DEBUG] ignore {}: {}".format(key, np.where(results[key]== 255)[0].shape))
        return results

#TODO: consider padding
# @PIPELINES.register_module()
# class RandomCropMedicalWithAnnotations(object):
#     def __init__(self, is_save=False, save_path=None):
#         self.is_save = is_save
#         self.save_path = save_path
#         if self.is_save and not os.path.exists(self.save_path):
#             os.makedirs(self.save_path)
#     def __call__(self, results):
#         if "gt_semantic_seg" not in results:
#             print("[WARNING] key: gt_semantic_seg not in input data")
#             return results
#         mask = results["gt_semantic_seg"]
#         crop_region = self._get_annotation_region(mask)
#         if mmcv.is_list_of(results["img"], np.ndarray):
#             for i in range(len(results["img"])):
#                 results["img"][i] = self.crop(results["img"][i], crop_region)
#                 if self.is_save:
#                     img_nii = nib.Nifti1Image(results["img"][i], results["img_affine_matrix"][i])
#                     img_name = results["filename"][i].split('/')[-1].split('.')[0]
#                     nib.save(img_nii, os.path.join(self.save_path, "{}_crop.nii.gz".format(img_name)))
#         else:
#             results["img"] = self.crop(results["img"], crop_region)
#             if self.is_save:
#                 img_nii = nib.Nifti1Image(results["img"], results["img_affine_matrix"])
#                 img_name = results["filename"].split('/')[-1].split('.')[0]
#                 nib.save(img_nii, os.path.join(self.save_path, "{}_crop.nii.gz".format(img_name)))
#         for key in results.get('seg_fields', []):
#             results[key] = self.crop(results[key], crop_region)
#             if self.is_save:
#                 img_nii = nib.Nifti1Image(results[key], results["seg_affine_matrix"])
#                 img_name = results["filename"].split('/')[-1].split('.')[0]
#                 nib.save(img_nii, os.path.join(self.save_path, "{}_crop_{}.nii.gz".format(img_name, key)))
#         return results
#
#     @staticmethod
#     def _get_annotation_region(mask):
#         location = np.where(mask > 0)
#         crop_region = []
#         for loc in location:
#             crop_region.append(np.min(loc))
#             crop_region.append(np.max(loc) + 1)
#         return crop_region
#
#     def crop(self, img, crop_region):
#         """Crop from ``img``"""
#         # crop_y1, crop_y2, crop_x1, crop_x2 = crop_region
#         # img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
#         crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2 = crop_region
#         img = img[crop_x1: crop_x2, crop_y1: crop_y2, crop_z1: crop_z2]
#         return img

@PIPELINES.register_module()
class RandomCropMedicalWithAnnotations(object):
    def __init__(self, pad_mode=None, **kwargs):
        self.pad_mode = pad_mode
        self.kwargs = kwargs
        self._check_cfg(pad_mode, kwargs)
    def __call__(self, results):
        if "gt_semantic_seg" not in results:
            print("[WARNING] key: gt_semantic_seg not in input data")
            return results
        mask = results["gt_semantic_seg"]
        crop_region = self._get_annotation_region(mask, self.pad_mode, self.kwargs)
        if mmcv.is_list_of(results["img"], np.ndarray):
            for i in range(len(results["img"])):
                results["img"][i] = self.crop(results["img"][i], crop_region)

        else:
            results["img"] = self.crop(results["img"], crop_region)

        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_region)
        return results

    def _check_cfg(self, pad_mode, kwargs):
        if pad_mode == 'static':
            assert 'static_size' in kwargs

    @staticmethod
    def _get_annotation_region(mask, pad_mode, kwargs):
        location = np.where(mask > 0)
        crop_region = []
        for i, loc in enumerate(location):
            # crop_region.append(np.min(loc))
            # crop_region.append(np.max(loc) + 1)
            min_l = np.min(loc)
            max_r = np.max(loc)
            if pad_mode == 'static':
                size = kwargs['static_size'][i]
                assert size <= mask.shape[i]
                left = np.random.randint(0, min_l + 1)
                right = np.random.randint(max_r, mask.shape[i])
                margin = (right - left) - size
                margin_l = int(np.random.uniform() * margin)
                left = max(0, left + margin_l)
                right = min(mask.shape[i], left + size)
                left = min(0, right - size)

                # left = np.random.randint(0, min_l + 1)
                # right = min(left + size, mask.shape[i])
                # left = max(right - size, 0)
                crop_region.append(left)
                crop_region.append(right)
            elif pad_mode == None:
                crop_region.append(min_l)
                crop_region.append(max_r)
            else:
                raise NotImplementedError
        return crop_region

    def crop(self, img, crop_region):
        """Crop from ``img``"""
        # crop_y1, crop_y2, crop_x1, crop_x2 = crop_region
        # img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2 = crop_region
        img = img[crop_x1: crop_x2, crop_y1: crop_y2, crop_z1: crop_z2]
        return img

@PIPELINES.register_module()
class CropMedicalExceptHoleArea(object):
    def __call__(self, results):
        img = results['img']
        nonzero_mask = self._create_nonzero_mask(img)
        bbox = self._get_bbox_from_mask(nonzero_mask)
        if isinstance(results['img'], list):
            for i in range(len(results['img'])):
                results['img'][i] = self.crop(results['img'][i], bbox)
        else:
            results['img'] = self.crop(results['img'], bbox)

        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], bbox)
        return results

    def _create_nonzero_mask(self, img):
        if isinstance(img, list):
            nonzero_mask = np.zeros(img[0].shape, dtype=bool)
            for i in range(len(img)):
                this_mask = img[i] != 0
                nonzero_mask = nonzero_mask | this_mask
            nonzero_mask = binary_fill_holes(nonzero_mask)
        else:
            nonzero_mask = img != 0
            nonzero_mask = binary_fill_holes(nonzero_mask)
        return nonzero_mask

    def _get_bbox_from_mask(self, mask, outside_value=0):
        mask_voxel_coords = np.where(mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        return [minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx]

    def crop(self, img, crop_region):
        crop_x1, crop_x2, crop_y1, crop_y2, crop_z1, crop_z2 = crop_region
        img = img[crop_x1: crop_x2, crop_y1: crop_y2, crop_z1: crop_z2]
        return img


@PIPELINES.register_module()
class NormalizeMedical(object):
    def __init__(self, norm_type: str, clip_intenisty=True, **kwargs):
        self.norm_type = norm_type.lower()
        self.clip_intenisty = clip_intenisty
        self.kwargs = kwargs
    def __call__(self, results):
        if self.norm_type == "full_volume_mean":
            if isinstance(results["img"], list):
                for i, img_np in enumerate(results["img"]):
                    if self.clip_intenisty:
                        img_np = self.percentile_clip(img_np,
                                                     self.kwargs.get("instensity_min_val", 0.1),
                                                     self.kwargs.get("instensity_max_val", 99.8))
                    results["img"][i] = (img_np - img_np.mean()) / (img_np.std() + 10e-5)
            else:
                if self.clip_intenisty:
                    results["img"] = self.percentile_clip(results["img"],
                                                  self.kwargs.get("instensity_min_val", 0.1),
                                                  self.kwargs.get("instensity_max_val", 99.8))
                results["img"] =(results["img"] - results["img"].mean()) / (results["img"].std() + 10e-5)
        elif self.norm_type == "max_min":
            if isinstance(results["img"], list):
                for i, img_np in results["img"]:
                    results["img"][i] = (img_np - img_np.max()) / (img_np.min() - img_np.max() + 10e-5)
            else:
                results["img"] = (results["img"] - results["img"].max()) / \
                                 (results["img"].min() - results["img"].max() + 10e-5)
        elif self.norm_type == "mean":
            if isinstance(results["img"], list):
                for i, img_np in enumerate(results["img"]):
                    mask = img_np > 10e-5
                    if self.clip_intenisty:
                        img_np = self.percentile_clip(img_np,
                                                      self.kwargs.get("instensity_min_val", 0.1),
                                                      self.kwargs.get("instensity_max_val", 99.8))
                    desired = img_np[mask]
                    mean_val, std_val = desired.mean(), desired.std()
                    results["img"][i] = (img_np - mean_val) / (std_val + 10e-5)
            else:
                mask = results["img"] > 10e-5
                if self.clip_intenisty:
                    results["img"] = self.percentile_clip(results["img"],
                                                  self.kwargs.get("instensity_min_val", 0.1),
                                                  self.kwargs.get("instensity_max_val", 99.8))
                desired = results["img"][mask]
                mean_val, std_val = desired.mean(), desired.std()
                results["img"] = (results["img"] - mean_val) / (std_val + 10e-5)
        elif self.norm_type == 'wcww':
            window_min = self.kwargs.get('window_center', -600) - self.kwargs.get('window_width', 700) / 2
            window_width = self.kwargs.get('window_width', 700)
            window_max = window_min + window_width
            if isinstance(results["img"], list):
                for i, img_np in enumerate(results["img"]):
                    # results["img"][i] = (img_np - window_min) / (window_width + 10e-5)
                    results['img'][i] = np.clip(results['img'][i], a_min=window_min, a_max=window_max)
                    results['img'][i] = (results['img'][i] - window_min) / window_width
            else:
                results['img'] = np.clip(results['img'], a_min=window_min, a_max=window_max)
                results['img'] = (results['img'] - window_min) / window_width
        else:
            print("[ERROR] norm type: {} is not implement")
            raise NotImplementedError
        return results

    @staticmethod
    def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
        b, t = np.percentile(img_numpy, (min_val, max_val))
        data = np.clip(img_numpy, b, t)
        return data

@PIPELINES.register_module()
class ConcatImage(object):
    def __init__(self):
        pass
    def __call__(self, results):
        if mmcv.is_list_of(results["img"], np.ndarray) or \
                isinstance(results["img"], np.ndarray):
            if isinstance(results["img"], list):
                for i in range(len(results["img"])):
                    results["img"][i] = results["img"][i][np.newaxis, ...]
                results["img"] = np.concatenate(results["img"], axis=0)
            else:
                results["img"] = results["img"][np.newaxis, ...]
        return results

@PIPELINES.register_module()
class IgnoreBlackArea(object):
    def __init__(self, set_label=255):
        self.set_label = set_label
    def __call__(self, results):
        if mmcv.is_list_of(results["img"], np.ndarray):
            for key in results.get('seg_fields', []):
                results[key][results["img"][0] <= 10e-3] = self.set_label
        else:
            for key in results.get('seg_fields', []):
                results[key][results["img"] <= 10e-3] = self.set_label
        return results

@PIPELINES.register_module()
class BinaryCateogry(object):
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
    def __call__(self, results):
       for key in results.get('seg_fields', []):
           results[key][(results[key] > 0) & (results[key] != self.ignore_index)] = 1
       return results

@PIPELINES.register_module()
class RandomFlipMedical(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlipMedical')
    def __init__(self, prob=None, direction=0):
        self.prob = prob

        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in [0, 1, 2]

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            if mmcv.is_list_of(results['img'], np.ndarray):
                for i in range(len(results['img'])):
                    results['img'][i] = self.flip_image(results['img'][i], self.direction).copy()
            else:
                results['img'] = self.flip_image(results['img'], self.direction).copy()
            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = self.flip_image(results[key], self.direction).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'

    def flip_image(self, image, direction):
        assert len(image.shape) > direction
        if direction == 0:
            return image[::-1, ...]
        elif direction == 1:
            return image[:, ::-1, ...]
        elif direction == 2:
            return image[:, :, ::-1]


@PIPELINES.register_module()
class PadMedical(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def impad(self, img, shape, padding=None, pad_val=0):
        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            padding = [0, 0, 0, shape[0] - img.shape[0],
                       shape[1] - img.shape[1], shape[2] - img.shape[2]]
            assert np.all(np.asarray(padding) >= 0)
        assert isinstance(pad_val, (float, int)), f"[ERROR] Padding Value Can't be {type(pad_val)}"
        if isinstance(padding, (tuple, list)) and len(padding) in [2, 3, 6]:
            if len(padding) == 2:
                padding = (padding[0], padding[0], padding[0], padding[1], padding[1], padding[1])
            elif len(padding) == 3:
                padding = (0, 0, 0, padding[0], padding[1], padding[2])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 6 element tuple.'
                             f'But received {padding}')
        org_shape = img.shape
        new_shape = (org_shape[0] + padding[0] + padding[3],
                     org_shape[1] + padding[1] + padding[4],
                     org_shape[2] + padding[2] + padding[5])
        new_image = np.full(shape=new_shape, fill_value=pad_val, dtype=img.dtype)
        new_image[padding[0]: org_shape[0] + padding[0],
                  padding[1]: org_shape[1] + padding[1],
                  padding[2]: org_shape[2] + padding[2]] = img
        return new_image, padding

    def impad_to_multiple(self, img, divisor, pad_val=0):
        if isinstance(divisor, numbers.Number):
            pad_c = int(np.ceil(img.shape[0] / divisor)) * divisor
            pad_h = int(np.ceil(img.shape[1] / divisor)) * divisor
            pad_w = int(np.ceil(img.shape[2] / divisor)) * divisor
        elif isinstance(divisor, (tuple, list)):
            assert len(divisor) == len(img.shape)
            pad_c = int(np.ceil(img.shape[0] / divisor[0])) * divisor[0]
            pad_h = int(np.ceil(img.shape[1] / divisor[1])) * divisor[1]
            pad_w = int(np.ceil(img.shape[2] / divisor[2])) * divisor[2]
        else:
            raise ValueError("[ERROR] Padding Operation \'divisor\'")
        return self.impad(img, shape=(pad_c, pad_h, pad_w), pad_val=pad_val)

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            # padded_img = mmcv.impad(
            #     results['img'], shape=self.size, pad_val=self.pad_val)
            if not isinstance(results["img"], list):
                padded_img, padding = self.impad(results['img'], shape=self.size, pad_val=self.pad_val)
                results['img'] = padded_img
                results['pad_shape'] = padded_img.shape
                results['pad_fixed_size'] = self.size
                results['pad_size_divisor'] = self.size_divisor
                results['padding'] = padding
            else:
                for i in range(len(results['img'])):
                    padded_img, padding = self.impad(results['img'][i], shape=self.size, pad_val=self.pad_val)
                    if i == 0:
                        results['img'][i] = padded_img
                        results['pad_shape'] = padded_img.shape
                        results['pad_fixed_size'] = self.size
                        results['pad_size_divisor'] = self.size_divisor
                        results['padding'] = padding
                    else:
                        results['img'][i] = padded_img
        elif self.size_divisor is not None:
            # padded_img = mmcv.impad_to_multiple(
            #     results['img'], self.size_divisor, pad_val=self.pad_val)
            if not isinstance(results['img'], list):
                padded_img, padding = self.impad_to_multiple(results['img'],
                                                    divisor=self.size_divisor,
                                                    pad_val=self.pad_val)
                results['img'] = padded_img
                results['pad_shape'] = padded_img.shape
                results['pad_fixed_size'] = self.size
                results['pad_size_divisor'] = self.size_divisor
                results['padding'] = padding
            else:
                for i in range(len(results['img'])):
                    padded_img, padding = self.impad_to_multiple(results['img'][i],
                                                    divisor=self.size_divisor,
                                                    pad_val=self.pad_val)
                    if i == 0:
                        results['img'][i] = padded_img
                        results['pad_shape'] = padded_img.shape
                        results['pad_fixed_size'] = self.size
                        results['pad_size_divisor'] = self.size_divisor
                        results['padding'] = padding
                    else:
                        results['img'][i] = padded_img
    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            # results[key] = mmcv.impad(
            #     results[key],
            #     shape=results['pad_shape'][:2],
            #     pad_val=self.seg_pad_val)
            results[key] = self.impad(results[key], shape=results['pad_shape'],
                                      pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str