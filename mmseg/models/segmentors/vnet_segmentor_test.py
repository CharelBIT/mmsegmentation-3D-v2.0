import mmcv
import torch.nn as nn
import torch
# from lib.medzoo.BaseModelClass import BaseModel
from .base import BaseSegmentor
from ..builder import SEGMENTORS, build_loss
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmseg.ops import resize
from mmcv.runner import auto_fp16, force_fp32
from mmseg.core import build_pixel_sampler
import torch.nn.functional as F
from mmseg.models.losses.medical_dice_loss import expand_as_one_hot
"""
Implementation of this model is borrowed and modified
(to support multi-channels and latest pytorch version)
from here:
https://github.com/Dawn90/V-Net.pytorch
"""


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super(InputTransition, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)

        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False, stride=2):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        if isinstance(stride, int):
            self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=stride)
        elif isinstance(stride, (tuple, list)):
            kernel_size = []
            for s in stride:
                if s == 2:
                    kernel_size.append(2)
                elif s == 1:
                    kernel_size.append(1)
            self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError


        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False, stride=2):
        super(UpTransition, self).__init__()
        if isinstance(stride, int):
            self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=stride)
        elif isinstance(stride, (tuple, list)):
            kernel_size = []
            for s in stride:
                if s == 2:
                    kernel_size.append(2)
                elif s == 1:
                    kernel_size.append(1)
            self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2,
                                              kernel_size=kernel_size, stride=stride)
        else:
            raise ValueError
        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, classes, elu):
        super(OutputTransition, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels, classes, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(classes)

        self.conv2 = nn.Conv3d(classes, classes, kernel_size=1)
        self.relu1 = ELUCons(elu, classes)

    def forward(self, x):
        # convolve 32 down to channels as the desired classes
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out

@SEGMENTORS.register_module()
class VNetSegmentor_test(BaseSegmentor):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True,
                 in_channels=1,
                 classes=4,
                 loss_cfg=dict(type="MedicalDiceLoss",
                               classes=5,
                               use_sigmoid=True,
                               background_as_first_channel=True),
                 sampler=None,
                 pretrained=None,
                 resize_mode="trilinear",
                 align_corners=False,
                 train_cfg=None,
                 test_cfg=None,
                 ignore_index=None
                 ):
        super(BaseSegmentor, self).__init__()
        self.classes = classes
        self.in_channels = in_channels
        self.elu = elu
        self.resize_mode = resize_mode
        self.align_corners = align_corners
        self._build_model()
        if not isinstance(loss_cfg, list):
            self._use_sigmoid = loss_cfg.get("use_sigmoid", False)
            self._background_as_first_channel = loss_cfg.get("background_as_first_channel", False)
        else:
            self._use_sigmoid = loss_cfg[0].get("use_sigmoid", False)
            self._background_as_first_channel = loss_cfg[0].get("background_as_first_channel", False)
        self.ignore_index = ignore_index
        if mmcv.is_list_of(loss_cfg, dict):
            self.loss_func = []
            for i in range(len(loss_cfg)):
                self.loss_func.append(build_loss(loss_cfg[i]))
        else:
            self.loss_func = build_loss(loss_cfg)
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained)

    def _build_model(self):
        self.in_tr = InputTransition(self.in_channels, elu=self.elu)
        self.down_tr32 = DownTransition(16, 1, self.elu, stride=(2, 2, 1))
        self.down_tr64 = DownTransition(32, 2, self.elu, stride=(2, 2, 2))
        self.down_tr128 = DownTransition(64, 3, self.elu, dropout=True, stride=(2, 2, 2))
        self.down_tr256 = DownTransition(128, 2, self.elu, dropout=True, stride=(2, 2, 2))
        self.up_tr256 = UpTransition(256, 256, 2, self.elu, dropout=True, stride=(2, 2, 2))
        self.up_tr128 = UpTransition(256, 128, 2, self.elu, dropout=True, stride=(2, 2, 2))
        self.up_tr64 = UpTransition(128, 64, 1, self.elu, stride=(2, 2, 2))
        self.up_tr32 = UpTransition(64, 32, 1, self.elu, stride=(2, 2, 1))
        self.out_tr = OutputTransition(32, self.classes, self.elu)

    def _model_infer(self, x):
        # print(x.size())
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self._model_infer(img)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         kaiming_init(m)
            #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
            #         constant_init(m, 1)
            #
            # if self.dcn is not None:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck) and hasattr(
            #                 m, 'conv2_offset'):
            #             constant_init(m.conv2_offset, 0)
            #
            # if self.zero_init_residual:
            #     for m in self.modules():
            #         if isinstance(m, Bottleneck):
            #             constant_init(m.norm3, 0)
            #         elif isinstance(m, BasicBlock):
            #             constant_init(m.norm2, 0)
            logger = get_root_logger()
            logger.info("random initialize!")
        else:
            raise TypeError('pretrained must be a str or None')


    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode=self.resize_mode,
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        if isinstance(self.loss_func, list):
            for i in range(len(self.loss_func)):
                l = self.loss_func[i](
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                if isinstance(l, dict):
                    for key, val in l.items():
                        loss["{}_{}".format(self.loss_func[i].__class__.__name__, key)] = val
                else:
                    loss["loss_{}".format(self.loss_func[i].__class__.__name__)] = l
        else:
            loss = self.loss_func(
                seg_logit,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index)
            if isinstance(loss, dict):
                return loss
            else:
                return {"loss_seg": loss}
        return loss

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self._model_infer(crop_img)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds
    # NxMxXxYxZ
    def slide_inference_3D(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        x_stride, y_stride, z_stride = self.test_cfg.stride
        x_crop_length, y_crop_length, z_crop_length = self.test_cfg.crop_size
        batch_size, num_mode, x_img, y_img, z_img = img.size()
        num_classes = self.classes
        x_grids = max(x_img - x_crop_length + x_stride - 1, 0) // x_stride + 1
        y_grids = max(y_img - y_crop_length + y_stride - 1, 0) // y_stride + 1
        z_grids = max(z_img - z_crop_length + z_stride - 1, 0) // z_stride + 1
        preds = img.new_zeros((batch_size, num_classes, x_img, y_img, z_img))
        count_mat = img.new_zeros((batch_size, 1, x_img, y_img, z_img))
        for x_idx in range(x_grids):
            for y_idx in range(y_grids):
                for z_idx in range(z_grids):
                    x1 = x_idx * x_stride
                    y1 = y_idx * y_stride
                    z1 = z_idx * z_stride
                    x2 = min(x1 + x_crop_length, x_img)
                    y2 = min(y1 + y_crop_length, y_img)
                    z2 = min(z1 + z_crop_length, z_img)
                    x1 = max(x2 - x_crop_length, 0)
                    y1 = max(y2 - y_crop_length, 0)
                    z1 = max(z2 - z_crop_length, 0)
                    crop_img = img[:, :, x1:x2, y1:y2, z1:z2]
                    crop_seg_logit = self._model_infer(crop_img)# n x c x x x y x z
                    # preds += F.pad(crop_seg_logit,
                    #                (int(x1), int(preds.shape[2] - x2), int(y1),
                    #                 int(preds.shape[2] - y2), int(z1)))
                    preds += F.pad(crop_seg_logit, (int(z1), int(preds.shape[4] - z2),
                                                    int(y1), int(preds.shape[3] - y2),
                                                    int(x1), int(preds.shape[2] - x2)))

                    count_mat[:, :, x1: x2, y1: y2, z1: z2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'],
                mode='trilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds


    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self._model_infer(img)
        if rescale:
            # support dynamic shape for onnx
            if not self.test_cfg.get("three_dim_input", False):
                if torch.onnx.is_in_onnx_export():
                    size = img.shape[2:]
                else:
                    size = img_meta[0]['ori_shape'][:2]
                seg_logit = resize(
                    seg_logit,
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
            else:
                if 'padding' in img_meta[0]:
                    seg_logit = seg_logit[:, :,
                                    img_meta[0]['padding'][0]: img_meta[0]['img_shape'][0] + img_meta[0]['padding'][0],
                                    img_meta[0]['padding'][1]: img_meta[0]['img_shape'][1] + img_meta[0]['padding'][1],
                                    img_meta[0]['padding'][2]: img_meta[0]['img_shape'][2] + img_meta[0]['padding'][2]]
                size = img_meta[0]['ori_shape']
                seg_logit = resize(
                    seg_logit,
                    size=size,
                    mode='trilinear',
                    align_corners=self.align_corners,
                    warning=False)
        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        assert len(img_meta) == 1, "[ERROR] Only Support one sample per batch"
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            if self.test_cfg.get("three_dim_input", False):
                seg_logit = self.slide_inference_3D(img, img_meta, rescale)
            else:
                seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        # output = seg_logit
        if self._use_sigmoid:
            output = torch.sigmoid(seg_logit)
        else:
            output = torch.softmax(seg_logit, dim=1)
        try:
            flip = img_meta[0]['flip']
        except:
            flip = False
        if flip:
            if not self.test_cfg.get("three_dim_input", False):
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))
            else:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in [0, 1, 2]
                output = output.flip(dims=(flip_direction + 2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if self._use_sigmoid:
            try:
                threshold = self.test_cfg.get("threshold", 0.5)
            except:
                threshold = 0.5
            seg_pred = seg_logit >= threshold
        else:
            seg_pred = torch.argmax(seg_logit, dim=1)
            seg_pred = expand_as_one_hot(seg_pred, self.classes)

        # seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_logit = seg_pred.unsqueeze(0)
            return seg_logit
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        # assert rescale
        # # to save memory, we get augmented seg logit inplace
        # seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        # for i in range(1, len(imgs)):
        #     cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
        #     seg_logit += cur_seg_logit
        # seg_logit /= len(imgs)
        # # seg_pred = seg_logit.argmax(dim=1)
        # seg_logit = seg_logit.cpu().numpy()
        # # unravel batch dim
        # seg_logit = list(seg_logit)
        # return seg_logit
        raise NotImplementedError