import numpy as np
import torch
from torch import nn as nn
from ..builder import LOSSES
from .utils import get_class_weight, weight_reduce_loss

# from lib.losses3D.basic import expand_as_one_hot


# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
# C 网络输出通道数
def expand_as_one_hot(input, C, ignore_index=None, background_as_first_channel=True):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 5:
        return input
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    if background_as_first_channel:
        shape[1] = C
    else:
        shape[1] = C + 1

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the lib tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        if background_as_first_channel:
            return result
        else:
            return result[:, 1:, ...]
    else:
        # scatter to get the one-hot tensor
        result =  torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        if background_as_first_channel:
            return result
        else:
            return result[:, 1:, ...]

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self,
                 classes=None,
                 loss_weight=1.0,
                 class_weight=None,
                 use_sigmoid=True,
                 reduction='mean',
                 avg_factor=None):
        super(_AbstractDiceLoss, self).__init__()
        # self.register_buffer('weight', weight)
        self.classes = classes
        self.skip_index_after = None
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.reduction = reduction
        self.avg_factor = avg_factor
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if use_sigmoid:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

        if self.class_weight is not None:
            if len(self.class_weight) != self.classes:
                raise ValueError
            else:
                self.class_weight = torch.from_numpy(np.asarray(self.class_weight, dtype=np.float))


    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def skip_target_channels(self, target, index):
        """
        Assuming dim 1 is the classes dim , it skips all the indexes after the desired class
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self,
                input,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None):
        """
        Expand to one hot added extra for consistency reasons
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        target = expand_as_one_hot(target.long(), self.classes,
                                   background_as_first_channel=self.background_as_first_channel,
                                   ignore_index=ignore_index)

        assert input.dim() == target.dim() == 5, "'input' and 'target' have different number of dims"

        if self.skip_index_after is not None:
            before_size = target.size()
            target = self.skip_target_channels(target, self.skip_index_after)
            print("Target {} after skip index {}".format(before_size, target.size()))

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target)

        loss_per_channel = (1. - per_channel_dice)
        loss = self.loss_weight * weight_reduce_loss(loss_per_channel,
                                  weight=self.class_weight.to(loss_per_channel.device),
                                  reduction=reduction,
                                  avg_factor=avg_factor)
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        # average Dice score across all channels/classes
        loss = dict(loss_seg=loss)
        for i in range(per_channel_dice.shape[0]):

            loss["{}_Channel".format(i)] = torch.tensor(per_channel_dice[i]).to(loss_per_channel.device)

        return loss


@LOSSES.register_module()
class MedicalDiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=None,
                 skip_index_after=None,
                 loss_weight=1.0,
                 class_weight=None,
                 use_sigmoid=True,
                 reduction='mean',
                 avg_factor=None,
                 background_as_first_channel=True):
        super().__init__(classes=classes,
                         loss_weight=loss_weight,
                         class_weight=class_weight,
                         use_sigmoid=use_sigmoid,
                         reduction=reduction,
                         avg_factor=avg_factor)
        self.classes = classes
        self.background_as_first_channel = background_as_first_channel
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight=None):
        return compute_per_channel_dice(input, target)

