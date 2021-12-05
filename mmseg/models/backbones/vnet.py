import torch.nn as nn
import torch
# from lib.medzoo.BaseModelClass import BaseModel
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
from mmcv.cnn import (constant_init, kaiming_init)
from ..builder import BACKBONES
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
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
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
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

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

@BACKBONES.register_module()
class VNet(nn.Module):
    """
    Implementations based on the Vnet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, elu=True, in_channels=1, norm_eval=False):
        super(VNet, self).__init__()
        # self.classes = classes
        self.norm_eval = norm_eval
        self.in_channels = in_channels
        self._whole_dowsample_ratio = 1
        self.in_tr = InputTransition(in_channels, elu=elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self._whole_dowsample_ratio *= 2
        self.down_tr64 = DownTransition(32, 2, elu)
        self._whole_dowsample_ratio *= 2
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self._whole_dowsample_ratio *= 2
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self._whole_dowsample_ratio *= 2
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.init_weights()
        # self.out_tr = OutputTransition(32, classes, elu)

    def forward(self, x):
        self._check_input_divisible(x)
        outs = []
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        outs.append(self.up_tr256(out256, out128))
        outs.append(self.up_tr128(outs[-1], out64))
        outs.append(self.up_tr64(outs[-1], out32))
        outs.append(self.up_tr32(outs[-1], out16))
        # out = self.up_tr256(out256, out128)
        # out = self.up_tr128(out, out64)
        # out = self.up_tr64(out, out32)
        # out = self.up_tr32(out, out16)
        # out = self.out_tr(out)
        return outs

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, self.classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        # summary(self.to(torch.device(device)), (self.in_channels, 32, 32, 32),device=device)
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("Vnet test is complete")

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(VNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()

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
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _check_input_divisible(self, x):
        x_dim, y_dim, z_dim = x.shape[-3:]
        assert (x_dim % self._whole_dowsample_ratio == 0) \
            and (y_dim % self._whole_dowsample_ratio == 0) \
            and (z_dim % self._whole_dowsample_ratio == 0), \
            f'The input image size {( x_dim, y_dim, z_dim)} should be divisible by the whole '\
            f'downsample rate {self._whole_dowsample_ratio}, when num_stages is '\
            f'is {self.downsamples}.'

class VNetLight(nn.Module):
    """
    A lighter version of Vnet that skips down_tr256 and up_tr256 in oreder to reduce time and space complexity
    """

    def __init__(self, elu=True, in_channels=1, classes=4):
        super(VNetLight, self).__init__()
        self.classes = classes
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, classes, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out = self.up_tr128(out128, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

    def test(self,device='cpu'):
        input_tensor = torch.rand(1, self.in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, self.classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))

        print("Vnet light test is complete")


#m = VNet(in_channels=1,num_classes=2)
#m.test()
