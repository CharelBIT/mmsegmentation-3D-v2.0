import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from mmcv.runner import load_checkpoint
from mmseg.utils import get_root_logger
from functools import partial
from ..builder import BACKBONES
__all__ = [
    'resnet103D', 'resnet183D', 'resnet343D', 'resnet503D', 'resnet1013D',
    'resnet1523D', 'resnet2003D']


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 no_cuda=False,
                 pretrained=None):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
                elif isinstance(m, nn.BatchNorm3d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        outputs.append(x)
        x = self.layer2(x)
        outputs.append(x)
        x = self.layer3(x)
        outputs.append(x)
        x = self.layer4(x)
        outputs.append(x)
        return outputs

@BACKBONES.register_module()
class resnet103D(ResNet):
    """Constructs a ResNet-18 model.
    """
    def __init__(self,**kwargs):
        super(resnet103D).__init__(BasicBlock, [1, 1, 1, 1],
                                   **kwargs)

@BACKBONES.register_module()
class resnet183D(ResNet):

    def __init__(self, **kwargs):
        super(resnet183D).__init__(BasicBlock, [2, 2, 2, 2],
                                   **kwargs)

@BACKBONES.register_module()
class resnet343D(ResNet):
    """Constructs a ResNet-34 model.
    """
    # model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # return model
    def __init__(self, **kwargs):
        super(resnet343D).__init__(BasicBlock, [3, 4, 6, 3],
                                   **kwargs)

@BACKBONES.register_module()
class resnet503D(ResNet):
    """Constructs a ResNet-50 model.
    """
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # return model
    def __init__(self, **kwargs):
        super(resnet503D).__init__(Bottleneck, [3, 4, 6, 3],
                                   **kwargs)

@BACKBONES.register_module()
class resnet1013D(ResNet):
    """Constructs a ResNet-101 model.
    """
    # model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # return model
    def __init__(self, **kwargs):
        super(resnet1013D).__init__(Bottleneck, [3, 4, 23, 3],
                                   **kwargs)

@BACKBONES.register_module()
class resnet1523D(ResNet):
    """Constructs a ResNet-101 model.
    """
    # model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    # return model
    def __init__(self, **kwargs):
        super(resnet1523D).__init__(Bottleneck, [3, 8, 36, 3],
                                   **kwargs)

@BACKBONES.register_module()
class resnet2003D(ResNet):
    """Constructs a ResNet-101 model.
    """
    # model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    # return model
    def __init__(self, **kwargs):
        super(resnet2003D).__init__(Bottleneck, [3, 24, 36, 3],
                                   **kwargs)
