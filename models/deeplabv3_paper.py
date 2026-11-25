"""PyTorch port of the ResNet-101 / ASPP DeepLabv3 configuration.

Reference: Chen et al., arXiv:1706.05587; see docs/THIRD_PARTY.md.
Uses TF-Slim's end-of-block subsampling and multi-grid (1,2,4), OS=16.
ImageNet initialization is torchvision's R101 V1, not a converted TF checkpoint.
Training normalization/input protocol is the common PCB protocol (documented).
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101
from common import ROOT, digest


class SameMaxPool(nn.Module):
    def forward(self, x):
        # TensorFlow SAME maxpool: for even dimensions pad bottom/right only.
        h, w = x.shape[-2:]
        ph, pw = 1+h % 2, 1+w % 2
        x = F.pad(x, (pw//2, pw-pw//2, ph//2, ph-ph//2), value=float('-inf'))
        return F.max_pool2d(x, 3, 2)


def conv_bn_relu(cin, cout, kernel, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=(kernel//2)*dilation,
                                   dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout, eps=1e-5, momentum=.0003), nn.ReLU(inplace=True))


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_relu(2048, 256, 1)] +
                                      [conv_bn_relu(2048, 256, 3, r) for r in (6, 12, 18)])
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_bn_relu(2048, 256, 1))
        self.project = nn.Sequential(conv_bn_relu(1280, 256, 1), nn.Dropout(.1))

    def forward(self, x):
        pooled = F.interpolate(self.image_pool(x), size=x.shape[-2:], mode='bilinear', align_corners=True)
        # Author TF implementation puts the image-level branch first.
        return self.project(torch.cat([pooled]+[branch(x) for branch in self.branches], dim=1))


class DeepLabV3Paper(nn.Module):
    def __init__(self, num_classes=5, pretrained=False, weights_path=None, expected_sha256=None):
        super().__init__()
        net = resnet101(weights=None)
        self.initialization_provenance = {'kind': 'random', 'pretrained': False}
        if pretrained:
            path = ROOT/weights_path
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; run python scripts/fetch_weights.py --only resnet101')
            actual = digest(path)
            if not expected_sha256 or actual != expected_sha256:
                raise ValueError('ResNet101 pretrained checkpoint SHA256 mismatch')
            net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            self.initialization_provenance = {'kind': 'torchvision ResNet101 IMAGENET1K_V1',
                                              'pretrained': True, 'path': str(path.relative_to(ROOT)),
                                              'sha256': actual}
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, SameMaxPool())
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        # TF-Slim puts stride 2 in the LAST unit of block1/block2. At OS16,
        # block3's terminal stride is removed; block4 uses rates 2*(1,2,4).
        for layer in self.layers:
            for unit in layer:
                unit.stride = 1
                unit.conv2.stride = (1, 1)
                if unit.downsample is not None:
                    unit.downsample[0].stride = (1, 1)
        for layer in self.layers[:2]:
            layer[-1].stride = 2
            layer[-1].conv2.stride = (2, 2)
            layer[-1].downsample = nn.MaxPool2d(1, stride=2)
        for unit, rate in zip(self.layers[3], (2, 4, 8)):
            unit.conv2.dilation = (rate, rate)
            unit.conv2.padding = (rate, rate)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps, module.momentum = 1e-5, .003
        self.aspp = ASPP()
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return F.interpolate(self.classifier(self.aspp(x)), size=size,
                             mode='bilinear', align_corners=True)
