import math

import jittor as jt
from jittor import init
from jittor import nn

__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']
model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth'}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor((planes * (baseWidth / 64.0))))
        self.conv1 = nn.Conv(inplanes, (width * scale), 1, bias=False)
        self.bn1 = nn.BatchNorm((width * scale))
        if (scale == 1):
            self.nums = 1
        else:
            self.nums = (scale - 1)
        if (stype == 'stage'):
            self.pool = nn.Pool(3, stride=stride, padding=1, op='mean')
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv(width, width, 3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv((width * scale), (planes * self.expansion), 1, bias=False)
        self.bn3 = nn.BatchNorm((planes * self.expansion))
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        spx = jt.split(out, self.width, 1)
        for i in range(self.nums):
            if ((i == 0) or (self.stype == 'stage')):
                sp = spx[i]
            else:
                sp = (sp + spx[i])
            sp = self.convs[i](sp)
            sp = nn.relu(self.bns[i](sp))
            if (i == 0):
                out = sp
            else:
                out = jt.contrib.concat((out, sp), dim=1)
        if ((self.scale != 1) and (self.stype == 'normal')):
            out = jt.contrib.concat((out, spx[self.nums]), dim=1)
        elif ((self.scale != 1) and (self.stype == 'stage')):
            out = jt.contrib.concat((out, self.pool(spx[self.nums])), dim=1)
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = nn.relu(out)
        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm(32), nn.ReLU(),
                                   nn.Conv(32, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm(32), nn.ReLU(),
                                   nn.Conv(32, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(3, stride=2, padding=1, op='maximum')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear((512 * block.expansion), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm):
                init.constant_(m.weight, value=1)
                init.constant_(m.bias, value=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if ((stride != 1) or (self.inplanes != (planes * block.expansion))):
            downsample = nn.Sequential(
                nn.Pool(stride, stride=stride, ceil_mode=True, op='mean'),
                nn.Conv(self.inplanes, (planes * block.expansion), 1, stride=1, bias=False),
                nn.BatchNorm((planes * block.expansion))
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth,
                  scale=self.scale))

        self.inplanes = (planes * block.expansion)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view((x.shape[0], (- 1)))
        x = self.fc(x)
        return x


def res2net50_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load(jt.load(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load(jt.load(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net50_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load(jt.load(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b_26w_4s(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load(jt.load((model_urls['res2net101_v1b_26w_4s'])))
    return model

if __name__ == '__main__':
    images = jt.rand(1, 3, 352, 352)
    model = res2net50_26w_4s(pretrained=False)
    model = model
    print(model(images).shape)
