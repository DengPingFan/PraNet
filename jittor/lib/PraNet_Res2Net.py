import jittor as jt
from jittor import nn

from lib.Res2Net_v1b import res2net50_26w_4s

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm(out_planes)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)), BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3))
        self.branch2 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)), BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)), BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5))
        self.branch3 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7))
        self.conv_cat = BasicConv2d((4 * out_channel), out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def execute(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(jt.contrib.concat((x0, x1, x2, x3), dim=1))
        x = nn.relu((x_cat + self.conv_res(x)))
        return x

class aggregation(nn.Module):

    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d((2 * channel), (2 * channel), 3, padding=1)
        self.conv_concat2 = BasicConv2d((2 * channel), (2 * channel), 3, padding=1)
        self.conv_concat3 = BasicConv2d((3 * channel), (3 * channel), 3, padding=1)
        self.conv4 = BasicConv2d((3 * channel), (3 * channel), 3, padding=1)
        self.conv5 = nn.Conv((3 * channel), 1, 1)

    def execute(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = (self.conv_upsample1(self.upsample(x1)) * x2)
        x3_1 = ((self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2))) * x3)
        x2_2 = jt.contrib.concat((x2_1, self.conv_upsample4(self.upsample(x1_1))), dim=1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = jt.contrib.concat((x3_1, self.conv_upsample5(self.upsample(x2_2))), dim=1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x

class PraNet(nn.Module):

    def __init__(self, channel=32, pretrained_backbone=False):
        super(PraNet, self).__init__()
        self.resnet = res2net50_26w_4s(pretrained=pretrained_backbone)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.agg1 = aggregation(channel)
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.upsample1_1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample1_2 = nn.Upsample(scale_factor=0.25, mode='bilinear')

        self.upsample2_1 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upsample2_2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.upsample3_1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample3_2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.upsample4 = nn.Upsample(scale_factor=8, mode='bilinear')

    def execute(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = nn.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = self.upsample1_1(ra5_feat)
        crop_4 = self.upsample1_2(ra5_feat)
        x = (((- 1) * jt.sigmoid(crop_4)) + 1)
        x = x.expand((- 1), 2048, (- 1), (- 1)).multiply(x4)
        x = self.ra4_conv1(x)
        x = nn.relu(self.ra4_conv2(x))
        x = nn.relu(self.ra4_conv3(x))
        x = nn.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = (ra4_feat + crop_4)
        lateral_map_4 = self.upsample2_1(x)
        crop_3 = self.upsample2_2(x)
        x = (((- 1) * jt.sigmoid(crop_3)) + 1)
        x = x.expand((- 1), 1024, (- 1), (- 1)).multiply(x3)
        x = self.ra3_conv1(x)
        x = nn.relu(self.ra3_conv2(x))
        x = nn.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = (ra3_feat + crop_3)
        lateral_map_3 = self.upsample3_1(x)
        crop_2 = self.upsample3_2(x)
        x = (((- 1) * jt.sigmoid(crop_2)) + 1)
        x = x.expand((- 1), 512, (- 1), (- 1)).multiply(x2)
        x = self.ra2_conv1(x)
        x = nn.relu(self.ra2_conv2(x))
        x = nn.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = (ra2_feat + crop_2)
        lateral_map_2 = self.upsample4(x)
        return (lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2)


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = PraNet()
    net.eval()

    dump_x = jt.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = (1 * float((1 / (end - start))))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
