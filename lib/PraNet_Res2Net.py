import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class PAM_Module(nn.Module):
    """ Position attention module """

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out



class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, ra='ra5'):
        super(aggregation, self).__init__()
        self.ra = ra
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        if self.ra == 'ra5' or self.ra == 'ra4':
            x1_1 = x1
            x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
            x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

            x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
            x2_2 = self.conv_concat2(x2_2)

            x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
            x3_2 = self.conv_concat3(x3_2)

        else:
            x1_1 = x1
            x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
            x3_1 = self.conv_upsample2(self.upsample(x1)) * self.conv_upsample3(x2) * x3

            x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
            x2_2 = self.conv_concat2(x2_2)

            x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
            x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x



class PraNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, ra='ra5'):
        super(PraNet, self).__init__()
        self.ra = ra
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=False)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel, self.ra)
        # ---- Receptive Field Block like module ----
        if ra == 'ra5':
            # ---- Receptive Field Block like module ----
            self.rfb2_1 = RFB_modified(512, channel)
            self.rfb3_1 = RFB_modified(1024, channel)
            self.rfb4_1 = RFB_modified(2048, channel)
            # ---- PAM ----
            self.pam2 = PAM_Module(512)
            self.pam3 = PAM_Module(1024)
            self.pam4 = PAM_Module(2048)
            # ---- reverse attention branch 4 ----
            self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
            self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
            self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
            self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
            self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
            # ---- reverse attention branch 3 ----
            self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
            self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
            # ---- reverse attention branch 2 ----
            self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
            self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        elif ra == 'ra4':
            # ---- Receptive Field Block like module ----
            self.rfb1_1 = RFB_modified(256, channel)
            self.rfb2_1 = RFB_modified(512, channel)
            self.rfb3_1 = RFB_modified(1024, channel)
            # ---- PAM ----
            self.pam1 = PAM_Module(256)
            self.pam2 = PAM_Module(512)
            self.pam3 = PAM_Module(1024)
            # ---- reverse attention branch 3 ----
            self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
            self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
            # ---- reverse attention branch 2 ----
            self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
            self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
            # ---- reverse attention branch 1 ----
            self.ra1_conv1 = BasicConv2d(256, 64, kernel_size=1)
            self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        elif ra == 'ra3':
            # ---- Receptive Field Block like module ----
            self.rfbx_1 = RFB_modified(64, channel)
            self.rfb1_1 = RFB_modified(256, channel)
            self.rfb2_1 = RFB_modified(512, channel)
            # ---- PAM ----
            self.pamx = PAM_Module(64)
            self.pam1 = PAM_Module(256)
            self.pam2 = PAM_Module(512)
            # ---- reverse attention branch 2 ----
            self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
            self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
            # ---- reverse attention branch 1 ----
            self.ra1_conv1 = BasicConv2d(256, 64, kernel_size=1)
            self.ra1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra1_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
            # ---- reverse attention branch x ----
            self.rax_conv1 = BasicConv2d(64, 32, kernel_size=1)
            self.rax_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
            self.rax_conv3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
            self.rax_conv4 = BasicConv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x, mask=None):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        x_x = x
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        if self.ra == 'ra5':
            x2_rfb = self.rfb2_1(x2)  # channel -> 32
            x3_rfb = self.rfb3_1(x3)  # channel -> 32
            x4_rfb = self.rfb4_1(x4)  # channel -> 32
            ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
            lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8,mode='bilinear')

            # ---- reverse attention branch_4 ----
            pam4 = self.pam4(x4)
            crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
            x = self.ra4_conv1(pam4)
            x = F.relu(self.ra4_conv2(x))
            x = F.relu(self.ra4_conv3(x))
            x = F.relu(self.ra4_conv4(x))
            ra4_feat = self.ra4_conv5(x)
            x = ra4_feat + crop_4
            lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')

            # ---- reverse attention branch_3 ----
            pam3 = self.pam3(x3)
            crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.ra3_conv1(pam3)
            x = F.relu(self.ra3_conv2(x))
            x = F.relu(self.ra3_conv3(x))
            ra3_feat = self.ra3_conv4(x)
            x = ra3_feat + crop_3
            lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')

            # ---- reverse attention branch_2 ----
            pam2 = self.pam2(x2)
            crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.ra2_conv1(pam2)
            x = F.relu(self.ra2_conv2(x))
            x = F.relu(self.ra2_conv3(x))
            ra2_feat = self.ra2_conv4(x)
            x = ra2_feat + crop_2
            lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')

            return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2

        elif self.ra == 'ra4':
            x1_rfb = self.rfb1_1(x1)
            x2_rfb = self.rfb2_1(x2)  # channel -> 32
            x3_rfb = self.rfb3_1(x3)  # channel -> 32
            ra5_feat = self.agg1(x3_rfb, x2_rfb, x1_rfb)
            lateral_map_5 = F.interpolate(ra5_feat, scale_factor=4, mode='bilinear')

            # ---- reverse attention branch_3 ----
            pam3 = self.pam3(x3)
            crop_3 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
            x = self.ra3_conv1(pam3)
            x = F.relu(self.ra3_conv2(x))
            x = F.relu(self.ra3_conv3(x))
            ra3_feat = self.ra3_conv4(x)
            x = ra3_feat + crop_3
            lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')

            # ---- reverse attention branch_2 ----
            pam2 = self.pam2(x2)
            crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.ra2_conv1(pam2)
            x = F.relu(self.ra2_conv2(x))
            x = F.relu(self.ra2_conv3(x))
            ra2_feat = self.ra2_conv4(x)
            x = ra2_feat + crop_2
            lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')

            # ---- reverse attention branch_1 ----
            pam1 = self.pam1(x1)
            crop_1 = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.ra1_conv1(pam1)
            x = F.relu(self.ra1_conv2(x))
            x = F.relu(self.ra1_conv3(x))
            ra1_feat = self.ra1_conv4(x)
            x = ra1_feat + crop_1
            lateral_map_1 = F.interpolate(x, scale_factor=4, mode='bilinear')

            return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1

        elif self.ra == 'ra3':
            x_rfb = self.rfbx_1(x_x)
            x1_rfb = self.rfb1_1(x1)
            x2_rfb = self.rfb2_1(x2)  # channel -> 32
            ra5_feat = self.agg1(x2_rfb, x1_rfb, x_rfb)
            lateral_map_5 = F.interpolate(ra5_feat, scale_factor=4, mode='bilinear')

            # ---- reverse attention branch_2 ----
            pam2 = self.pam2(x2)
            crop_2 = F.interpolate(ra5_feat, scale_factor=0.5, mode='bilinear')
            x = self.ra2_conv1(pam2)
            x = F.relu(self.ra2_conv2(x))
            x = F.relu(self.ra2_conv3(x))
            ra2_feat = self.ra2_conv4(x)
            x = ra2_feat + crop_2
            lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')

            # ---- reverse attention branch_1 ----
            pam1 = self.pam1(x1)
            crop_1 = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.ra1_conv1(pam1)
            x = F.relu(self.ra1_conv2(x))
            x = F.relu(self.ra1_conv3(x))
            ra1_feat = self.ra1_conv4(x)
            x = ra1_feat + crop_1
            lateral_map_1 = F.interpolate(x, scale_factor=4, mode='bilinear')

            pamx = self.pamx(x_x)
            crop_x = F.interpolate(x, scale_factor=1, mode='bilinear')
            x = self.rax_conv1(pamx)
            x = F.relu(self.rax_conv2(x))
            x = F.relu(self.rax_conv3(x))
            rax_feat = self.rax_conv4(x)
            x = rax_feat + crop_x
            lateral_map_x = F.interpolate(x, scale_factor=4, mode='bilinear')

            return lateral_map_5, lateral_map_2, lateral_map_1, lateral_map_x

        ## ---- Original ---- ##
        ''''# ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_2, lateral_map_1, lateral_map_x'''


if __name__ == '__main__':
    ras = PraNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)