import os, argparse
import imageio

import jittor as jt
from jittor import nn

from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset

jt.flags.use_cuda = 1


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet-ori.pth')
opt = parser.parse_args()

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/PraNet/{}/'.format(_data_name)
    model = PraNet()
    model.load(opt.pth_path)
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize) \
        .set_attrs(batch_size=1, shuffle=False)

    for image, gt, name in test_loader:
        gt /= (gt.max() + 1e-08)
        (res5, res4, res3, res2) = model(image)

        res = res2
        c, h, w = gt.shape
        upsample = nn.upsample(res, size=(h, w), mode='bilinear')
        res = res.sigmoid().data.squeeze()
        res = ((res - res.min()) / ((res.max() - res.min()) + 1e-08))
        print('> {} - {}'.format(_data_name, name))
        imageio.imwrite((save_path + name[0]), res)