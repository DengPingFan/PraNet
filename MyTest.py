import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os, argparse
from scipy import misc
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset
from torchvision.transforms.functional import to_pil_image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=320, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/256_ra5/PraNet_BestLoss.pth')
parser.add_argument('--mask_threshold', type=float, default=0.5)
parser.add_argument('--ra', type=str, default='ra5')


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

for _data_name in ['SIJ_Test']:
    opt = parser.parse_args()
    data_path = './data/{}/'.format(_data_name)
    save_path = './results/{}_{}_img/'.format(opt.testsize, opt.ra)
    ra5_path = './results/{}_{}_ra/'.format(opt.testsize, opt.ra)
    model = PraNet(ra=opt.ra)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ra5_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        torch.cuda.empty_cache()
        ori, image, name = test_loader.load_data()
        #gt = np.asarray(gt, np.float32)
        #gt /= (gt.max() + 1e-8)
        new_image, new_image5 = np.zeros((1024, 1024, 3)), np.zeros((1024, 1024, 3))
        new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2] = ori, ori, ori
        new_image5[:, :, 0], new_image5[:, :, 1], new_image5[:, :, 2] = ori, ori, ori
        img = to_pil_image(image.squeeze(0))
        rgb_img = np.array(img.convert('RGB'))
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=(1024, 1024), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        new_res = np.ones((res.shape[0], res.shape[1], 3))
        for row in range(res.shape[0]):
            for col in range(res.shape[1]):
                if res[row, col] > opt.mask_threshold:
                    new_res[row, col, 0] = 255.
                    new_image[row, col, 0] = 255.

        res5 = F.upsample(res5, size=(1024, 1024), mode='bilinear', align_corners=False)
        res5 = res5.sigmoid().data.cpu().numpy().squeeze()
        res5 = (res5 - res5.min()) / (res5.max() - res5.min() + 1e-8)

        new_res5 = np.ones((res5.shape[0], res5.shape[1], 3))
        for row in range(res5.shape[0]):
            for col in range(res5.shape[1]):
                if res5[row, col] > opt.mask_threshold:
                    new_res5[row, col, 0] = 255.
                    new_image5[row, col, 0] = 255.
                    #rgb_img[row, col] = 1.

        misc.imsave(save_path+name, new_image)
        misc.imsave(ra5_path+name, new_image5)