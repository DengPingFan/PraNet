import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc
import random
import cv2
import copy

random.seed(42)
global best_epoch
global best_loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, writer=None):
    model.train()
    #best_loss = copy.deepcopy(best_loss)
    #best_epoch = copy.deepcopy(best_epoch)
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    with tqdm(total=len(train_loader), desc=f'Train Epoch {epoch}/{opt.epoch}', unit='img') as pbar:
        for i, pack in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                # ---- data prepare ----
                imgs, gts = pack

                import numpy as np
                images = np.zeros_like(imgs)
                # ---- blur image making
                for bi in range(imgs.shape[0]):
                    image = cv2.blur(imgs[bi].numpy(), (7, 7))
                    images[bi] = image
                #images = torch.from_numpy(images)
                # ---- sharpening image ----
                #sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                #for bi in range(imgs.shape[0]):
                #    image = cv2.filter2D(imgs[bi].numpy(), -1, sharpening_mask)
                #    images[bi] = image
                images = torch.from_numpy(images)

                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- mask making ----
                b, c, h, w = gts.shape
                mask = torch.zeros([b, 3, h, w])
                for m in range(b):
                    rgb_mask = torch.cat((gts[m], gts[m], gts[m]), 0)
                    mask[m] = rgb_mask
                new_img = np.array(images.cpu())
                new_gts = np.array(gts.cpu())
                new_mask = np.array(mask)
                mask = mask.cuda()
                # ---- forward ----
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images, mask)
                # ---- loss function ----
                loss5 = structure_loss(lateral_map_5, gts)
                loss4 = structure_loss(lateral_map_4, gts)
                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record2.update(loss2.data, opt.batchsize)
                    loss_record3.update(loss3.data, opt.batchsize)
                    loss_record4.update(loss4.data, opt.batchsize)
                    loss_record5.update(loss5.data, opt.batchsize)
                # ---- visualize ----
                if writer is not None:
                    writer.add_scalar('Loss/Train', loss.item(), epoch)
                    #writer.add_scalar('Loss/Train_map2', loss2, train_step)
                    #writer.add_scalar('Loss/Train_map3', loss3, train_step)
                    #writer.add_scalar('Loss/Train_map4', loss4, train_step)
                    #writer.add_scalar('Loss/Train_map5', loss5, train_step)
                    writer.add_scalars('Loss/Train_Maps', {'Map2': loss2, 'Map3': loss3, 'Map4': loss4, 'Map5': loss5}, epoch)
            # ---- train visualization ----
            #if i == train_step:
            #    print('{} Train_Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
            #          '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
            #          format(datetime.now(), epoch, opt.epoch, i, train_step,
            #                 loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update(images.shape[0])
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)


def eval(val_loader, model, optimizer, epoch, best_epoch, best_loss, writer=None):
    model.eval()
    # ---- multi-scale training ----
    #size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    with tqdm(total=len(val_loader), desc=f'Validation round', unit='batch') as pbar:
        for i, pack in enumerate(val_loader, start=1):
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- mask making ----
            b, c, h, w = gts.shape
            mask = torch.zeros([b, 3, h, w])
            for m in range(b):
                rgb_mask = torch.cat((gts[m], gts[m], gts[m]), 0)
                mask[m] = rgb_mask
            mask = mask.cuda()
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images, mask)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
            # ---- backward ----
            #loss.backward()
            #clip_gradient(optimizer, opt.clip)
            #optimizer.step()
            # ---- recording loss ----
            loss_record2.update(loss2.data, opt.batchsize)
            loss_record3.update(loss3.data, opt.batchsize)
            loss_record4.update(loss4.data, opt.batchsize)
            loss_record5.update(loss5.data, opt.batchsize)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            pbar.update()
            # ---- visualize ----
            if writer is not None:
                writer.add_scalar('Loss/Validation', loss.item(), epoch)
                #writer.add_scalar('Loss/Validation_map2', loss2, train_step)
                #writer.add_scalar('Loss/Validation_map3', loss3, train_step)
                #writer.add_scalar('Loss/Validation_map4', loss4, train_step)
                #writer.add_scalar('Loss/Validation_map5', loss5, train_step)
                writer.add_scalars('Loss/Validation_Maps', {'Map2': loss2, 'Map3': loss3, 'Map4': loss4, 'Map5': loss5}, epoch)
            # ---- val visualization ----
            #if i == val_step:
            #    print('{} Val_Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
            #          '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
            #          format(datetime.now(), epoch, opt.epoch, i, val_step,
            #                 loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch
        print(f'best_epoch:{best_epoch}  best_loss:{best_loss}')
        save_path = 'snapshots/{}/'.format(opt.train_save)
        torch.save(model.state_dict(), save_path + 'PraNet_BestLoss.pth')
    return best_epoch, best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_epoch', type=int,
                        default=0)
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=128, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--train_rate', type=int,
                        default=0.8, help='rate of train and validation data')
    parser.add_argument('--train_path', type=str,
                        default='./data/SIJ', help='path to train dataset')
    parser.add_argument('--ra', type=str,
                        default='ra5', help='ra5, ra4, ra3')
    parser.add_argument('--train_save', type=str,
                        default='512_blur')
    parser.add_argument('--visualize', type=str,
                        default=False)
    parser.add_argument('--tensorboard', type=str,
                        default='./run/512_blur')
    parser.add_argument('--load', type=bool,
                        default=False)
    parser.add_argument('--load_path', type=str,
                        default='./snapshots/256_ra5/PraNet-129.pth')
    opt = parser.parse_args()

    torch.cuda.empty_cache()
    gc.collect()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = PraNet(ra=opt.ra).cuda()

    # ---- load models ----
    if opt.load:
        model.load_state_dict(torch.load(opt.load_path))

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader, val_loader = get_loader(image_root, gt_root, rate=opt.train_rate, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_step = len(train_loader)
    val_step = len(val_loader)

    print("#"*20, "Start Training", "#"*20)

    if opt.visualize:
        writer = SummaryWriter(log_dir=opt.tensorboard)
    else:
        writer = None

    best_loss = 100
    best_epoch = 0

    for epoch in range(opt.load_epoch, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, writer)
        best_epoch, best_loss = eval(val_loader, model, optimizer, epoch, best_epoch, best_loss, writer)
        torch.cuda.empty_cache()

