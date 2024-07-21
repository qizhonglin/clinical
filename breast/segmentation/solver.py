# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
import os
import time
import matplotlib.pyplot as plt
from pprint import pprint
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils.AverageMeter import AverageMeter
from utils.util import get_logger
from utils.util4torch import get_device, net2device, save_checkpoint, resume_checkpoint
from breast.config import CHECKPOINT_DIR
from breast.segmentation.model import ResNetUNet
from breast.segmentation.dataset import get_dataset_train_val_test


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    diceloss = dice_loss(pred, target)

    loss = bce * bce_weight + diceloss * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice loss'] += diceloss.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss, 1-diceloss


def train_config(config, train_ds, val_ds, checkpoint_dir):
    batch_size = int(config["batch_size"])
    num_epochs = config["num_epochs"]

    logger = get_logger(name='clinical')
    writer = SummaryWriter(log_dir=checkpoint_dir)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    net = ResNetUNet(1)
    net2device(net)

    optimizer = getattr(torch.optim, config["optimizer"])(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = lr_scheduler.MultiStepLR(optimizer, [int(num_epochs * 0.8)], gamma=0.1)

    start_epoch = resume_checkpoint(net, optimizer, checkpoint_dir=checkpoint_dir)
    best = 0
    for epoch in range(start_epoch, num_epochs + 1):

        logger.info(f'train at epoch {epoch}')
        net.train()
        train_loss = AverageMeter()
        dices = AverageMeter()
        metrics = defaultdict(float)
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(get_device()), labels.to(get_device())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss, dice = calc_loss(outputs, labels, metrics)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss.update(loss.item())
            dices.update(dice.item(), inputs.size(0))
            writer.add_scalar("Loss/train/batch", train_loss.avg, (epoch - 1) * len(trainloader) + i)
            if i % 10 == 9:  # print every 10 mini-batches
                lr = optimizer.param_groups[0]['lr']
                info = f'Epoch: [{epoch}/{num_epochs}, {i + 1}/{len(trainloader)}] loss: {train_loss.avg: .4f}\tDice {dices.val:.4f} ({dices.avg:.4f})\tlr: {lr}'
                logger.info(info)

        save_checkpoint(epoch, net, optimizer, checkpoint_dir=checkpoint_dir)
        scheduler.step()

        # Validation loss
        logger.info(f'val at epoch {epoch}')
        net.eval()
        val_loss = AverageMeter()
        val_dice = AverageMeter()
        metrics = defaultdict(float)
        for i, (inputs, labels) in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = inputs.to(get_device()), labels.to(get_device())

                outputs = net(inputs)
                loss, dice = calc_loss(outputs, labels, metrics)

                val_loss.update(loss.item())
                val_dice.update(dice.item(), inputs.size(0))

        info = f'Epoch: [{epoch}/{num_epochs}, val loss: {val_loss.avg: .4f}, val dice: {val_dice.avg: .4f}'
        logger.info(info)

        writer.add_scalars("Loss", {'train': train_loss.avg, 'val': val_loss.avg}, epoch)
        writer.add_scalar("Dice/val", val_dice.avg, epoch)
        writer.flush()

        if val_dice.avg >= best:
            logger.info(f'the model of epoch {epoch} is improved and saved from val dice = {best} to {val_dice.avg}')
            best = val_dice.avg
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

    writer.close()
    print("Finished Training")


def infer(net, test_int_ds):
    valloader = DataLoader(test_int_ds, batch_size=4, shuffle=False, num_workers=8)

    net.eval()
    # dices = []
    truths = []
    preds = []
    for i, (inputs, labels) in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = inputs.to(get_device()), labels.to(get_device())

            outputs = net(inputs)
            pred = F.sigmoid(outputs)

            truths.append(labels)
            preds.append(pred)

    #         diceloss = dice_loss(pred, labels)
    #         dice = 1 - diceloss
    #         dices.append(dice)
    #
    # dices = torch.cat(dices, dim=0)
    # dices = dices.data.cpu().numpy()
    # return dices

    truths = torch.cat(truths, dim=0)
    truths = truths.data.cpu().numpy()
    preds = torch.cat(preds, dim=0)
    preds = preds.data.cpu().numpy()
    return truths, preds



def main():
    CHECKPOINT_DIR_NORMAL = os.path.join(CHECKPOINT_DIR, 'normal')
    os.makedirs(CHECKPOINT_DIR_NORMAL, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(CHECKPOINT_DIR_NORMAL, f'{timestamp}.log')
    logger = get_logger(name='clinical', log_file=log_file, log_level='INFO')

    train_ds, val_ds, test_int_ds = get_dataset_train_val_test()

    logger.info(
        f"length of train {len(train_ds)}, length of val {len(val_ds)}, length of internal test {len(test_int_ds)}")

    config = {
        "backbone": 'resnet18',
        "fix_depth": 1,
        "optimizer": "Adam",
        "lr": 0.001,
        "weight_decay": 1e-4,
        "drop_out": 0.5,
        "hidden_dim": 0,
        "batch_size": 4,
        "num_epochs": 20
    }
    logger.info(f"hyperparameter is {config}")

    # train_config(config, train_ds, val_ds, CHECKPOINT_DIR_NORMAL)

    net = ResNetUNet(1)
    net2device(net)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR_NORMAL, 'best.pth')))

    dices = infer(net, test_int_ds)



if __name__ == '__main__':
    main()
