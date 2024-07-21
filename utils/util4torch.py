# -*- coding: utf-8 -*-

import os

import torch

from utils.util import symlink


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def net2device(net):
    device = get_device()
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
    net.to(device)


def save_checkpoint(epoch, model, optimizer, checkpoint_dir):
    [os.remove(os.path.join(checkpoint_dir, file)) for file in os.listdir(checkpoint_dir) if
     'epoch_' in file]
    save_file_path = os.path.join(checkpoint_dir, 'epoch_{}.pth'.format(epoch))
    states = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(states, save_file_path)

    # use relative symlink
    linkpath = os.path.join(checkpoint_dir, 'latest.pth')
    symlink(save_file_path, linkpath)


def resume_checkpoint(model, optimizer, checkpoint_dir):
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch
    model_file = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    return start_epoch

