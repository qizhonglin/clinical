# -*- coding: utf-8 -*-
import os
from pathlib import Path
import tempfile

import torch

from ray.train import Checkpoint, get_checkpoint
from ray import train


def resume_checkpoint(net, optimizer):
    start_epoch = 1

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            net.load_state_dict(checkpoint_dict["net_state_dict"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    return start_epoch


def save_checkpoint(epoch, net, optimizer, metrics):
    checkpoint_data = {
        "epoch": epoch,
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        torch.save(checkpoint_data, os.path.join(checkpoint_dir, "checkpoint.pt"))
        train.report(metrics, checkpoint = Checkpoint.from_directory(checkpoint_dir))

