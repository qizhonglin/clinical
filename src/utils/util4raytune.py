# -*- coding: utf-8 -*-
from pathlib import Path
import tempfile

from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from ray import train

def resume_checkpoint(net, optimizer):
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    return start_epoch


def save_checkpoint(epoch, net, optimizer, loss, accuracy):
    checkpoint_data = {
        "epoch": epoch,
        "net_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)

        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            {"loss": loss, "accuracy": accuracy},
            checkpoint=checkpoint,
        )

