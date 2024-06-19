# -*- coding: utf-8 -*-
"""
split dataset into train (60%), val(20%), test(20%)
train model with train dataset, validate and save best model with val dataset
test model with best model (trained by 60%, validated by 20%)
"""

import os
import time
import matplotlib.pyplot as plt
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from classification.dl_based.dataset import get_dataset_train_val_test, get_dataset_external_test, get_dataset_train_val_test_ext
from classification.dl_based.Model2D import get_model
from classification.utils.util import get_logger
from classification.utils.AverageMeter import AverageMeter
from classification.utils.util4torch import get_device, net2device, save_checkpoint, resume_checkpoint
from classification.config import CHECKPOINT_DIR, classes, is_train_with_external_data
from classification.utils.SensitivitySpecificityStatistics import SensitivitySpecificityStatistics


def train_config(config, train_ds, val_ds, checkpoint_dir):
    lr = config["lr"]
    batch_size = int(config["batch_size"])
    weight_decay = config["weight_decay"]
    fix_depth = config["fix_depth"]
    backbone = config["backbone"]
    num_epochs = config["num_epochs"]

    logger = get_logger(name='clinical')
    writer = SummaryWriter(log_dir=checkpoint_dir)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    net = get_model(fix_depth, backbone, len(classes))
    net2device(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [int(num_epochs * 0.8)], gamma=0.1)

    start_epoch = resume_checkpoint(net, optimizer, checkpoint_dir=checkpoint_dir)
    best = 0
    for epoch in range(start_epoch, num_epochs + 1):

        logger.info(f'train at epoch {epoch}')
        net.train()
        train_loss = AverageMeter()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(get_device()), labels.to(get_device())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss.update(loss.item())
            writer.add_scalar("Loss/train/batch", train_loss.avg, (epoch - 1) * len(trainloader) + i)
            if i % 10 == 9:  # print every 10 mini-batches
                lr = optimizer.param_groups[0]['lr']
                info = f'Epoch: [{epoch}/{num_epochs}, {i + 1}/{len(trainloader)}] loss: {train_loss.avg: .4f}\tlr: {lr}'
                logger.info(info)

        save_checkpoint(epoch, net, optimizer, checkpoint_dir=checkpoint_dir)
        scheduler.step()

        # Validation loss
        logger.info(f'val at epoch {epoch}')
        net.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        for i, (inputs, labels) in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = inputs.to(get_device()), labels.to(get_device())

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                num = labels.size(0)
                mean_val = (predicted == labels).sum().item() / num
                val_acc.update(mean_val, num)

                loss = criterion(outputs, labels)
                val_loss.update(loss.item())

        info = f'Epoch: [{epoch}/{num_epochs}, val loss: {val_loss.avg: .4f}, val acc: {val_acc.avg: .4f}'
        logger.info(info)

        writer.add_scalars("Loss", {'train': train_loss.avg, 'val': val_loss.avg}, epoch)
        writer.add_scalar("Accuracy/val", val_acc.avg, epoch)
        writer.flush()

        if val_acc.avg >= best:
            logger.info(f'the model of epoch {epoch} is improved and saved from val acc = {best} to {val_acc.avg}')
            best = val_acc.avg
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

    writer.close()
    print("Finished Training")


def infer_each_class(net, test_ds, classes):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    testloader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=8)

    truth = []
    probs = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(get_device()), labels.to(get_device())
            outputs = net(inputs)
            preds = torch.softmax(outputs, 1)
            truth.append(labels)
            probs.append(preds)

            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    truth = torch.cat(truth, dim=0)
    probs = torch.cat(probs, dim=0)
    truth = truth.data.cpu().numpy()
    probs = probs.data.cpu().numpy()
    return truth, probs


def analyze_log(log_path):
    # get event files
    files = []
    for (dirpath, dirnames, filenames) in os.walk(log_path):
        files.extend([os.path.join(dirpath, file) for file in filenames])
    files = [file for file in files if "events.out.tfevents" in file]

    # get result dictionary
    result = {}
    for file in files:
        event_accu = EventAccumulator(file)
        event_accu.Reload()
        print(file)
        print(f"Tags: {event_accu.Tags()}")

        parent_dir, name = os.path.split(file)
        parent_name = parent_dir.removeprefix(log_path)

        for name in event_accu.Tags()["scalars"]:
            events = event_accu.Scalars(name)
            result[name if not parent_name else parent_name] = [(e.step, e.value) for e in events]

    pprint(result)

    batchs = [epoch for epoch, value in result['Loss/train/batch']]
    train_loss_batch = [value for epoch, value in result['Loss/train/batch']]
    plt.plot(batchs, train_loss_batch, "b", label="Train loss batch")
    plt.xlabel("Batchs")
    plt.ylabel("Loss")
    plt.legend()

    epochs = [epoch for epoch, value in result['/Loss_train']]
    train_loss = [value for epoch, value in result['/Loss_train']]
    val_loss = [value for epoch, value in result['/Loss_val']]
    val_acc = [value for epoch, value in result['Accuracy/val']]
    plt.figure()
    plt.plot(epochs, train_loss, "bo", label="Train loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_loss, "bo", label="Train loss")
    plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()


def main(is_train_with_external_data=True):
    CHECKPOINT_DIR_NORMAL = os.path.join(CHECKPOINT_DIR,
                                         'normal_train_with_external_data' if is_train_with_external_data else 'normal_train_without_external_data')
    os.makedirs(CHECKPOINT_DIR_NORMAL, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(CHECKPOINT_DIR_NORMAL, f'{timestamp}.log')
    logger = get_logger(name='clinical', log_file=log_file, log_level='INFO')

    if is_train_with_external_data:
        train_ds, val_ds, test_int_ds, test_ext_ds = get_dataset_train_val_test_ext()
    else:
        train_ds, val_ds, test_int_ds = get_dataset_train_val_test()
        test_ext_ds = get_dataset_external_test()
    logger.info(
        f"length of train {len(train_ds)}, length of val {len(val_ds)}, length of internal test {len(test_int_ds)}, length of external test {len(test_ext_ds)}")

    config = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "fix_depth": 1,
        "backbone": 'resnet18',
        "num_epochs": 20,
    }
    logger.info(f"hyperparameter is {config}")

    train_config(config, train_ds, val_ds, CHECKPOINT_DIR_NORMAL)

    net = get_model(config["fix_depth"], config["backbone"], len(classes))
    net2device(net)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR_NORMAL, 'best.pth')))

    truth, probs = infer_each_class(net, test_int_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'internal-test')

    truth, probs = infer_each_class(net, test_ext_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'external-test')


if __name__ == '__main__':
    main(is_train_with_external_data)

    # # plot train/val loss to get optimal num_epochs, I recommend you play tensorboard with more functions.
    # analyze_log(os.path.join(CHECKPOINT_DIR, 'normal'))

    plt.show()
