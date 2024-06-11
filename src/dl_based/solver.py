# -*- coding: utf-8 -*-
"""
split dataset into train (60%), val(20%), test(20%)
train model with train dataset, validate and save best model with val dataset
test model with best model (trained by 60%, validated by 20%)
"""


import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dl_based.dataset import get_dataset, get_dataset_external_test
from src.dl_based.Model2D import Model
from src.utils.util import get_logger
from src.utils.util4torch import get_device, net2device, save_checkpoint, resume_checkpoint
from src.config import CHECKPOINT_DIR, classes
from src.utils.SensitivitySpecificityStatistics import SensitivitySpecificityStatistics


def train_config(config, train_ds, val_ds=None, model_file=None, num_epochs=20):
    lr = config["lr"]
    batch_size = int(config["batch_size"])
    weight_decay = config["weight_decay"]
    fix_depth = config["fix_depth"]
    backbone = config["backbone"]

    logger = get_logger(name='haina')

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    if val_ds is not None:
        valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    net = Model(fix_depth=fix_depth).model_define(backbone=backbone, n_class=len(classes))
    net2device(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=weight_decay)

    # # scheduler (add scheduler, result get worse!!!)
    # step1, step2 = num_epochs * 7 / 10, num_epochs * 9 / 10
    # step1, step2 = int(step1), int(step2)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [step1, step2], gamma=0.1)

    start_epoch = 0
    if val_ds is not None:
        start_epoch = resume_checkpoint(net, optimizer, checkpoint_dir=CHECKPOINT_DIR)
    best_correct = 0
    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times

        net.train()
        running_loss = 0.0
        epoch_steps = 0
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
            running_loss += loss.item()
            epoch_steps += 1
            if i % 10 == 9:  # print every 10 mini-batches
                lr = optimizer.param_groups[0]['lr']
                info = f'Epoch: [{epoch + 1}, {i + 1}] loss: {running_loss / epoch_steps: .3f}\tlr: {lr}'
                logger.info(info)
                running_loss = 0.0

        if val_ds is not None:
            save_checkpoint(epoch, net, optimizer, checkpoint_dir=CHECKPOINT_DIR)

            # Validation loss
            net.eval()
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, (inputs, labels) in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = inputs.to(get_device()), labels.to(get_device())

                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            if correct >= best_correct:
                logger.info(f'the model of epoch {epoch} is improved and saved from correct = {best_correct} to {correct}')
                best_correct = correct
                torch.save(net.state_dict(), model_file)
        else:
            torch.save(net.state_dict(), model_file)        # save model for best solver_best_hyperparameter.py

        # scheduler.step()

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

def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(CHECKPOINT_DIR, f'{timestamp}.log')
    logger = get_logger(name='haina', log_file=log_file, log_level='INFO')

    train_ds, val_ds, test_ds = get_dataset()
    logger.info(
        f"length of train {len(train_ds)}, length of val {len(val_ds)}, length of test {len(test_ds)}")

    config = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "fix_depth": 1,
        "backbone": 'resnet18'
    }
    logger.info(f"hyperparameter is {config}")

    model_file = os.path.join(CHECKPOINT_DIR, 'best.pth')
    # train_config(config, train_ds, val_ds, model_file, num_epochs=20)

    net = Model().model_define(n_class=len(classes))
    net2device(net)
    net.load_state_dict(torch.load(model_file))

    truth, probs = infer_each_class(net, test_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'internal-test')

    external_test_ds = get_dataset_external_test()
    truth, probs = infer_each_class(net, external_test_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'external-test')


if __name__ == '__main__':
    main()

    plt.show()