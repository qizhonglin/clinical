# -*- coding: utf-8 -*-
from functools import partial
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from src.utils.util4raytune import resume_checkpoint, save_checkpoint
from src.config import CHECKPOINT_DIR
from src.dl_based.solver import (get_device,
                             net2device,
                             get_dataset,
                             get_dataset_external_test,
                             classes,
                             Model,
                             infer_each_class,
                             SensitivitySpecificityStatistics)


def train_config(config, train_ds, val_ds, num_epochs):
    lr = config["lr"]
    batch_size = int(config["batch_size"])
    weight_decay = config["weight_decay"]
    fix_depth = config["fix_depth"]
    backbone = config["backbone"]
    optim = config["optimizer"]

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    net = Model(fix_depth=fix_depth).model_define(backbone=backbone, n_class=len(classes))
    net2device(net)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(
    #     filter(lambda p: p.requires_grad, net.parameters()),
    #     lr=lr, weight_decay=weight_decay)
    optimizer = getattr(torch.optim, optim)(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=weight_decay)

    start_epoch = resume_checkpoint(net, optimizer)
    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times
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
            if i % 200 == 199:  # print every 200 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
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

        save_checkpoint(epoch, net, optimizer, loss=val_loss/val_steps, accuracy=correct/total)

    print("Finished Training")


def search_optimal(train_ds, val_ds, num_samples, max_num_epochs, gpus_per_trial):
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    config = {
        "backbone": 'resnet18',
        "fix_depth": tune.choice([1, 2, 3]),
        "optimizer": "Adam",    #tune.choice(["SGD", "Adam", "AdamW"]),
        "lr": tune.choice([0.01, 0.001, 0.0001]),       #tune.loguniform(1e-4, 1e-2),
        "weight_decay": 1e-4, #tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32]),
    }
    result = tune.run(
        partial(train_config, train_ds=train_ds, val_ds=val_ds, num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir=CHECKPOINT_DIR,
    )

    # # bayesian optimization
    # config = {
    #     "backbone": 'resnet18',
    #     "fix_depth": tune.choice([1, 2]),
    #     "lr": tune.loguniform(1e-4, 1e-2),
    #     "weight_decay": 1e-4,
    #     "batch_size": tune.choice([16, 32]),
    # }
    # hyperopt_search = HyperOptSearch(config, metric="accuracy", mode="max")
    # result = tune.run(
    #     partial(train_config, train_ds=train_ds, val_ds=val_ds, num_epochs=max_num_epochs),
    #     resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
    #     num_samples=num_samples,
    #     scheduler=scheduler,
    #     search_alg=hyperopt_search
    # )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    fix_depth = best_trial.config["fix_depth"]
    backbone = best_trial.config["backbone"]
    best_trained_model = Model(fix_depth=fix_depth).model_define(backbone=backbone, n_class=len(classes))
    net2device(best_trained_model)
    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    return best_trial.config, best_trained_model


def main(num_samples, max_num_epochs, gpus_per_trial):
    model_file = os.path.join(CHECKPOINT_DIR,'best_raytune.pth')
    config_file = os.path.join(CHECKPOINT_DIR,'best_raytune.json')

    train_ds, val_ds, test_ds = get_dataset()

    config, best_trained_model = search_optimal(train_ds, val_ds, num_samples, max_num_epochs, gpus_per_trial)
    torch.save(best_trained_model.state_dict(), model_file)
    with open(config_file, 'w') as ftxt:
        json.dump(config, ftxt, indent=4)

    config = json.loads(open(config_file).read())
    fix_depth = config["fix_depth"]
    backbone = config["backbone"]
    best_trained_model = Model(fix_depth=fix_depth).model_define(backbone=backbone, n_class=len(classes))
    net2device(best_trained_model)
    best_trained_model.load_state_dict(torch.load(model_file))

    truth, probs = infer_each_class(best_trained_model, test_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'internal-test')

    external_test_ds = get_dataset_external_test()
    truth, probs = infer_each_class(best_trained_model, external_test_ds, classes)
    SensitivitySpecificityStatistics(truth, probs[:, 1], 'external-test')


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=18, max_num_epochs=20, gpus_per_trial=1)

    plt.show()