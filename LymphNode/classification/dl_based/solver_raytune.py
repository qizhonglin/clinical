# -*- coding: utf-8 -*-
import os
import time
import json
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from utils.util import get_logger
from utils.AverageMeter import AverageMeter
from utils.util4raytune import resume_checkpoint, save_checkpoint
from LymphNode.config import CHECKPOINT_DIR, RAYTUNE, is_train_with_external_data
from LymphNode.classification.dl_based.solver import (
    get_device,
    net2device,
    get_dataset_train_val_test,
    get_dataset_external_test,
    split_train_val_test_ext,
    split_train_val_test,
    get_data,
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    classes,
    get_model,
    infer_each_class,
    SensitivitySpecificityStatistics,
    save_checkpoint as _save_checkpoint,
    resume_checkpoint as _resume_checkpoint
)
from LymphNode.classification.dl_based.dataset import get_dataset_train_val_test_ext
from LymphNode.config import DATA_DIR

# below imports are used to print out pretty pandas dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
pd.set_option('display.width', 1024)  # console defaults to 80 characters


def template_train_config(config, data):
    batch_size = int(config["batch_size"])
    num_epochs = config["num_epochs"]

    if is_train_with_external_data:
        train_ds, val_ds, _, _ = get_dataset_train_val_test_ext(data)
    else:
        train_ds, val_ds, _ = get_dataset_train_val_test(data)
    info = {
        'train num': len(train_ds),
        'val num': len(val_ds),
        'val': [img.replace(DATA_DIR, "") for img in val_ds.images]
    }
    print(info)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    net = get_model(config["fix_depth"], config["backbone"], len(classes), config["drop_out"], config["hidden_dim"])
    net2device(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, config["optimizer"])(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = lr_scheduler.MultiStepLR(optimizer, [int(num_epochs * 0.8)], gamma=0.1)

    start_epoch = resume_checkpoint(net, optimizer)
    for epoch in range(start_epoch, num_epochs + 1):  # loop over the dataset multiple times

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
            if i % 200 == 199:  # print every 200 mini-batches
                lr = optimizer.param_groups[0]['lr']
                info = f'Epoch: [{epoch}/{num_epochs}, {i + 1}/{len(trainloader)}] loss: {train_loss.avg: .4f}\tlr: {lr}'
                print(info)

        scheduler.step()

        # Validation loss
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
        print(info)

        metrics = {"val_loss": val_loss.avg, "val_acc": val_acc.avg, "train_loss": train_loss.avg}
        save_checkpoint(epoch, net, optimizer, metrics)

    print("Finished Training")


def search_optimal_grid(param_space, num_samples, cpus_per_trial=4, gpus_per_trial=1, scheduler=ASHAScheduler()):
    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)

    tuner = tune.Tuner(
        tune.with_resources(train_config, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_acc",
            mode="max",
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            storage_path=CHECKPOINT_DIR,
            name=RAYTUNE
        )
    )
    result_grid = tuner.fit()

    return result_grid


def search_optimal_hyperopt(param_space, num_samples, cpus_per_trial=4, gpus_per_trial=1, scheduler=ASHAScheduler()):
    from ray.tune.search.hyperopt import HyperOptSearch
    initial_params = [
        {
            "backbone": 'resnet18',
            "fix_depth": 1,
            "optimizer": "Adam",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "drop_out": 0.5,
            "hidden_dim": 8,
            "batch_size": 32,
            "num_epochs": 20
        }
    ]
    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)
    tuner = tune.Tuner(
        tune.with_resources(train_config, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_acc",
            mode="max",
            search_alg=HyperOptSearch(points_to_evaluate=initial_params),
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            storage_path=CHECKPOINT_DIR,
            name=RAYTUNE
        )
    )
    result_grid = tuner.fit()

    return result_grid


def search_optimal_bayesopt(num_samples, cpus_per_trial=4, gpus_per_trial=1, scheduler=ASHAScheduler()):
    from ray.tune.search.bayesopt import BayesOptSearch
    param_space = {
        "backbone": 'resnet18',
        "fix_depth": 1,
        "optimizer": "Adam",
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.uniform(1e-4, 1e-2),
        "drop_out": tune.uniform(0.1, 0.7),
        "hidden_dim": 8,
        "batch_size": 32,
        "num_epochs": 20
    }
    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)
    tuner = tune.Tuner(
        tune.with_resources(train_config, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_acc",
            mode="max",
            search_alg=BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}),
            scheduler=scheduler
        ),
        run_config=ray.train.RunConfig(
            storage_path=CHECKPOINT_DIR,
            name=RAYTUNE
        )
    )
    result_grid = tuner.fit()

    return result_grid


def search_optimal_tunebohb(param_space, num_samples, cpus_per_trial=4, gpus_per_trial=1):
    from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
    from ray.tune.search.bohb import TuneBOHB

    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)

    tuner = tune.Tuner(
        tune.with_resources(train_config, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_acc",
            mode="max",
            search_alg=TuneBOHB(),
            scheduler=HyperBandForBOHB(),
        ),
        run_config=ray.train.RunConfig(
            storage_path=CHECKPOINT_DIR,
            name=RAYTUNE
        )
    )
    result_grid = tuner.fit()

    return result_grid


def search_optimal_optuna(param_space, num_samples, cpus_per_trial=4, gpus_per_trial=1, scheduler=ASHAScheduler()):
    from ray.tune.search.optuna import OptunaSearch

    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)
    tuner = tune.Tuner(
        tune.with_resources(train_config, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_acc",
            mode="max",
            search_alg=OptunaSearch(),
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            storage_path=CHECKPOINT_DIR,
            name=RAYTUNE
        )
    )
    result_grid = tuner.fit()

    return result_grid


def search_optimal_pbt(param_space, num_samples, cpus_per_trial=4, gpus_per_trial=1):
    from ray.tune.schedulers import PopulationBasedTraining

    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)

    scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        perturbation_interval=1,
        hyperparam_mutations=param_space
    )

    tuner = tune.Tuner(
        tune.with_resources(train_config, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_acc",
            mode="max",
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            storage_path=CHECKPOINT_DIR,
            name=RAYTUNE
        )
    )
    result_grid = tuner.fit()

    return result_grid


def search_optimal_nevergrad(param_space, num_samples, cpus_per_trial=4, gpus_per_trial=1, scheduler=ASHAScheduler()):
    from ray.tune.search.nevergrad import NevergradSearch
    import nevergrad as ng

    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)

    tuner = tune.Tuner(
        tune.with_resources(train_config, {"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            metric="val_acc",
            mode="max",
            search_alg=NevergradSearch(optimizer=ng.optimizers.OnePlusOne),
            scheduler=scheduler,
        ),
        run_config=ray.train.RunConfig(
            storage_path=CHECKPOINT_DIR,
            name=RAYTUNE
        )
    )
    result_grid = tuner.fit()

    return result_grid


def analyze_experiment_results(result_grid):
    for i, result in enumerate(result_grid):  # Iterate over results
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
        else:
            print(f"Trial #{i} finished successfully with a mean accuracy metric of:", result.metrics["val_acc"])

    df_results = result_grid.get_dataframe()  # Get last results for each trial
    [print(df) for df in df_results]

    ax = None
    for result in result_grid:
        label = f"fix_depth={result.config['fix_depth']}, lr={result.config['lr']:.3f}, batch_size={result.config['batch_size']}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "val_acc", label=label)
        else:
            result.metrics_dataframe.plot("training_iteration", "val_acc", ax=ax, label=label)
    ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
    ax.set_ylabel("Mean Val Accuracy")

    best_trial = result_grid.get_best_result(metric="val_acc", mode="max")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial path: {best_trial.path}")
    print(f"Best trial metrics: {best_trial.metrics}")

    return best_trial


def tune_hyperparameter():
    from ray.tune.schedulers import ASHAScheduler
    scheduler = ASHAScheduler()  # prefer

    # from ray.tune.schedulers import AsyncHyperBandScheduler
    # scheduler = AsyncHyperBandScheduler()
    # from ray.tune.schedulers import HyperBandScheduler
    # scheduler = HyperBandScheduler()

    param_space = {
        "backbone": 'resnet18',
        "fix_depth": tune.choice([1, 2, 3, 4]),
        "optimizer": "Adam",
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.uniform(1e-4, 1e-2),
        "drop_out": tune.uniform(0.1, 0.7),
        "hidden_dim": tune.choice([0, 32, 64]),
        "batch_size": 32,
        "num_epochs": 20
    }

    result_grid = search_optimal_tunebohb(param_space, num_samples=40, cpus_per_trial=2, gpus_per_trial=2 / 4)  # prefer
    # result_grid = search_optimal_hyperopt(param_space, num_samples=8, cpus_per_trial=2, gpus_per_trial=2/4)
    # result_grid = search_optimal_bayesopt(num_samples=8, cpus_per_trial=2, gpus_per_trial=2 / 4)
    # result_grid = search_optimal_nevergrad(param_space, num_samples=8, cpus_per_trial=2, gpus_per_trial=2 / 4)
    # result_grid = search_optimal_optuna(param_space, num_samples=8, cpus_per_trial=2, gpus_per_trial=2 / 4)
    # result_grid = search_optimal_pbt(param_space, num_samples=8, cpus_per_trial=2, gpus_per_trial=2 / 4)
    # result_grid = search_optimal_grid(param_space, num_samples=1, cpus_per_trial=2, gpus_per_trial=2/4)

    # analyze experiment results
    experiment_path = os.path.join(CHECKPOINT_DIR, RAYTUNE)
    print(f"Loading results from {experiment_path}...")
    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    else:
        data = split_train_val_test(DATA_DIR)
    train_config = partial(template_train_config, data=data)
    restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_config)
    result_grid = restored_tuner.get_results()
    best_trial = analyze_experiment_results(result_grid)

    # best trained model from {CHECKPOINT_DIR}/{RAYTUNE}/checkpoint.pt
    config = best_trial.config
    best_trained_model = get_model(config["fix_depth"], config["backbone"], len(classes), config["drop_out"],
                                   config["hidden_dim"])
    net2device(best_trained_model)
    with best_trial.checkpoint.as_directory() as checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        best_trained_model.load_state_dict(checkpoint["net_state_dict"])

    # best hyperparameter to {CHECKPOINT_DIR}/{RAYTUNE}/best_config.json
    config_file = os.path.join(CHECKPOINT_DIR, RAYTUNE, 'best_config.json')
    with open(config_file, 'w') as ftxt:
        json.dump(best_trial.config, ftxt, indent=4)
    config = json.loads(open(config_file).read())

    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
        _, _, test_int_ds, test_ext_ds = get_dataset_train_val_test_ext(data)
    else:
        data = split_train_val_test(DATA_DIR)
        _, _, test_int_ds = get_dataset_train_val_test(data)
        test_ext_ds = get_dataset_external_test(*get_data(EXTERNAL_DATA_DIR))

    # internal test
    truth, probs = infer_each_class(best_trained_model, test_int_ds, classes)
    if len(classes) == 2:
        SensitivitySpecificityStatistics(truth, probs[:, 1], 'internal-test-subtrain')

    # external test
    truth, probs = infer_each_class(best_trained_model, test_ext_ds, classes)
    if len(classes) == 2:
        SensitivitySpecificityStatistics(truth, probs[:, 1], 'external-test-subtrain')


def train_with_best_hypermaters(config, train_ds, checkpoint_dir):
    batch_size = int(config["batch_size"])
    num_epochs = config["num_epochs"]

    logger = get_logger(name='clinical')

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)

    net = get_model(config["fix_depth"], config["backbone"], len(classes), config["drop_out"], config["hidden_dim"])
    net2device(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, config["optimizer"])(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = lr_scheduler.MultiStepLR(optimizer, [int(num_epochs * 0.8)], gamma=0.1)

    start_epoch = _resume_checkpoint(net, optimizer, checkpoint_dir=checkpoint_dir)
    for epoch in range(start_epoch, num_epochs + 1):

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
            if i % 10 == 9:  # print every 10 mini-batches
                lr = optimizer.param_groups[0]['lr']
                info = f'Epoch: [{epoch}/{num_epochs}, {i + 1}/{len(trainloader)}] loss: {train_loss.avg: .4f}\tlr: {lr}'
                logger.info(info)

        _save_checkpoint(epoch, net, optimizer, checkpoint_dir=checkpoint_dir)
        scheduler.step()

    torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

    print("Finished Training")


def main():
    # # tune hyper parameter
    # tune_hyperparameter()

    # get optimal hyperparameter
    config_file = os.path.join(CHECKPOINT_DIR, RAYTUNE, 'best_config.json')
    config = json.loads(open(config_file).read())

    # train the whole train dataset with optimal hyperparameter
    config["num_epochs"] = int(config["num_epochs"] * 1.2)
    CHECKPOINT_DIR_BEST_HYPERPARA = os.path.join(CHECKPOINT_DIR, RAYTUNE, 'best_hyperparameter')
    os.makedirs(CHECKPOINT_DIR_BEST_HYPERPARA, exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(CHECKPOINT_DIR_BEST_HYPERPARA, f'{timestamp}.log')
    logger = get_logger(name='clinical', log_file=log_file, log_level='INFO')
    logger.info(f"hyperparameter is {config}")

    if is_train_with_external_data:
        data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR, ratio_val=0)
        train_ds, _, test_int_ds, test_ext_ds = get_dataset_train_val_test_ext(data)
    else:
        data = split_train_val_test(DATA_DIR, ratio_val=0)
        train_ds, _, test_int_ds = get_dataset_train_val_test(data)
        test_ext_ds = get_dataset_external_test(*get_data(EXTERNAL_DATA_DIR))
    logger.info(
        f"length of train {len(train_ds)}, length of val 0, length of internal test {len(test_int_ds)}, length of external test {len(test_ext_ds)}")

    train_with_best_hypermaters(config, train_ds, CHECKPOINT_DIR_BEST_HYPERPARA)

    net = get_model(config["fix_depth"], config["backbone"], len(classes), config["drop_out"], config["hidden_dim"])
    net2device(net)
    net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR_BEST_HYPERPARA, 'best.pth')))

    truth, probs = infer_each_class(net, test_int_ds, classes)
    if len(classes) == 2:
        SensitivitySpecificityStatistics(truth, probs[:, 1], 'internal-test-dl')

    truth, probs = infer_each_class(net, test_ext_ds, classes)
    if len(classes) == 2:
        SensitivitySpecificityStatistics(truth, probs[:, 1], 'external-test-dl')


if __name__ == "__main__":
    main()

    plt.show()
