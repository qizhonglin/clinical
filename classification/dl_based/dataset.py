# -*- coding: utf-8 -*-
import os

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

from classification.config import DATA_DIR
from classification.split_data import split_train_val_test, get_data
from classification.config import EXTERNAL_DATA_DIR, classes, random_seed

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = self.images[item]
        image = Image.open(img_path)
        image = ImageOps.equalize(image)
        label = self.labels[item]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class RandomCenterCrop(object):
    def __init__(self, margin_ratio=0.5 / 4):
        self.margin_ratio = margin_ratio

    def __call__(self, tensor):
        _, h, w = tensor.shape
        x1 = random.randint(0, int(self.margin_ratio * w))
        y1 = random.randint(0, int(self.margin_ratio * h))
        cropped = tensor[:, y1:h - y1, x1:w - x1]
        return cropped


def train_transform(im_h, im_w):
    transform_list = [
        transforms.ToTensor(),      # will normalize to (0, 1) by divided 255
        transforms.Grayscale(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),

        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(45),
        # transforms.RandomCenterCrop(),
        transforms.Resize((im_h, im_w)),

        # transforms.RandomResizedCrop((im_h, im_w), scale=(1 / 4, 1.0)),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


def val_transform(im_h, im_w):
    transform_list = [
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),

        transforms.Resize((im_h, im_w)),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    return transforms.Compose(transform_list)


"""
get train from 60% DATA_DIR
get val from 20% DATA_DIR
get internal test from 20% DATA_DIR
get external test from 100% EXTERNAL_DATA_DIR
"""
def get_dataset_train_val_test(data_dir=DATA_DIR, ratio_val=0.25):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(data_dir, ratio_val=ratio_val)

    image_size = 224
    transform_train = train_transform(image_size, image_size)
    transform_val = val_transform(image_size, image_size)

    train_ds = CustomImageDataset(X_train, y_train, transform=transform_train)
    val_ds = CustomImageDataset(X_val, y_val, transform=transform_val)
    test_ds = CustomImageDataset(X_test, y_test, transform=transform_val)

    return train_ds, val_ds, test_ds


def get_dataset_external_test(data_dir=EXTERNAL_DATA_DIR):
    X_test, y_test = get_data(data_dir)

    image_size = 224
    transform_val = val_transform(image_size, image_size)

    test_ds = CustomImageDataset(X_test, y_test, transform=transform_val)

    return test_ds


"""
get train from 60% DATA_DIR and train 50% from EXTERNAL_DATA_DIR
get val from 20% DATA_DIR and 50% EXTERNAL_DATA_DIR
get internal test from 20% DATA_DIR
get external test from 50% EXTERNAL_DATA_DIR
"""
def get_dataset_train_val_test_ext(data_dir=DATA_DIR, ratio_val=0.25):
    (X_train, y_train), (X_val, y_val), (X_test_int, y_test_int) = split_train_val_test(data_dir, ratio_val=ratio_val)

    X_test_ext, y_test_ext = get_data(EXTERNAL_DATA_DIR)
    X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_test_ext, y_test_ext,
                                                                        test_size=0.5, random_state=random_seed,
                                                        shuffle=True, stratify=y_test_ext)

    if ratio_val > 0:
        X_train_ext, X_val_ext, y_train_ext, y_val_ext = train_test_split(X_train_ext, y_train_ext, test_size=0.5,
                                                                            random_state=random_seed,
                                                                            shuffle=True, stratify=y_train_ext)
        X_val.extend(X_val_ext)
        y_val.extend(y_val_ext)

    X_train.extend(X_train_ext)
    y_train.extend(y_train_ext)


    image_size = 224
    transform_train = train_transform(image_size, image_size)
    transform_val = val_transform(image_size, image_size)

    train_ds = CustomImageDataset(X_train, y_train, transform=transform_train)
    val_ds = CustomImageDataset(X_val, y_val, transform=transform_val)
    test_int_ds = CustomImageDataset(X_test_int, y_test_int, transform=transform_val)
    test_ext_ds = CustomImageDataset(X_test_ext, y_test_ext, transform=transform_val)

    return train_ds, val_ds, test_int_ds, test_ext_ds



if __name__ == '__main__':
    train_ds, val_ds, test_int_ds = get_dataset_train_val_test()
    test_ext_ds = get_dataset_external_test()
    info = {
        'ratio_val': 0.2,
        'all': len(train_ds) + len(val_ds) + len(test_int_ds) + len(test_ext_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_int_ds),
        'external test': len(test_ext_ds)
    }
    print(info)
    train_ds, val_ds, test_int_ds = get_dataset_train_val_test(ratio_val=0)
    test_ext_ds = get_dataset_external_test()
    info = {
        'ratio_val': 0,
        'all': len(train_ds) + len(val_ds) + len(test_int_ds) + len(test_ext_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_int_ds),
        'external test': len(test_ext_ds)
    }
    print(info)

    train_ds, val_ds, test_int_ds, test_ext_ds = get_dataset_train_val_test_ext()
    info = {
        'ratio_val': 0.2,
        'all': len(train_ds) + len(val_ds) + len(test_int_ds) + len(test_ext_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_int_ds),
        'external test': len(test_ext_ds)
    }
    print(info)

    train_ds, val_ds, test_int_ds, test_ext_ds = get_dataset_train_val_test_ext(ratio_val=0)
    info = {
        'ratio_val': 0,
        'all': len(train_ds) + len(val_ds) + len(test_int_ds) + len(test_ext_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_int_ds),
        'external test': len(test_ext_ds)
    }
    print(info)

    labels_map = {i: cls for i, cls in enumerate(classes)}

    trainloader = DataLoader(train_ds, batch_size=9, shuffle=True, num_workers=0)
    images, labels = next(iter(trainloader))


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    imshow(torchvision.utils.make_grid(images))

    # image_dir = "/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/LymphNode/huaxi/images/benign"
    # # image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    # # image = Image.open(image_files[0])
    #
    # image_file = os.path.join(image_dir, "3564638.jpg")
    # image = Image.open(image_file)
    # plt.imshow(image)

    # figure = plt.figure(figsize=(10, 10))
    # cols, rows = 3, 3
    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_ds), size=(1,)).item()
    #     img, label = train_ds[sample_idx]
    #
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(labels_map[label])
    #     plt.axis("off")
    #     plt.imshow(img.permute(1, 2, 0))
    #
    # plt.show()
