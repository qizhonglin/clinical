# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from breast.config import DATA_DIR
from breast.clean_data.split_data import split_train_val_test


class BreastDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = [Image.open(file).convert('RGB') for file in images]
        self.labels = [Image.open(file) for file in labels]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform[0](image)
            label = self.transform[1](label)
            # label /= label.max()
            # print(torch.unique(label))

        return [image, label]


def train_transform(im_h, im_w):
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_mask = transforms.Compose([
        transforms.ToTensor()])

    return transform_image, transform_mask


def val_transform(im_h, im_w):
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_mask = transforms.Compose([
        transforms.ToTensor()])

    return transform_image, transform_mask


"""
get train from 60% DATA_DIR
get val from 20% DATA_DIR
get internal test from 20% DATA_DIR
"""
def get_dataset_train_val_test(data_dir=DATA_DIR, ratio_val=0.25):
    (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) = split_train_val_test(data_dir, ratio_val=ratio_val)

    image_size = 224
    transform_train = train_transform(image_size, image_size)
    transform_val = val_transform(image_size, image_size)

    train_ds = BreastDataset(images_train, labels_train, transform=transform_train)
    val_ds = BreastDataset(images_val, labels_val, transform=transform_val)
    test_ds = BreastDataset(images_test, labels_test, transform=transform_val)

    return train_ds, val_ds, test_ds