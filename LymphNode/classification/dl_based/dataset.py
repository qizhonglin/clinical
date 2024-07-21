# -*- coding: utf-8 -*-
import os

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

from LymphNode.config import DATA_DIR, EXTERNAL_DATA_DIR, classes, random_seed
from LymphNode.clean_data.split_data import split_train_val_test, get_data, split_train_val_test_ext
from transform import RandomCenterCrop, RandomTopCrop


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


def train_transform(im_h, im_w):
    transform_list = [
        transforms.ToTensor(),      # will normalize to (0, 1) by divided 255
        transforms.Grayscale(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),

        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),

        # # for echo
        # transforms.RandomRotation(45),
        # RandomTopCrop(),
        # transforms.Resize((im_h, im_w)),

        # for lesion in the center of image
        transforms.RandomRotation(45),
        # RandomCenterCrop(),
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



def get_dataset_train_val_test(data):
    image_size = 224
    transform_train = train_transform(image_size, image_size)
    transform_val = val_transform(image_size, image_size)

    train_ds = CustomImageDataset(data["X_train"], data["y_train"], transform=transform_train)
    val_ds = CustomImageDataset(data["X_val"], data["y_val"], transform=transform_val)
    test_ds = CustomImageDataset(data["X_test"], data["y_test"], transform=transform_val)

    return train_ds, val_ds, test_ds


def get_dataset_external_test(images, labels):
    image_size = 224
    transform_val = val_transform(image_size, image_size)

    test_ds = CustomImageDataset(images, labels, transform=transform_val)

    return test_ds


def get_dataset_train_val_test_ext(data):
    image_size = 224
    transform_train = train_transform(image_size, image_size)
    transform_val = val_transform(image_size, image_size)

    train_ds = CustomImageDataset(data["X_train"], data["y_train"], transform=transform_train)
    val_ds = CustomImageDataset(data["X_val"], data["y_val"], transform=transform_val)
    test_int_ds = CustomImageDataset(data["X_test_int"], data["y_test_int"], transform=transform_val)
    test_ext_ds = CustomImageDataset(data["X_test_ext"], data["y_test_ext"], transform=transform_val)

    return train_ds, val_ds, test_int_ds, test_ext_ds



if __name__ == '__main__':
    data = split_train_val_test(DATA_DIR)
    train_ds, val_ds, test_int_ds = get_dataset_train_val_test(data)
    test_ext_ds = get_dataset_external_test(*get_data(EXTERNAL_DATA_DIR))
    info = {
        'ratio_val': 0.2,
        'all': len(train_ds) + len(val_ds) + len(test_int_ds) + len(test_ext_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_int_ds),
        'external test': len(test_ext_ds)
    }
    print(info)
    data = split_train_val_test(DATA_DIR, ratio_val=0)
    train_ds, val_ds, test_int_ds = get_dataset_train_val_test(data)
    test_ext_ds = get_dataset_external_test(*get_data(EXTERNAL_DATA_DIR))
    info = {
        'ratio_val': 0,
        'all': len(train_ds) + len(val_ds) + len(test_int_ds) + len(test_ext_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_int_ds),
        'external test': len(test_ext_ds)
    }
    print(info)

    data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR)
    train_ds, val_ds, test_int_ds, test_ext_ds = get_dataset_train_val_test_ext(data)
    info = {
        'ratio_val': 0.2,
        'all': len(train_ds) + len(val_ds) + len(test_int_ds) + len(test_ext_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_int_ds),
        'external test': len(test_ext_ds)
    }
    print(info)

    data = split_train_val_test_ext(DATA_DIR, EXTERNAL_DATA_DIR, ratio_val=0)
    train_ds, val_ds, test_int_ds, test_ext_ds = get_dataset_train_val_test_ext(data)
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
