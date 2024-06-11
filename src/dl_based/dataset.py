# -*- coding: utf-8 -*-

from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from src.config import DATA_DIR, EXTERNAL_DATA_DIR
from src.split_data import split_train_val_test, split_train_test, get_data

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
        image = Image.open(img_path).convert('RGB')
        # image = Image.open(img_path).convert('L').convert('RGB')
        label = self.labels[item]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def train_transform(im_h, im_w):
    transform_list = [transforms.RandomRotation(45),
                      transforms.RandomHorizontalFlip(),
                      transforms.Resize((im_h, im_w)),
                      # transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.3),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(transform_list)


def val_transform(im_h, im_w):
    return transforms.Compose([
        transforms.Resize((im_h, im_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_dataset(data_dir=DATA_DIR):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(data_dir)

    image_size = 224
    transform_train = train_transform(image_size, image_size)
    transform_val = val_transform(image_size, image_size)

    train_ds = CustomImageDataset(X_train, y_train, transform=transform_train)
    val_ds = CustomImageDataset(X_val, y_val, transform=transform_val)
    test_ds = CustomImageDataset(X_test, y_test, transform=transform_val)

    return train_ds, val_ds, test_ds


def get_dataset_train_test(data_dir=DATA_DIR):
    (X_train, y_train), (X_test, y_test) = split_train_test(data_dir)

    image_size = 224
    transform_train = train_transform(image_size, image_size)
    transform_val = val_transform(image_size, image_size)

    train_ds = CustomImageDataset(X_train, y_train, transform=transform_train)
    test_ds = CustomImageDataset(X_test, y_test, transform=transform_val)

    return train_ds, test_ds


def get_dataset_external_test(data_dir=EXTERNAL_DATA_DIR):
    X_test, y_test = get_data(data_dir)

    image_size = 224
    transform_val = val_transform(image_size, image_size)

    test_ds = CustomImageDataset(X_test, y_test, transform=transform_val)

    return test_ds


if __name__ == '__main__':
    train_ds, val_ds, test_ds = get_dataset()
    external_test_ds = get_dataset_external_test()
    info = {
        'all': len(train_ds) + len(val_ds) + len(test_ds),
        'train': len(train_ds),
        'val': len(val_ds),
        'test': len(test_ds),
        'external test': len(external_test_ds)
    }
    print(info)

