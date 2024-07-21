# -*- coding: utf-8 -*-
import os.path
from sklearn.model_selection import train_test_split

from LymphNode.config import DATA_DIR, EXTERNAL_DATA_DIR, classes, random_seed


def get_data(data_dir):
    class2label = {cls: i for i, cls in enumerate(classes)}
    images = []
    labels = []
    for group in classes:
        group_dir = os.path.join(data_dir, group)
        group_images = [os.path.join(group_dir, file) for file in os.listdir(group_dir)]
        images.extend(group_images)
        group_labels = [class2label[group]] * len(group_images)
        labels.extend(group_labels)

    img_lbl = [(img, lbl) for img, lbl in zip(images, labels)]
    img_lbl.sort(key=lambda ele: ele[0], reverse=True)
    images = [img for img, lbl in img_lbl]
    labels = [lbl for img, lbl in img_lbl]

    return images, labels


def split_train_val_test(data_dir, ratio_test=0.2, ratio_val=0.25):
    """
    get train from 60% DATA_DIR
    get val from 20% DATA_DIR, if ratio_val is 0, then val is empty
    get internal test from 20% DATA_DIR

    :param data_dir:
    :param ratio_test: test size with ratio_test * len(data), 20% dataset by default
    :param ratio_val: val size with ratio_val * (1-ratio_test) * len(data), 25%*80% = 20% dataset by default
    :return:
    """
    images, labels = get_data(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=ratio_test, random_state=random_seed,
                                                        shuffle=True, stratify=labels)

    X_val = []
    y_val = []
    if ratio_val > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ratio_val, random_state=random_seed,
                                                          shuffle=True, stratify=y_train)  # 0.25 x 0.8 = 0.2

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    return data



def split_train_val_test_ext(data_dir, external_data_dir, ratio_val=0.25):
    """
    get train from 60% DATA_DIR and train 50% from EXTERNAL_DATA_DIR
    get val from 20% DATA_DIR and 50% EXTERNAL_DATA_DIR, if ratio_val is 0, then val is empty
    get internal test from 20% DATA_DIR
    get external test from 50% EXTERNAL_DATA_DIR

    :param data_dir:
    :param ratio_val:
    :return:
    """
    data = split_train_val_test(data_dir, ratio_val=ratio_val)

    X_test_ext, y_test_ext = get_data(external_data_dir)
    X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_test_ext, y_test_ext,
                                                                        test_size=0.5, random_state=random_seed,
                                                                        shuffle=True, stratify=y_test_ext)

    if ratio_val > 0:
        X_train_ext, X_val_ext, y_train_ext, y_val_ext = train_test_split(X_train_ext, y_train_ext, test_size=0.5,
                                                                          random_state=random_seed,
                                                                          shuffle=True, stratify=y_train_ext)
        data["X_val"].extend(X_val_ext)
        data["y_val"].extend(y_val_ext)

    data["X_train"].extend(X_train_ext)
    data["y_train"].extend(y_train_ext)

    data["X_test_int"] = data["X_test"]
    data["y_test_int"] = data["y_test"]
    data["X_test_ext"] = X_test_ext
    data["y_test_ext"] = y_test_ext

    del data["X_test"]
    del data["y_test"]

    return data




if __name__ == '__main__':
    data = split_train_val_test(DATA_DIR)
    data1 = split_train_val_test(DATA_DIR)
    assert set(data["X_train"]) == set(data1["X_train"])
    assert set(data["X_val"]) == set(data1["X_val"])
    assert set(data["X_test"]) == set(data1["X_test"])
    empty = set(data["X_train"]) & set(data["X_val"]) & set(data["X_test"])
    assert not empty

    data = split_train_val_test(DATA_DIR, ratio_val=0)
    assert not data["X_val"] and not data["y_val"]

    images, labels = get_data(data_dir=EXTERNAL_DATA_DIR)
    print(len(images))
    print(images)

