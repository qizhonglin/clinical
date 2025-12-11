# -*- coding: utf-8 -*-
import os.path
import json 
import cv2
import numpy as np 
from tqdm import tqdm
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from LymphNode.config import DATA_ROOT, DATA_DIR, EXTERNAL_DATA_DIR, classes, random_seed


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

def prepare_data(json_file, data_dir, target_width=640):
    with open(json_file, "r") as f:
        data = json.load(f)
        
    cls2labels = {
        0: 'benign',
        1: 'malignant'
    }

    for split in ["train", "val", "test"]:
        X_split = data[f"X_{split}"]
        y_split = data[f"y_{split}"]

        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for cls in cls2labels.values():
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

        for img_path, cls in tqdm(zip(X_split, y_split)):
            label = cls2labels[cls]
            dst_path = os.path.join(split_dir, label, os.path.basename(img_path))
            
            image = Image.open(img_path)
            image = ImageOps.equalize(image)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            new_size = (target_width, target_width)
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_path, resized)

def main():
    data = split_train_val_test(DATA_DIR)
    data1 = split_train_val_test(DATA_DIR)
    assert set(data["X_train"]) == set(data1["X_train"])
    assert set(data["X_val"]) == set(data1["X_val"])
    assert set(data["X_test"]) == set(data1["X_test"])
    empty = set(data["X_train"]) & set(data["X_val"]) & set(data["X_test"])
    assert not empty
    
    json_file = os.path.join(DATA_DIR, "data_split.json")
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
        
    prepare_data(json_file, data_dir=os.path.join(DATA_ROOT, "experiments"))

    data = split_train_val_test(DATA_DIR, ratio_val=0)
    assert not data["X_val"] and not data["y_val"]

    images, labels = get_data(data_dir=EXTERNAL_DATA_DIR)
    print(len(images))
    print(images)
    
    
        

if __name__ == '__main__':
    main()

