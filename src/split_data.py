# -*- coding: utf-8 -*-
import os.path
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, EXTERNAL_DATA_DIR, classes, random_seed


def get_data(data_dir=EXTERNAL_DATA_DIR):
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


def split_train_test(data_dir):
    images, labels = get_data(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=random_seed,
                                                        shuffle=True, stratify=labels)

    return (X_train, y_train), (X_test, y_test)

def split_train_val_test(data_dir=DATA_DIR):
    (X_train, y_train), (X_test, y_test) = split_train_test(data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_seed,
                                                      shuffle=True, stratify=y_train)  # 0.25 x 0.8 = 0.2

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



if __name__ == '__main__':
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test()
    (X_train1, y_train1), (X_val1, y_val1), (X_test1, y_test1) = split_train_val_test()
    assert set(X_train) == set(X_train1)
    assert set(X_val) == set(X_val1)
    assert set(X_test) == set(X_test1)
    empty = set(X_train).intersection(X_val).intersection(X_test)
    assert not empty

