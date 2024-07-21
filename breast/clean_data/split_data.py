# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
import os
from sklearn.model_selection import train_test_split

from breast.config import random_seed, DATA_DIR


def split_train_val_test(data_dir=DATA_DIR, ratio_test=0.2, ratio_val=0.25):
    """
    get train from 60% DATA_DIR
    get val from 20% DATA_DIR, if ratio_val is 0, then val is empty
    get internal test from 20% DATA_DIR

    :param data_dir:
    :param ratio_test: test size with ratio_test * len(data), 20% dataset by default
    :param ratio_val: val size with ratio_val * (1-ratio_test) * len(data), 25%*80% = 20% dataset by default
    :return:
    """
    images = [os.path.join(data_dir, image) for image in os.listdir(data_dir)]
    images_train, images_test = train_test_split(images, test_size=ratio_test,
                                                 random_state=random_seed, shuffle=True)

    images_val = []
    if ratio_val > 0:
        images_train, images_val = train_test_split(images_train, test_size=ratio_val,
                                                          random_state=random_seed, shuffle=True)  # 0.25 x 0.8 = 0.2

    labels_train = [file.replace('imagesTr', 'labelsTr') for file in images_train]
    labels_val = [file.replace('imagesTr', 'labelsTr') for file in images_val]
    labels_test = [file.replace('imagesTr', 'labelsTr') for file in images_test]

    return (images_train, labels_train), (images_val, labels_val), (images_test, labels_test)


if __name__ == '__main__':
    (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) = split_train_val_test()
    (images_train1, labels_train1), (images_val1, labels_val1), (images_test1, labels_test1) = split_train_val_test()
    assert set(images_train) == set(images_train1)
    assert set(images_val) == set(images_val1)
    assert set(images_test) == set(images_test1)
    empty = set(images_train).intersection(images_val).intersection(images_test)
    assert not empty

    (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) = split_train_val_test(ratio_val=0)
    assert not images_val
