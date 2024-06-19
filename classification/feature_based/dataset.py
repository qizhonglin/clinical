# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from pprint import pprint

from classification.config import DATA_ROOT, DATA_DIR, classes, EXTERNAL_DATA_DIR
from classification.split_data import split_train_test, get_data
from classification.utils.preprocess import preprocess_numeric, preprocess_category

# below imports are used to print out pretty pandas dataframes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
pd.set_option('display.width', 1024)  # console defaults to 80 characters

EXCEL_NAME = "华西本部淋巴结数据.xlsx"
EXTERNAL_EXCEL_NAME = "天府院区甲状腺癌淋巴结统计用表.xlsx"
ID_NAME = "检查号"
OUTCOME_NAME = "淋巴结最终病理（1为恶性，0为良性）"

NUMERICAL_NAME = [
    '长径',
    '短径',
    '长短径比',
]
BINARY_NAME = [
    '性别（男0，女1）',
    '淋巴门（有0，无1）',
    '皮质回声（均匀0，不均匀1）',
]
CATEGORIES_NAME = [
    '部位(1区、2区、3区、4区、5区、6区）',
    '血流模式（0为无，1为门样，2为周边，3为混合）',
    '形状（椭圆形0，圆形1，不规则2）',
    '边缘（锐利0，模糊1，分界不清2，融合3）',
    '液化（无0，可疑1，有2）',
    '钙化（无0，粗钙化1，可疑微钙化2，微钙化3）',
    '回声（低回声0，无回声1，等回声2，高回声3，）'
]
FEATURES_NAME = NUMERICAL_NAME + BINARY_NAME + CATEGORIES_NAME
COLUMNS = [ID_NAME, '初步超声诊断', OUTCOME_NAME] + FEATURES_NAME


def load_excel(table_file):
    """

    :param table_file:
    :return: dataframe with column [ID, outcome, features]
    """
    df = pd.read_excel(table_file)
    # print(df.head())
    # print(df.columns)

    df[ID_NAME] = df[ID_NAME].astype("str")

    return df


def compare_images_excel(data_dir, tabelfile):
    # IDs and labels from images
    images, labels = get_data(data_dir)
    IDs = [os.path.splitext(os.path.basename(file))[0] for file in images]
    ID_labels = {id: y for id, y in zip(IDs, labels)}

    # IDs and labels from excel
    df = load_excel(tabelfile)
    ID_labels_excel = {id: outcome for id, outcome in zip(df[ID_NAME], df[OUTCOME_NAME])}

    # IDs from both image and excel
    ID_both = set(IDs).intersection(set(df[ID_NAME].values))

    # check different outcome
    label_clc = {label: cls for label, cls in enumerate(classes)}
    different_outcome = []
    for i, id in enumerate(ID_both):
        if ID_labels[id] != ID_labels_excel[id]:
            different_outcome.append((id, label_clc(ID_labels[id]), label_clc(ID_labels_excel[id])))

    ID_image_not_excel = set(IDs) - set(df[ID_NAME].values)
    ID_excel_not_image = set(df[ID_NAME].values) - set(IDs)
    info = {
        'data_dir': data_dir,
        'table file': tabelfile,
        'ID in image but not in excel NUM': len(ID_image_not_excel),
        'ID in image but not in excel': ID_image_not_excel,
        'ID in excel but not in image NUM': len(ID_excel_not_image),
        'ID in excel but not in image': ID_excel_not_image,
        'diagnosis is different between image and excel': different_outcome
    }
    pprint(info)


def _get_train_test_from_excel(data_dir, tabelfile):
    (X_train, y_train), (X_test, y_test) = split_train_test(data_dir)
    ID_train = [os.path.splitext(os.path.basename(file))[0] for file in X_train]
    ID_test = [os.path.splitext(os.path.basename(file))[0] for file in X_test]

    df = load_excel(tabelfile)
    IDs = df[ID_NAME].values.tolist()
    train_image_excel = list(set(ID_train).intersection(set(IDs)))
    test_image_excel = list(set(ID_test).intersection(set(IDs)))

    mask_row = df.loc[:, ID_NAME].isin(train_image_excel)
    df_train = df.loc[mask_row, FEATURES_NAME]
    diagnosis_train = df.loc[mask_row, OUTCOME_NAME]

    mask_row = df.loc[:, ID_NAME].isin(test_image_excel)
    df_test = df.loc[mask_row, FEATURES_NAME]
    diagnosis_test = df.loc[mask_row, OUTCOME_NAME]

    return (df_train, diagnosis_train), (df_test, diagnosis_test)


def get_train_test_from_excel(data_dir=DATA_DIR,
                              tablefile=os.path.join(DATA_ROOT, EXCEL_NAME)):
    # split to train and test dataset
    (df_train, diagnosis_train), (df_test, diagnosis_test) = _get_train_test_from_excel(data_dir, tablefile)

    # preprocess
    df_train_category, df_test_category = preprocess_category(df_train[CATEGORIES_NAME], df_test[CATEGORIES_NAME])
    df_train = pd.concat([df_train[NUMERICAL_NAME], df_train[BINARY_NAME], df_train_category], axis=1)
    df_test = pd.concat([df_test[NUMERICAL_NAME], df_test[BINARY_NAME], df_test_category], axis=1)

    df_train_num, df_test_num = preprocess_numeric(df_train[NUMERICAL_NAME], df_test[NUMERICAL_NAME])
    df_train = pd.concat([df_train_num, df_train[BINARY_NAME], df_train_category], axis=1)
    df_test = pd.concat([df_test_num, df_test[BINARY_NAME], df_test_category], axis=1)

    return (df_train, diagnosis_train), (df_test, diagnosis_test)


def get_external_test_from_excel(data_dir=DATA_DIR,
                                 trainfile=os.path.join(DATA_ROOT, EXCEL_NAME),
                                 external_data_dir=EXTERNAL_DATA_DIR,
                                 external_testfile=os.path.join(DATA_ROOT, EXTERNAL_EXCEL_NAME)):
    # get external test dataset
    X_test, y_test = get_data(data_dir=external_data_dir)
    ID_test = [os.path.splitext(os.path.basename(file))[0] for file in X_test]

    df = load_excel(table_file=external_testfile)
    IDs = df[ID_NAME].values.tolist()
    test_image_excel = list(set(ID_test).intersection(set(IDs)))

    mask_row = df.loc[:, ID_NAME].isin(test_image_excel)
    df_test = df.loc[mask_row, FEATURES_NAME]
    diagnosis_test = df.loc[mask_row, OUTCOME_NAME]

    # preprocess
    (df_train, diagnosis_train), _ = _get_train_test_from_excel(data_dir, trainfile)
    df_train_category, df_test_category = preprocess_category(df_train[CATEGORIES_NAME], df_test[CATEGORIES_NAME])
    df_test = pd.concat([df_test[NUMERICAL_NAME], df_test[BINARY_NAME], df_test_category], axis=1)

    df_train_num, df_test_num = preprocess_numeric(df_train[NUMERICAL_NAME], df_test[NUMERICAL_NAME])
    df_test = pd.concat([df_test_num, df_test[BINARY_NAME], df_test_category], axis=1)

    return df_test, diagnosis_test


if __name__ == '__main__':
    # different IDs between images and excel
    compare_images_excel(data_dir=DATA_DIR,
                         tabelfile=os.path.join(DATA_ROOT, EXCEL_NAME))
    compare_images_excel(data_dir=EXTERNAL_DATA_DIR,
                         tabelfile=os.path.join(DATA_ROOT, EXTERNAL_EXCEL_NAME))

    # get train and test dataset
    (df_train, y_train), (df_test, y_test) = get_train_test_from_excel()
    df_test_external, y_test_external = get_external_test_from_excel()
    info = {
        'X_train': df_train.values.shape,
        'y_train': y_train.values.shape,
        'X_test': df_test.values.shape,
        'y_test': y_test.values.shape,
        'X_test_external': df_test_external.values.shape,
        'y_test_external': y_test_external.values.shape,
        'train columns': df_train.columns,
        'external columns': df_test_external.columns
    }
    print(info)
    print(sorted(df_train.columns.values.tolist()))
    print(sorted(df_test_external.columns.values.tolist()))
    empty = set(df_train.columns).symmetric_difference(set(df_test_external.columns))
    assert not empty
