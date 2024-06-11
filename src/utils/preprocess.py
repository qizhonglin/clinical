#!/usr/bin/env python3


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pandas as pd


class OutlierImputer(object):
    def fit(self, df_train, colname):
        self.med = df_train[colname].median()
        self.std = df_train[colname].std()
        self.colname = colname

    def transform(self, df_test, times=2):
        colname = self.colname
        med = self.med
        std = self.std

        outliers = df_test[colname] < (med - times * std)
        df_test.loc[outliers, colname] = med

        outliers = df_test[colname] > (med + times * std)
        df_test.loc[outliers, colname] = med

        return df_test


def preprocess_numeric(df_train, df_test=None):
    """
    :param df_train:
    :param df_test:
    """
    columns = df_train.columns
    index_train = df_train.index
    index_test = df_test.index

    # handle outlier
    for col in columns:
        outlier = OutlierImputer()
        outlier.fit(df_train, col)
        df_train = outlier.transform(df_train)

        if df_test is not None:
            df_test = outlier.transform(df_test)

    X_train = df_train.values
    X_test = df_test.values

    # fill na
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)

    if X_test is not None:
        X_test = imputer.transform(X_test)

    # normalize features
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    if X_test is not None:
        X_test = scaler.transform(X_test)

    df_train = pd.DataFrame(X_train, columns=columns, index=index_train)
    df_test = pd.DataFrame(X_test, columns=columns, index=index_test)

    return df_train, df_test


def preprocess_category(df_train, df_test=None):
    columns = df_train.columns

    for col in columns:
        df_train[col] = df_train[col].astype('category')
        categories = df_train[col].cat.categories
        df_test[col] = pd.Categorical(df_test[col], categories)

    df_train_dummies = pd.get_dummies(df_train)
    df_test_dummies = pd.get_dummies(df_test)

    return df_train_dummies, df_test_dummies

