#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:19:52 2020

@author: elliotgross
"""


import pandas as pd
import numpy as np
import regex as re


from Data_Transformer_Pipeline import Data_Transformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score


def get_prepared_data(df, transformer):
    X = df.drop(['Half Life (Seconds)'], axis=1)
    y = df['Half Life (Seconds)']

    prepared_X, prepared_y = transformer.transform(X, y)

    return prepared_X, prepared_y

def normalize_data(df):
    norm = MinMaxScaler().fit(df)
    df_norm = pd.DataFrame(norm.transform(df), columns=df.columns)
    return df_norm

def get_final_data(df, transformer):
    prepared_X, prepared_y = get_prepared_data(df, transformer)
    normalized_X = normalize_data(prepared_X)

    return normalized_X, prepared_y


def get_data_transformer(target_vector='Seconds', m_threshold=2, s_threshold=3600, X_features='all'):
    data_transformer = Data_Transformer(X_features=X_features,
                                        target_vector=target_vector, prediction_type='Binary',
                                        magnitude_threshold=m_threshold,
                                        seconds_threshold=s_threshold,
                                        X_imputer_strat='drop', X_fill_value='None',
                                        y_imputer_strat='drop', y_fill_value='None')

    return data_transformer


def train_model(final_X, final_y, model):
    model.fit(final_X, final_y)
    return model

def get_custom_model(data, default_model, target_vector='Seconds', s_threshold=60, m_threshold=2, X_features='all'):
    transformer = get_data_transformer(target_vector=target_vector, s_threshold=s_threshold, X_features=X_features)
    x, y = get_final_data(data, transformer)

    custom_model = train_model(x, y, default_model)

    return custom_model
