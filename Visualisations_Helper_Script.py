#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:19:52 2020

@author: elliotgross
"""


import pandas as pd
import numpy as np

from Data_Merger_Pipeline import DataMerger
from Data_Transformer_Pipeline import Data_Transformer

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def load_data():
    web_data_df = pd.read_csv("Data/Loaded_Data/Web_Data.csv")
    keV_index = web_data_df['mass'][web_data_df['mass'].str.contains('keV') == True].index
    web_data_df = web_data_df.drop(keV_index)
    web_data_df.reset_index(drop=True)

    github_data_df = pd.read_csv("Data/Loaded_Data/Github_Data.csv")

    return web_data_df, github_data_df

def clean_data(web_data_df, github_data_df):
    cols_to_keep = ['Z','N','Mass','Half Life']
    data_merger = DataMerger(cols_to_keep)
    df = data_merger.transform(web_data_df, github_data_df)[1:].reset_index(drop=True)
    return df



def get_prepared_data(df, transformer):
    X = df.drop(['Half Life'], axis=1)
    y = df['Half Life']

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



def get_default_xgb_model(df):

    final_X, final_y = get_final_data(df, get_data_transformer())

    parameters = {'nthread':1,
                  'objective':'binary:logistic',
                  'learning_rate': 0.01,
                  'max_depth': 8,
                  'min_child_weight': 3,
                  'silent': 1,
                  'subsample': 0.8,
                  'colsample_bytree': 0.5,
                  'n_estimators': 1000,
                  'missing':-999,
                  'seed': 1337}


    xgb_model = XGBClassifier(verbosity=0)
    xgb_model.set_params(**parameters)
    xgb_model.fit(final_X, final_y)

    return xgb_model

def get_data_transformer(target_vector='Seconds', m_threshold=2, s_threshold=60):
    data_transformer = Data_Transformer(X_features='all',
                                        target_vector=target_vector, prediction_type='Binary',
                                        magnitude_threshold=m_threshold,
                                        seconds_threshold=s_threshold,
                                        X_imputer_strat='drop', X_fill_value='None',
                                        y_imputer_strat='drop', y_fill_value='None')

    return data_transformer


def train_model(final_X, final_y, model):
    model.fit(final_X, final_y)
    return model

def get_custom_model(data, default_model, target_vector='Seconds', s_threshold=60, m_threshold=2):
    transformer = get_data_transformer(target_vector=target_vector, s_threshold=s_threshold)
    x, y = get_final_data(data, transformer)

    custom_model = train_model(x, y, default_model)

    return custom_model
