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
    cols_to_keep = ['Z','N','Mass','Half Life','M']
    data_merger = DataMerger(cols_to_keep)
    df = data_merger.transform(web_data_df, github_data_df)[1:].reset_index(drop=True)
    return df

def get_prepared_data(df):
    X = df.drop(['Half Life','M'], axis=1)
    y = df['Half Life']
    
    X_features = ['Z','N','Mass','N/P','Adj. N/P','P/N','Adj. P/N','N/Mass','P/Mass','Adj. N/Mass',
              'Adj. P/Mass', 'Adj. N/Mass - Z', 'Adj. P/Mass - Z', 'Z-N']

    data_transformer = Data_Transformer(X_features='all',
                                        target_vector='Seconds', prediction_type='Binary',
                                        magnitude_threshold=2, seconds_threshold=3600,
                                        X_imputer_strat='drop', X_fill_value='None',
                                        y_imputer_strat='drop', y_fill_value='None')

    prepared_X, prepared_y = data_transformer.transform(X, y)
    
    return prepared_X, prepared_y

def normalize_data(df):
    norm = MinMaxScaler().fit(df)
    df_norm = pd.DataFrame(norm.transform(df), columns=df.columns)
    return df_norm
    
def get_split_data(X_norm, prepared_y):
    return train_test_split(X_norm,prepared_y,test_size=0.2,random_state=42)  
    

def get_xgb_model(df):
    
    prepared_X, prepared_y = get_prepared_data(df)
    X_norm = normalize_data(prepared_X)
    X_train, X_test, y_train, y_test = get_split_data(X_norm, prepared_y)
     
    
    xgb_model = XGBClassifier(verbosity=0)
    
    parameters = {'nthread':[1],
                  'objective':['binary:logistic'],
                  'learning_rate': [0.01],
                  'max_depth': [8],
                  'min_child_weight': [3],
                  'silent': [1],
                  'subsample': [0.8],
                  'colsample_bytree': [0.5],
                  'n_estimators': [1000],
                  'missing':[-999],
                  'seed': [1337]}
    
    xgb_model = GridSearchCV(estimator=xgb_model, param_grid=parameters, n_jobs=1, 
                       cv=StratifiedKFold(n_splits=5, shuffle=True), 
                       scoring='f1',
                       refit=True)
    

    xgb_model.fit(X_train, y_train)
    
    
    return xgb_model

def quick_transformer_generator(df,  target, m_threshold=2, s_threshold=10):
    X = df.drop(['Half Life','M'], axis=1)
    y = df['Half Life']
    
    data_transformer = Data_Transformer(X_features='all',
                                        target_vector='Seconds', prediction_type='Binary',
                                        magnitude_threshold=m_threshold, 
                                        seconds_threshold=s_threshold,
                                        X_imputer_strat='drop', X_fill_value='None',
                                        y_imputer_strat='drop', y_fill_value='None')

    

    
    return data_transformer


def quick_model_generator(df, model, target, m_threshold=2, s_threshold=10):
    X = df.drop(['Half Life','M'], axis=1)
    y = df['Half Life']
    
    data_transformer = quick_transformer_generator(df,  target,
                                                   m_threshold=m_threshold,
                                                   s_threshold=s_threshold)

    prepared_X, prepared_y = data_transformer.transform(X, y)
    X_norm = normalize_data(prepared_X)
    
    X_train, X_test, y_train, y_test = get_split_data(X_norm, prepared_y)
    
    
    model.fit(X_train, y_train)
    
    return model

