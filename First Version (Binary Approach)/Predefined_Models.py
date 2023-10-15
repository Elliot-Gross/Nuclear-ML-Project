#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:58:38 2020

@author: elliotgross
"""

import Visualisations_Helper_Script as hs
import Clean_Data_Methods as cd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def get_default_xgb_model(df):

    final_X, final_y = hs.get_final_data(df, hs.get_data_transformer())

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

web_data_df, github_data_df = cd.load_data()

data = cd.clean_data(web_data_df, github_data_df)
default_transformer = hs.get_data_transformer()
all_features = list(default_transformer.transform(data.drop(['Half Life (Seconds)'], axis=1)).columns)



default_model = get_default_xgb_model(data)

one_ms_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=.001)
one_second_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=1)
one_minute_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=60)
one_hour_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=3600)
one_day_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=86400)
one_year_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=3.154e+7)

def get_data():
    return data

def get_default_model():
    return default_model
