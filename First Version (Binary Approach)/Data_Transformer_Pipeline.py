#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:40:07 2020

@author: elliotgross
"""


import pandas as pd
import numpy as np
import re

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class Data_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, X_features='all', target_vector='Seconds',
                 prediction_type='Binary',
                 magnitude_threshold=2, seconds_threshold=10,
                 X_imputer_strat='constant', X_fill_value=np.nan,
                 y_imputer_strat='constant', y_fill_value=np.nan):

        self.X_features = X_features
        self.target_vector = target_vector

        self.prediction_type= prediction_type

        self.magnitude_threshold = magnitude_threshold
        self.seconds_threshold = seconds_threshold

        self.X_imputer_strat = X_imputer_strat
        self.X_fill_value = X_fill_value

        self.y_imputer_strat = y_imputer_strat
        self.y_fill_value = y_fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        #Feature Engeneering
        X = self.add_features(X)

        #Feature Selection
        if self.X_features != 'all':
            X = X[self.X_features]

        #Used to rename X data
        X_columns = X.columns

        #Imputing Missing Values/Dropping Missing Values
        Xrows_with_nan = [index for index, row in X.iterrows() if row.isnull().any()]
        if self.X_imputer_strat == 'drop':
            X = X.drop(Xrows_with_nan).reset_index(drop=True)

        else:
            X = SimpleImputer(strategy=self.X_imputer_strat,
                              fill_value=self.X_fill_value).fit_transform(X)

        #Reshaping/Renaming X
        X = pd.DataFrame(X, columns=X_columns)

        #Handles case when only X needs to be transformed; i.e. Predictions;
        if type(y) == type(None):
            return X

        #Drops Y-Values so that if X-NaN values were dropped, the data still
        # matches with the target data
        if self.X_imputer_strat == 'drop':
            y = y.drop(Xrows_with_nan).reset_index(drop=True)


        #Reshaping y
        y = pd.DataFrame(y, columns=["Half Life (Seconds)"]).reset_index(drop=True)

        #Creating Target Vectors
        y = self.add_magnitude_and_value_features(y)
        isStable = self.get_isStable_feature(y, self.magnitude_threshold, self.seconds_threshold)
        y_columns = y.columns

        #Imputing Missing Values/Dropping Missing Values
        Yrows_with_nan = [index for index, row in pd.DataFrame(y).iterrows() if row.isnull().any()]

        if self.y_imputer_strat == 'drop':
            y = y.drop(Yrows_with_nan).reset_index(drop=True)
            X = X.drop(Yrows_with_nan).reset_index(drop=True)
            isStable = isStable.drop(Yrows_with_nan).reset_index(drop=True)

        else:
            y = SimpleImputer(strategy=self.y_imputer_strat,
                              fill_value=self.y_fill_value).fit_transform(y)

        #Reshaping/Renaming y
        y = pd.DataFrame(y, columns=y_columns)

        #Choosing Target Vector
        if self.target_vector == 'Seconds':
            y = y['Half Life (Seconds)']
        elif self.target_vector == 'Magnitude':
            y = y['Magnitude of Value']

        if self.prediction_type == 'Binary':
            y = isStable

        #Returns X, y
        return X, y

    #Helper Methods
    def add_neutron_proton_ratios(self, X):
        X['N/P'] = (X['N']/X['Z']).astype(float)
        X['Adj. N/P'] = abs(X['N/P'] - 1).astype(float)

        X['P/N'] = (X['Z']/X['N']).astype(float)
        X['Adj. P/N'] = abs(X['P/N'] - 1).astype(float)

        return X

    def add_energy_mass_ratios(self, X):
        X['M/P'] = (X['M']/X['Z']).astype(float)
        X['M/N'] = (X['M']/X['N']).astype(float)

        X['M*N/P'] = (X['M']*X['N/P']).astype(float)
        X['M*P/N'] = (X['M']*X['P/N']).astype(float)

        X['Adj. M*N/P'] = (X['M']*X['Adj. N/P']).astype(float)
        X['Adj. M*P/N'] = (X['M']*X['Adj. P/N']).astype(float)

        return X

    def add_mass_ratios(self, X):
        X['N/Mass'] = (X['N']/X['Mass']).astype(float)
        X['P/Mass'] = (X['Z']/X['Mass']).astype(float)

        X['Adj. N/Mass'] = abs(X['N/Mass']-1).astype(float)
        X['Adj. P/Mass'] = abs(X['P/Mass']-1).astype(float)

        X['Adj. N/Mass - Z'] = (X['Adj. N/Mass'] - X['Z']).astype(float)
        X['Adj. P/Mass - Z'] = (X['Adj. P/Mass'] - X['Z']).astype(float)

        X = X.replace(np.inf, np.nan)
        X = X.replace(-np.inf, np.nan)

        return X

    def get_magnitude_and_magnitude_digit_value(self, y):
        magnitudes = []
        values = []
        for i,half_life in enumerate([format(n,'e') for n in y['Half Life (Seconds)']]):
            if half_life == 'nan':
                #Adding Value
                values.append(half_life)
                #Adding Magnitude
                magnitudes.append(half_life)
            elif '-1' in str(half_life):
                values.append(2)
                magnitudes.append(2)
            else:
                #Adding value
                values.append(float(half_life.split('e')[0]))
                #Adding Magnitude
                magnitudes.append(float(half_life.split('e')[1]))

        return magnitudes, values

    #Wrapper Methods
    def add_features(self, X):
        X = self.add_neutron_proton_ratios(X)
        #X = self.add_energy_mass_ratios(X) IDK if M is availiable
        X = self.add_mass_ratios(X)

        #Extra
        #X['-ZM/N'] = X['M']/ (X['N']/-X['Z']) IDK if I can use M
        #X['abs(Z-N)'] = abs(X['Z']-X['N'])

        #Replacing inf/-inf with NaN
        X = X.replace(np.inf, np.nan)
        X = X.replace(-np.inf, np.nan)

        return X

    def add_magnitude_and_value_features(self, y):

        magnitudes, values = self.get_magnitude_and_magnitude_digit_value(y)

        y['Half Life Seconds Value'] = values
        y['Magnitude of Value'] = magnitudes
        y['Magnitude of Value'] = y['Magnitude of Value'].astype(float)

        y.loc[(y['Half Life Seconds Value'] == 'nan'), 'Half Life Seconds Value'] = np.nan
        y.loc[(y['Magnitude of Value'] == 'nan'), 'Magnitude of Value'] = np.nan

        return y

    def get_isStable_feature(self, y, magnitude_threshold, seconds_threshold:float):

        y['isStable'] = 0
        y.loc[y['Half Life (Seconds)'] == -1, 'isStable'] = 1

        seconds_threshold = float(seconds_threshold)

        if self.target_vector == 'Seconds':
            y.loc[y['Half Life (Seconds)'] >= seconds_threshold, 'isStable'] = 1
        elif self.target_vector == 'Magnitude':
            y.loc[y['Magnitude of Value'] >= magnitude_threshold, 'isStable'] = 1

        y.loc[y['Half Life (Seconds)'] != y['Half Life (Seconds)'], 'isStable'] = np.nan

        return y['isStable']



### TEST ###


#df = pd.read_csv("Data/Clean_Data/Cleaned_Master_Data.csv")[1:]

#X = df.drop('Half Life', axis=1)
#y = df['Half Life']

#data_transformer = Data_Transformer(target_vector='Seconds',
                                    #magnitude_threshold=2, seconds_threshold='100',
                                    #prediction_type='Regression')
#X, y = data_transformer.transform(X, y)


#print(y)
