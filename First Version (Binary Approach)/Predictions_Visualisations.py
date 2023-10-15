#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:16:10 2020

@author: elliotgross
"""

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from decimal import Decimal

import Visualisations_Helper_Script as hs
import Predefined_Models as pm

df = pm.get_data()
model = pm.get_default_model()

def generate_test_data(seconds_threshold, min_proton, max_proton, min_neutron, max_neutron, X_features='all'):
    z, n = [],[]
    for i in range(min_proton,max_proton+1):
        for j in range(min_neutron,max_neutron+1):
            z.append(i)
            n.append(j)

    test_df = pd.DataFrame({'Z':z, 'N':n})
    test_df['Mass'] = test_df['N'] + test_df['Z']
    test_df['Mass'] = [Decimal(test_df.loc[i,'Z']*0.00054386734)+test_df.loc[i,'Mass'] for i in range(test_df.shape[0])]
    #fix mass

    data_transformer = hs.get_data_transformer(target_vector = 'Seconds',
                                               s_threshold=seconds_threshold, X_features=X_features)

    X_test = data_transformer.transform(test_df)
    final_X = hs.normalize_data(X_test)

    return final_X, X_test

def plot_model(df, X_test, predictions, seconds_threshold,
               min_proton, max_proton, min_neutron, max_neutron,
               alpha_predictions, show_known_values, alpha_known_values):

    fig = plt.figure()

    if show_known_values:
        data_transformer = hs.get_data_transformer(target_vector='Seconds',
                                                    s_threshold=seconds_threshold)
        X = df.drop(['Half Life (Seconds)'], axis=1)
        y = df['Half Life (Seconds)']
        X,y = data_transformer.transform(X,y)
        X = X[(X['Z']>=min_proton) & (X['Z']<=max_proton)]
        X = X[(X['N']>=min_neutron) & (X['N']<=max_neutron)]
        y = y[X.index]


        true_stable = X[y==1]
        plt.scatter(true_stable['N'],true_stable['Z'],
                    color='blue', alpha=alpha_known_values, s=12, label='True')



    predicted_stable = X_test[predictions==1]
    plt.scatter(predicted_stable['N'], predicted_stable['Z'],
                color='green', alpha=alpha_predictions, s=12, label='Predicted')


    plt.xlim(min_neutron-2, max_neutron+5)
    plt.ylim(min_proton-2, max_proton+5)

    plt.xlabel('N')
    plt.ylabel('P')

    plt.legend()
    plt.show()
    st.pyplot()



#Customizations
st.sidebar.markdown('**Data Options:**')
min_proton = int(st.sidebar.text_input("Min Protons: ", '1'))
max_proton = int(st.sidebar.text_input("Max Protons: ", '118'))
min_neutron = int(st.sidebar.text_input("Min Neutrons: ", '1'))
max_neutron = int(st.sidebar.text_input("Max Neutrons: ", '117'))


#features_to_keep = st.sidebar.multiselect("Features to Use:", pm.all_features, default=['Z','N','Mass'])
show_raw_data = st.sidebar.checkbox("Show Raw Data", True)

seconds_threshold = float(st.text_input("Half-Life Threshold (Seconds): ", '3600'))

st.sidebar.markdown('**Graph Options:**')
alpha_predictions = st.sidebar.slider('Alpha - Predictions:', 0.0,1.0,.37)

show_known_values = st.sidebar.checkbox("Show Known Data", True)
alpha_known_values = st.sidebar.slider('Alpha - Known Data:', 0.0,1.0,.5)

#Predictions
final_X, X_test = generate_test_data(seconds_threshold, min_proton, max_proton,
                                     min_neutron, max_neutron, X_features='all')

model = hs.get_custom_model(df, model, 'Seconds',
                            s_threshold=seconds_threshold, X_features='all')

predictions_test = model.predict(final_X)

plot_model(df, X_test, predictions_test, seconds_threshold, min_proton,
           max_proton, min_neutron, max_neutron, alpha_predictions,
           show_known_values, alpha_known_values)

if show_raw_data:
    st.write("Raw Data: ")
    st.write(X_test)
