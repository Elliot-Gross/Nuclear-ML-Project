#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:46:30 2020

@author: elliotgross
"""


import pandas as pd
import numpy as np
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class DataMerger(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_keep='all'):
        self.cols_to_keep = cols_to_keep
    
    def fit(self, web_data_df, github_data_df):
        return self
    
    def transform(self, web_data_df, github_data_df):
        
        #Cleaning Web Data Df
        cleaned_web_data_df = self.clean_web_df(web_data_df)
        
        #Setting Common Indices
        cleaned_web_data_df = cleaned_web_data_df.set_index(['Z', 'N'])
        github_data_df = github_data_df.set_index(['Z','N'])
        
        #Merging On Indices
        master_df = cleaned_web_data_df.join(github_data_df, how='outer').reset_index()
        
        if self.cols_to_keep == 'all':
            return master_df
        return master_df[self.cols_to_keep]
    
    #Helper Methods
    def get_percent_uncertainty(self, row):
        if "keV".lower() in str(row["mass"]).lower():
            row["Mass Uncertainty (%)"] = np.nan
        else:
            row["Mass Uncertainty (%)"] = round(float(str(row['mass']).split()[-1]) / row["Mass"] * 100, 4)


        return row
    
    #Cleaner Method
    def clean_web_df(self, web_data_df):
        #Mass Value Column
        web_data_df["Mass"] = web_data_df['mass'].str.split(' ').str[0].astype(float)
        
        #Half Life Column
        web_data_df.loc[(web_data_df['half-life'] == 'stable'), 'Half Life'] = -1
        web_data_df.loc[(web_data_df['half-life'] == 'unknown'), 'Half Life'] = np.nan
        web_data_df.loc[(web_data_df['half-life'] != 'stable') & 
                (web_data_df['half-life'] != 'unknown'), 
                'Half Life'] = web_data_df['half-life']
        
        
        #Selecting Relevant Columns
        columns_to_keep = ["Z", "Nuclide", "Mass", "Half Life"]
        web_data_df = web_data_df[columns_to_keep]
        
        #Neutron Column
        web_data_df['N'] = np.floor(web_data_df['Mass']) - web_data_df['Z']
        
        return web_data_df
    
    
### TEST ###
        
#web_data_df = pd.read_csv("Data/Loaded_Data/Web_Data.csv")
#github_data_df = pd.read_csv("Data/Loaded_Data/Github_Data.csv")
        
#data_merger = DataMerger(['N'])
#df = data_merger.transform(web_data_df, github_data_df)


#print(df)
   
    