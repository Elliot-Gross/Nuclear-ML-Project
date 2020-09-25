#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:58:38 2020

@author: elliotgross
"""

import Visualisations_Helper_Script as hs

web_data_df, github_data_df = hs.load_data()
data = hs.clean_data(web_data_df, github_data_df)

default_model = hs.get_default_xgb_model(data)

one_ms_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=.001)
one_second_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=1)
one_minute_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=60)
one_hour_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=3600)
one_day_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=86400)
one_year_model = hs.get_custom_model(data, default_model, target_vector='Seconds', s_threshold=3.154e+7)
