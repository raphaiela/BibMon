#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:38:16 2020

@author: afranio
"""

import bibmon
import pandas as pd
from bibmon._bibmon_tools import detect_outliers_iqr

def test_complete_analysis():
    
    # load data
    data = bibmon.load_real_data()
    data = data.apply(pd.to_numeric, errors='coerce')
    
    # preprocessing pipeline
    
    preproc_tr = ['remove_empty_variables',
                  'ffill_nan',
                  'remove_frozen_variables',
                  'normalize']
    
    preproc_ts = ['ffill_nan','normalize']
    
    # define training set
        
    (X_train, X_validation, 
     X_test, Y_train, 
     Y_validation, Y_test) = bibmon.train_val_test_split(data, 
                                            start_train = '2017-12-24T12:00', 
                                            end_train = '2018-01-01T00:00', 
                                           end_validation = '2018-01-02T00:00', 
                                            end_test = '2018-01-04T00:00',
                                            tags_Y = 'tag100')
                                                         
    # define the model
                                                         
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    
    model = bibmon.sklearnRegressor(reg)                                                          

    # define regression metrics
                                                         
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    
    mtr = [r2_score, mean_absolute_error]
                           
    # complete analysis!
                              
    bibmon.complete_analysis(model, X_train, X_validation, X_test, 
                            Y_train, Y_validation, Y_test,
                            f_pp_train = preproc_tr,
                            f_pp_test = preproc_ts,
                            metrics = mtr, 
                            count_window_size = 3, count_limit = 2,
                            fault_start = '2018-01-02 06:00:00',
                            fault_end = '2018-01-02 09:00:00') 
    
    model.plot_importances()                                                                             

def test_detect_outliers_iqr():
    # lower_bound=4 e upper_bound=10
    base_data = [6.25] * 100 + [7.75] * 100
    test_data = base_data + [2, 12, 6, 1]
    df = pd.DataFrame({'col1': test_data})
    df_outliers = detect_outliers_iqr(df, cols=['col1'])
    
    # Índices dos casos de teste (últimas 4 linhas)
    ct1_index = len(base_data)     # 200
    ct2_index = len(base_data) + 1 # 201
    ct3_index = len(base_data) + 2 # 202
    ct4_index = len(base_data) + 3 # 203
    
    # CT1: Valor 2 (abaixo de 4) → outlier (1)
    assert df_outliers.loc[ct1_index, 'col1'] == 1, "CT1 falhou"
    
    # CT2: Valor 12 (acima de 10) → outlier (1)
    assert df_outliers.loc[ct2_index, 'col1'] == 1, "CT2 falhou"
    
    # CT3: Valor 6 (entre 4 e 10) → não outlier (0)
    assert df_outliers.loc[ct3_index, 'col1'] == 0, "CT3 falhou"
    
    # CT4: Valor 1 (abaixo de 4) → outlier (1)
    assert df_outliers.loc[ct4_index, 'col1'] == 1, "CT4 falhou"
