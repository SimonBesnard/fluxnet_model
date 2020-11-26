#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:39:15 2020

@author: simon
"""
#%% Load library
import numpy as np
    
#%% Function to split train and test sets
def get_fold(data, fold):
    test_sites  = fold
    train_sites = data.site[~np.in1d(data.site, fold)]
    test_dataset  = data.sel(site=test_sites)
    train_dataset = data.sel(site=train_sites)
    return (train_dataset, test_dataset)
        