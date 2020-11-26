#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:11:05 2020

@author: simon
"""
#%%Load library
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import glob
import os

#%% Run isolation forest
fluxnet_site = glob.glob('/home/simon/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/input_data/fluxnet/*')
for site_ in fluxnet_site: 
    # Load data
    site= xr.open_dataset(site_)
    site['NEE_orig'] = site['NEE'].copy()
    NEE_valid = site.NEE.where(site.NEE_QC>=0.85)
    mask = np.isnan(NEE_valid.values)
    NEE_valid = NEE_valid[~mask]
    
    # Fit model
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(.05),
                          max_features=1.0)
    model.fit(NEE_valid.values.reshape(-1, 1))
    scores= model.decision_function(NEE_valid.values.reshape(-1, 1))
    anomaly = model.predict(NEE_valid.values.reshape(-1, 1))
    NEE_valid[anomaly == -1] = np.nan
    site['NEE'][~mask] = NEE_valid
    site.to_netcdf('/home/simon/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/input_data/fluxnet_qc/' + os.path.basename(site_), mode ='w')


test = xr.open_mfdataset('/home/simon/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/input_data/fluxnet/*.nc')

test = test.NEE.values.reshape(-1)
test = (test - -0.66842107) /  2.4312782
