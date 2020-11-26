#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:20:25 2020

@author: simon
"""
#%% Load library
import sys
from experiments.experiment_config import get_config
from run_experiment import run_experiment
from data.CVfold import create_folds
import xarray as xr
import multiprocessing as mp
import argparse
import numpy as np

#%% Retrieve arguments to parse in gapfilling function
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
       '--exp_name',
       '-exp_name',
       type=str,
       help='name of the experiment',
       default='nee.n_perm',
       required=False)
    
    args = parser.parse_args()

    return args

#%% Retrieve parsed arguments
args = parse_args()

#%% Get config experiment
args.config = get_config(args.exp_name)

#%% Load dataset
#flux_data = xr.open_mfdataset(args.config['data_path'] +'/*.nc')

#%% Create fold
if args.config['training_procedure'] == 'cross_validation':    
    #folds = create_folds(flux_data, args.config['nfolds'])
    folds = np.load(args.config['cv_path'] + '/cv_folds.npy', allow_pickle=True)
elif args.config['training_procedure'] == 'global':   
    folds =  [np.concatenate(np.load(args.config['cv_path'] + '/cv_folds.npy', allow_pickle=True))]

#%% Run model run
print('-------------------------------------------\n'
      'Initialization of the model training procedures\n'
      '-------------------------------------------')
if __name__ == '__main__':
    model_run = run_experiment(**vars(args))
    p = mp.Pool(1, maxtasksperchild=1)
    p.map(model_run.model_experiment, folds)
    p.close()
    p.join()
    print('----------------------------\n'
          'End of model training procedures\n'
        '----------------------------')
