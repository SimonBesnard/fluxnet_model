#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:30:44 2020

@author: simon
"""
#%% Load lbrary
import sys
import warnings
import ray
if not sys.warnoptions:
    warnings.simplefilter("ignore")
        
#%% Main function to run the model r
class run_experiment:    
    def __init__(self, config, exp_name):
        from utils import hp_tune
        from utils import model_tune
        from utils import inference
        self.nfolds = config['nfolds']
        self.valid_frac = config['valid_frac'] 
        self.tune_hp = hp_tune.tune_run
        self.tune_model = model_tune.tune_run        
        self.inference = inference.tune_run        
        self.config = config
        #import pdb; pdb.set_trace()
        
    #%% Define global RF function
    def model_experiment(self, fold):
        
        #%% Append fold to config of the experiments
        self.config['fold'] = fold
        
        #%% run hyperparameters opimization
        ray.init(include_webui=False, memory=2000 * 1024 * 1024,
                object_store_memory=2000 * 1024 * 1024,
                driver_object_store_memory=100 * 1024 * 1024,
                temp_dir = '/Net/Groups/BGI/scratch/sbesnard/tmp')
        if self.config['training_procedure'] == 'global':
            self.tune_hp(self)
        
        if self.config['training_procedure'] == 'cross_validation':
            #%% run model tuning
            self.tune_model(self)
        
             #%% run inference
            self.inference(self)
        ray.shutdown()
        
        
            
