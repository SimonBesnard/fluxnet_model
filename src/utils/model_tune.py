#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:20:25 2020

@author: simon
"""
#%% Load library
from models.emulator import Emulator, get_target_path
from utils.summarize_runs import summarize_run
from experiments.experiment_config import get_config
from ray.tune.logger import CSVLogger, JsonLogger
import ray
import argparse
import os
import pickle
import shutil
import numpy as np
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

def load_best_config(store):
    best_config = os.path.join(store, 'summary/best_params.pkl')
    if not os.path.isfile(best_config):
        raise ValueError(
            'Tried to load best model config, file does not exist:\n'
            f'{best_config}\nRun `summarize_results.py` to create '
            'such a file.'
        )
    with open(best_config, 'rb') as f:
        config = pickle.load(f)

    return config


def tune_run(self):
    tune(config_name= self.config['experiment_name'],
         fold = self.config['fold'],
         run_single= False,         
         overwrite=True)

def tune(config_name:str = 'default', fold=None, overwrite:bool= True, run_single:bool = False):

    config = get_config(config_name)
    config.update({'is_tune': False})

    tune_store = get_target_path(config, mode='hptune')
    store = get_target_path(config, mode='modeltune')
    
    if overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all existing runs will be lost!')
    os.makedirs(store)

    best_config = load_best_config(tune_store)
    best_config.update({'hc_config': config})
    best_config['hc_config']['fold'] = fold

    config.update({
        'store': store
    })

    import torch
    ngpu = torch.cuda.device_count()
    ncpu = os.cpu_count()

    #max_concurrent = int(
    #    np.min((
    #        np.floor(ncpu / config['ncpu_per_run']),
    #        np.floor(ngpu / config['ngpu_per_run'])
    #    ))
    #)

    max_concurrent = np.floor(ncpu / config['ncpu_per_run'])
    
    print(
        '\nTuning hyperparameters;\n'
        f'  Available resources: {ngpu} GPUs | {ncpu} CPUs\n'
        f'  Number of concurrent runs: {max_concurrent}\n'
        )
    
    print(
        '\nTuning hyperparameters;\n'
        f'  Available resources: {ncpu} CPUs\n'
        f'  Number of concurrent runs: {max_concurrent}\n'
    )

    ray.tune.run(
        Emulator,
        config=best_config,
        resources_per_trial={
            'cpu': config['ncpu_per_run'],
            'gpu': config['ngpu_per_run']},
        num_samples=1,
        local_dir=store,
        raise_on_failed_trial=False,
        verbose=1,
        with_server=False,
        ray_auto_init=False,
        loggers=[JsonLogger, CSVLogger],
        keep_checkpoints_num=1,
        reuse_actors=False,
        stop={
            'patience_counter': config['patience']
        }
    )
    summarize_run(store)
