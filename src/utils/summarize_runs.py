"""
Analyze a ray.tune run; plot training curves, show best run etc.
"""

import argparse
from typing import Dict, Any
from ray.tune.analysis import Analysis
import os
import shutil
import glob2
from shutil import copyfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRAIN_METRIC = 'loss_train'
DEFAULT_VALID_METRIC = 'loss_eval'
#BASE_PATH = '/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/experiments/'
#DEFAULT_TARGET_BASE_DIR = '/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/experiments/'

def summarize_run(store):
    summarize(
        path=store,
        overwrite=True)

def summarize(
        path: str,
        train_metric: str = DEFAULT_TRAIN_METRIC,
        eval_metric: str = DEFAULT_VALID_METRIC,
        overwrite: bool = False) -> None:

    print(f'\nLoading experiment from:  \n{path}\n')

    #if not os.path.isfile(path):
    #    raise ValueError(f'Path does not exist or is directory:\n{path}')
    #if path[-5:] != '.json':
    #    raise ValueError(f'Not a .json file:\n{path}')

    #path_split = path.split(BASE_PATH)[1].split('/')
    #var = path_split[0]
    #name = path_split[1]
    #mode = path_split[2]
    summary_dir = os.path.join(path, 'summary')

    if os.path.isdir(summary_dir):
        if not overwrite:
            raise ValueError(
                f'Target directory `{summary_dir}` exists, use `--overwrite` to replace.')
        shutil.rmtree(summary_dir)
    os.makedirs(summary_dir)
    
    exp = Analysis(path)

    configs = exp.dataframe()
    configs['rundir'] = [os.path.join(l, 'progress.csv')
                         for l in configs['logdir']]
    runs = []
    for i, f in enumerate(configs['rundir']):
        df = pd.read_csv(f)
        df['uid'] = i
        runs.append(df)
    runs = pd.concat(runs)

    best_run_dir = exp.get_best_logdir(eval_metric, mode='min')
    best_run_file = os.path.join(best_run_dir, 'progress.csv')
    best_run = pd.read_csv(best_run_file)

    print(f'Best run ID: {best_run_dir}')

    for f in ['json', 'pkl']:
        in_file = os.path.join(best_run_dir, f'params.{f}')
        out_file = os.path.join(summary_dir, f'best_params.{f}')
        
        copyfile(in_file, out_file)

    # Plot runs.
    plot_all(runs, eval_metric, os.path.join(summary_dir, 'all_runs.png'))
    plot_single(best_run, eval_metric, os.path.join(
        summary_dir, 'best_run.png'))

def plot_all(runs: pd.core.frame.DataFrame, metric: str, savepath: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(
        8, 6), sharex=True, sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
    box = dict(facecolor='yellow', pad=6, alpha=0.2)

    ax[0].text(
        1.0, 1.0, 'HYPERBAND OPTIMIZATION', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
    ax[0].text(
        0.5, 0.98, 'TRAINING', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)
    ax[1].text(
        0.5, 0.98, 'VALIDATION', transform=ax[1].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)

    train_name = DEFAULT_TRAIN_METRIC
    valid_name = DEFAULT_VALID_METRIC

    runs.groupby(['uid']).plot(
        x='epoch', y=train_name, ax=ax[0], legend=False)
    runs.groupby(['uid']).plot(
        x='epoch', y=valid_name, ax=ax[1], legend=False)

    ymin = np.min((
        np.min(runs[train_name]),
        np.min(runs[valid_name]))) * 0.9
    ymax = np.max(
        (np.percentile(runs[train_name], 95), np.percentile(runs[valid_name], 95)))
    xmin = np.min(runs['epoch'])-np.max(runs['epoch'])*0.01
    xmax = np.max(runs['epoch'])*1.01

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)
    ax[0].yaxis.set_label_coords(-0.15, 0.5, transform=ax[0].transAxes)
    ax[0].set_ylabel('loss', bbox=box)

    fig.savefig(savepath, bbox_inches='tight', dpi=200, transparent=True)


def plot_single(single_run: pd.core.frame.DataFrame, metric: str, savepath: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(
        8, 6), sharex=True, sharey='row', gridspec_kw={'wspace': 0, 'hspace': 0})
    box = dict(facecolor='yellow', pad=6, alpha=0.2)

    ax[0].text(
        1.0, 1.0, 'BEST RUN', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
    ax[0].text(
        0.5, 0.98, 'TRAINING', transform=ax[0].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)
    ax[1].text(
        0.5, 0.98, 'EVALUATION', transform=ax[1].transAxes,
        horizontalalignment='center', verticalalignment='top', bbox=box)

    train_name = DEFAULT_TRAIN_METRIC
    valid_name = DEFAULT_VALID_METRIC

    single_run.plot(x='epoch', y=train_name, ax=ax[0], legend=False)
    single_run.plot(x='epoch', y=valid_name, ax=ax[1], legend=False)

    ymin = np.min((np.min(single_run[train_name]), np.min(
        single_run[valid_name]))) * 0.95
    ymax = np.max((np.percentile(single_run[train_name], 95), np.percentile(
        single_run[valid_name], 95)))
    xmin = np.min(single_run['epoch'])-np.max(single_run['epoch'])*0.01
    xmax = np.max(single_run['epoch'])*1.01

    ax[0].set_xlim(xmin, xmax)
    ax[0].set_ylim(ymin, ymax)
    ax[0].yaxis.set_label_coords(-0.15, 0.5, transform=ax[0].transAxes)
    ax[0].set_ylabel('loss', bbox=box)

    fig.savefig(savepath, bbox_inches='tight', dpi=200, transparent=True)
