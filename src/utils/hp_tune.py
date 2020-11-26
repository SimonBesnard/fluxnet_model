##%% Load library
from utils.summarize_runs import summarize_run
from experiments.experiment_config import get_search_space, get_config
from models.emulator import Emulator, get_target_path
from ray.tune.logger import CSVLogger, JsonLogger
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
import ray
import argparse
import os
import shutil
import numpy as np
import logging
import sys
import torch
from ray.tune.progress_reporter import CLIReporter

#os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'

#%% Define tuning function
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_name',
        '-c',
        type=str,
        help='Configuration name.',
        default='default'
    )

    parser.add_argument(
        '--overwrite',
        '-O',
        help='Flag to overwrite existing runs (all existing runs will be lost!).',
        action='store_true'
    )

    parser.add_argument(
        '--run_single',
        help='Bypass ray.tune and run a single model train / eval iteration.',
        action='store_true'
    )

    args = parser.parse_args()

    return args

    def _train(self):
        train_stats = self.trainer.train_epoch()
        test_stats = self.trainer.test_epoch()

        stats = {**train_stats, **test_stats}

        # Disable early stopping before 'grace period' is reached.
        if stats['epoch'] < self.hc_config['grace_period']:
            stats['patience_counter'] = -1

        return stats

    def _save(self, path):
        path = os.path.join(path, 'model.pth')
        return self.trainer.save(path)

    def _restore(self, path):
        self.trainer.restore(path)
        
def tune_run(self):
    tune(config_name= self.config['experiment_name'],
         fold = self.config['fold'],
         overwrite=True,
         run_single= False)

def tune(config_name:str = 'default', fold=None, overwrite:bool= True, run_single:bool = False):

    search_space = get_search_space(config_name)
    config = get_config(config_name)

    store = get_target_path(config, 'hptune')

    if overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The tune directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all existing runs will be lost!')
    os.makedirs(store)

    config.update({
        'is_tune': True,
        'fold': fold
    })

    #ngpu = torch.cuda.device_count()
    ncpu = os.cpu_count()

    #max_concurrent = int(
    #    np.min((
    #        np.floor(ncpu / config['ncpu_per_run']),
    #        np.floor(ngpu / config['ngpu_per_run'])
    #    ))
    #)

    max_concurrent = np.floor(ncpu / config['ncpu_per_run'])
    
    
    #print(
    #    '\nTuning hyperparameters;\n'
    #    f'  Available resources: {ngpu} GPUs | {ncpu} CPUs\n'
    #    f'  Number of concurrent runs: {max_concurrent}\n'
    #)
    
    print(
        '\nTuning hyperparameters;\n'
        f'  Available resources: {ncpu} CPUs\n'
        f'  Number of concurrent runs: {max_concurrent}\n'
    )


    bobh_search = TuneBOHB(
        space=search_space,
        max_concurrent=max_concurrent,
        metric=config['metric'],
        mode='min'
    )

    bohb_scheduler = HyperBandForBOHB(
        time_attr='epoch',
        metric=config['metric'],
        mode='min',
        max_t=config['max_t'],
        reduction_factor=config['halving_factor'])

    if run_single:
        logging.warning('Starting test run.')
        e = Emulator(search_space.sample_configuration())
        logging.warning('Starting training loop.')
        e._train()
        logging.warning('Finishing test run.')
        sys.exit('0')
    ray.tune.run(
        Emulator,
        config={'hc_config': config},
        resources_per_trial={
            'cpu': config['ncpu_per_run'],
            'gpu': config['ngpu_per_run']},
        num_samples=config['num_samples'],
        local_dir=store,
        raise_on_failed_trial=True,
        verbose=1,
        with_server=False,
        ray_auto_init=False,
        search_alg=bobh_search,
        scheduler=bohb_scheduler,
        loggers=[JsonLogger, CSVLogger],
        keep_checkpoints_num=1,
        reuse_actors=False,
        stop={'patience_counter': config['patience']}, 
        progress_reporter = CLIReporter()
    )

    summarize_run(store)
