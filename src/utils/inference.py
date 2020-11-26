from models.emulator import Emulator, get_target_path
from experiments.experiment_config import get_config
import argparse
import os
import ray
import pickle
import shutil
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '4'


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

    args = parser.parse_args()

    return args


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
         fold = self.config['fold'])
    
def tune(config_name:str = 'default', fold=None, overwrite:bool= True):

    config = get_config(config_name)
    config.update({'is_tune': False})
    config['time'].update({'train_seq_length': 0})
    config.update({'permute': False})

    model_tune_store = get_target_path(config, mode='modeltune')
    model_restore_path = os.path.join(
        model_tune_store,
        'model.pth'
    )
    store = get_target_path(config, mode='inference')

    if overwrite:
        if os.path.isdir(store):
            shutil.rmtree(store)
    else:
        if os.path.isdir(store):
            raise ValueError(
                f'The directory {store} exists. Set flag "--overwrite" '
                'if you want to overwrite runs - all existing runs will be lost!')
    os.makedirs(store)

    best_config = load_best_config(model_tune_store)
    best_config.update({'hc_config': config})
    best_config['hc_config']['fold'] = fold
    
    # Inference is a single run, we can use more resources.
    #best_config['hc_config']['ncpu_per_run'] = 60
    #best_config['hc_config']['ngpu_per_run'] = 1
    #best_config['hc_config']['num_workers'] = 20
    best_config['hc_config']['batch_size'] = len(best_config['hc_config']['fold'])

    #%% Creating data output directory
    output_dir = os.path.join(best_config['hc_config']['output_path'], config_name)
    if os.path.isdir(output_dir):
        f'The directory {output_dir} exists.'
    else:
        os.makedirs(output_dir)

    print('Restoring model from: ', model_restore_path)
    e = Emulator(best_config)
    e._restore(model_restore_path)
    e._predict(output_dir, predict_training_set=False)
    #e._predict(output_dir, predict_training_set=True)