#%% Load library
from data.split_train_test_data import get_fold
from data.data_loader import Data
from models.trainer import Trainer
from models.lstm import LSTM
from models.multilinear import DENSE
from models.modules import BaseModule
from ray import tune
from torch.utils.data.dataloader import DataLoader
import torch
import os
from utils import loss_functions
from sklearn.model_selection import train_test_split
import xarray as xr   
import numpy as np     

#% Define funtions for the emulator
def get_target_path(config, mode):
    if mode not in ['hptune', 'modeltune', 'inference']:
        raise ValueError(
            'Argument `mode` must be one of (`hptune`, `modeltune`, `inference`) but is {mode}.')
    path = os.path.join(
        config["store"],
        config['target_var'],
        config['experiment_name'],
        mode)
    return path

#%% Define funtions for the emulator
class Emulator(tune.Trainable):
    def _setup(self, config):
        
        self.hc_config = config['hc_config']
        
        #%% Get training, validation and test sets
        data = xr.open_mfdataset(self.hc_config['data_path'] + '/*.nc')
        data[self.hc_config['target_var']] = data[self.hc_config['target_var']].where(data[self.hc_config['qc_var']] >= self.hc_config['qc_threshold']) 
        if self.hc_config['training_procedure'] == 'cross_validation':
            self.train_data, self.test_data = get_fold(data, self.hc_config['fold'])
            self.train_indx, self.valid_indx = train_test_split(self.train_data.site, test_size = self.hc_config['valid_frac'])
            self.train_data, self.valid_data = self.train_data.sel(site = self.train_indx), self.train_data.sel(site = self.valid_indx) 
        elif self.hc_config['training_procedure'] == 'global':
            self.train_indx, self.valid_indx = train_test_split(data.site, test_size = self.hc_config['valid_frac'])
            self.train_data, self.valid_data = data.sel(site = self.train_indx), data.sel(site = self.valid_indx) 
            self.test_data = self.valid_data
        self.is_tune = self.hc_config['is_tune']
        
        activation = torch.nn.ReLU()
        #activation = torch.nn.LeakyReLU()

        train_loader = get_dataloader(
            self.hc_config,
            data = self.train_data,
            partition_set='train',
            is_tune=self.is_tune,
            batch_size=self.hc_config['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )
        eval_loader = get_dataloader(
            self.hc_config,
            data = self.valid_data,
            partition_set='valid',
            is_tune=self.is_tune,
            batch_size=self.hc_config['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )

        test_loader = get_dataloader(
            self.hc_config,
            data = self.test_data,
            partition_set='test',
            is_tune=self.is_tune,
            batch_size=self.hc_config['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=self.hc_config['num_workers'],
            pin_memory=self.hc_config['pin_memory']
        )
        
        if not self.hc_config['is_temporal']:
            model = DENSE(
                input_size=train_loader.dataset.num_dynamic + train_loader.dataset.num_static,
                hidden_size=config['dense_hidden_size'],
                num_layers=config['dense_num_layers'],
                activation=activation,
                dropout_in=config['dropout_in'],
                dropout_linear=config['dropout_linear']
            )
        else:
            model = LSTM(
                num_dynamic=train_loader.dataset.num_dynamic,
                num_static=train_loader.dataset.num_static,
                lstm_hidden_size=config['lstm_hidden_size'],
                lstm_num_layers=config['lstm_num_layers'],
                dense_hidden_size=config['dense_hidden_size'],
                dense_num_layers=config['dense_num_layers'],
                output_size=1,
                dropout_in=config['dropout_in'],
                dropout_lstm=config['dropout_lstm'],
                dropout_linear=config['dropout_linear'],
                dense_activation=activation
            )

        if not isinstance(model, BaseModule):
            raise ValueError(
                'The model is not a subclass of models.modules:BaseModule')

        if self.hc_config['optimizer'] == 'Adam':
            optimizer = torch.optim.AdamW(
                model.parameters(), config['learning_rate'], weight_decay=config['weight_decay'])
        else:
            raise ValueError(
                f'Optimizer {self.hc_config["optimizer"]} not defined.')

        if self.hc_config['loss_fn'] == 'MSE':
            #loss_fn = torch.nn.MSELoss()
            loss_fn = loss_functions.nanmse
        else:
            raise ValueError(
                f'Loss function {self.hc_config["loss_fn"]} not defined.')

        self.trainer = Trainer(
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader = test_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            #train_seq_length=self.hc_config['time']['train_seq_length'],
            train_sample_size=self.hc_config['train_sample_size']
        )
        
    def _train(self):
        train_stats = self.trainer.train_epoch()
        eval_stats = self.trainer.eval_epoch()

        stats = {**train_stats, **eval_stats}

        # Disable early stopping before 'grace period' is reached.
        if stats['epoch'] < self.hc_config['grace_period']:
            stats['patience_counter'] = -1

        return stats

    def _stop(self):
        if not self.is_tune:
            self._save(self.hc_config['store'])

    def _predict(self, prediction_dir, predict_training_set=False):

        self.trainer.predict(prediction_dir, predict_training_set)

    def _save(self, path):
        path = os.path.join(path, 'model.pth')
        return self.trainer.save(path)

    def _restore(self, path):
        self.trainer.restore(path)

def get_dataloader(
        config,
        data,
        partition_set,
        is_tune,
        **kwargs):

    dataset = Data(
        config=config,
        data = data,
        partition_set=partition_set,
        is_tune=is_tune)
    dataloader = DataLoader(
        dataset=dataset,
        **kwargs
    )
    return dataloader
