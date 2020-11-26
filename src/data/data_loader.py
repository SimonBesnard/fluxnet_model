import xarray as xr
import numpy as np
import pandas as pd
#import torch
from torch.utils.data.dataset import Dataset

class Data(Dataset):
    """Data loader for in-situ FLUXNET data.

    Notes
    ----------
    
    Parameters
    ----------
    config (dict):
        Configuration.
    data (xarray):
        dataset to be used    
    partition_set (str):
        Cross validation partition set, one of 'train' | 'valid'.
    permute_time: bool
        Whether to permute the samples, default is False.

    Returns
    ----------
    features_d: nd array
        Dynamic features with shape <time, num_features>
    target: nd array
        Dynamic target with shape <time, 1>
    """

    def __init__(
            self,
            config,
            data,
            partition_set,
            is_tune=False):

        if partition_set not in ['train', 'valid', 'test']:
            raise ValueError(
                f'Argument `partition_set`: Must be one of: [`train` | `valid`].')

        def msg(
            x): return f'Argument ``{x}`` is not an iterable of string elements.'
        if not self._check_striterable(config['input_vars']):
            raise ValueError(msg('input_vars'))
        if not isinstance(config['target_var'], str):
            raise ValueError('Argument ``target_var`` must be a string.')

        self.input_vars = config['input_vars']
        self.input_vars_static = config['input_vars_static']
        self.target_var = config['target_var']
        self.qc_var = config['qc_var']
        self.qc_threshold = config['qc_threshold']       

        self.partition_set = partition_set
        self.is_tune = is_tune
        
        self.config = config
        
        self.data = data
        self.coords = self.data.site.values
        
        self.dynamic_vars, self.static_vars = self._get_static_and_dynamic_varnames()

        self.time_slicer = TimeSlice(
            data = self.data,
            date_range=self.config['time']['date_range'],
            warmup=self.config['time']['warmup'],
            partition_set=self.partition_set)

        self.num_warmup_steps = self.time_slicer.num_warmup
       
        self.ds_stats = {
            var: {
                'mean': np.float32(data[var].mean(dim= ('time', 'site'))),
                'std': np.float32(data[var].std(dim= ('time', 'site')))
            } for var in data.data_vars
        }

        self._check_all_vars_present_in_dataset()
        self._check_var_time_dim()

        self.num_dynamic = len(self.input_vars)
        self.num_static = len(self.input_vars_static)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, inx):
        
        site = self.coords[inx]
        
        t_start, t_end = self.time_slicer.get_time_range()

        # Each single temporal variable has shape <time, site>. We select one coordinate, yielding
        # shape <time>. All variables are then stacked along last dimension, yielding <time, num_vars>
        features_d = np.stack(
            [self.standardize(self.data[var].sel(time = slice(t_start, t_end), site = site), var)
             for var in self.input_vars],
            axis=-1
        )
        features_s = self.standardize(self.data.sel(site =site)[self.input_vars_static[0]].mean(dim='time'), self.input_vars_static[0]).values.reshape(1, -1)
        
        # The (temporal) target variable has shape <time, site>. We select one coordinate, yielding
        # shape <time>, and expand in the last dimension, yielding <time, 1>.
        target = self.standardize(
            self.data.sel(time = slice(t_start, t_end), site = site)[self.target_var],
                                self.target_var).values.reshape(-1, 1)
        #target = self.data.sel(time = slice(t_start, t_end), site = site)[self.target_var].values.reshape(-1, 1).astype('float32')

        if np.any(np.isnan(features_d)):
            raise ValueError('NaN in features, training stopped.')

        return features_d, features_s, target, site

    def get_empty_xr(self):
        data = self.data[self.target_var].sel(
                time=slice(self.time_slicer.seldate_first, self.time_slicer.seldate_last))
        return data

    def data_QA(self,x):
        return x[self.target_var].where(x[self.qc_var] >= self.qc_threshold)
        
    def standardize(self, x, varname):
        return ((x - self.ds_stats[varname]['mean']) / self.ds_stats[varname]['std']).astype('float32')

    def unstandardize(self, x, varname):
        return x * self.ds_stats[varname]['std'] + self.ds_stats[varname]['mean']

    def _check_striterable(self, x):
        is_iterable_non_str = hasattr(x, '__iter__') & (not isinstance(x, str))
        all_elements_are_str = all([isinstance(x_, str) for x_ in x])
        return is_iterable_non_str & all_elements_are_str

    def _check_all_vars_present_in_dataset(self):
        def msg(
            x): return f'Variable ``{x}`` not found in dataset located at {self.config["path"]}'

        for var in self.input_vars + [self.target_var]:
            if var not in self.data:
                raise ValueError(msg(var))

    def _check_var_time_dim(self):
        def msg(
            x, y): return f'Variable ``{x}`` seems to be {"non-" if y else ""}temporal, check the variable arguments.'

        for var in self.input_vars + [self.target_var]:
            if self.data[var].ndim != 2:
                raise ValueError(msg(var, True))

    def _get_static_and_dynamic_varnames(self):
        time_vars = []
        non_time_vars = []
        for var in self.input_vars:
            time_vars.append(var)
        for var in self.input_vars_static:
            non_time_vars.append(var)            
        return time_vars, non_time_vars

class TimeSlice(object):
    """Manage time slicing for training and validation set.

    Parameters
    ----------
    ds_path: str
        Path to the Fluxnet data (.nc format).
    data_range: tuple(str, str)
        Date range to read, e.g. ('2000-01-01', '2015-12-31')
    warmup: int
        Number of warmup years that are not used in loss function. For the training set,
        the warmup period is added **after** the lower time bound, for the validation set it
        is added **before** the lower time bound, overlapping with the training time-range.
    partition_set: str
        One of ['train' | 'valid'].

    """

    def __init__(
            self,
            data,
            date_range,
            warmup,
            partition_set):

        if partition_set not in ['train', 'valid', 'test']:
            raise ValueError(
                f'Argument `partition_set`: Must be one of: [`train` | `valid`].')

        self.partition_set = partition_set
        
        ds_time = data.time
        date_first = pd.to_datetime(ds_time.values[0])
        date_last = pd.to_datetime(ds_time.values[-1])

        seldate_first = pd.to_datetime(date_range['start_date'])
        seldate_last = pd.to_datetime(date_range['end_date'])

        if not (date_first <= seldate_first < date_last):
            raise ValueError(
                f'The selected lower time-series bound ({seldate_first}) '
                f'is not in the dataset time range ({date_first} - {date_last})'
            )
        if not (date_first < seldate_last <= date_last):
            raise ValueError(
                f'The selected upper time-series bound ({seldate_last}) '
                f'is not in the dataset time range ({date_first} - {date_last})'
            )

        warmup_delta = pd.DateOffset(years=warmup)

        # warmup_first is the actual lower limit after applying the warmup period
        # seldate_first is the first date after the warmup period
        warmup_first = seldate_first
        seldate_first += warmup_delta
        if not (date_first <= seldate_first <= date_last):
            raise ValueError(
                f'After applying the warmup period of {warmup} year(s), the upper time-series '
                f'bound ({seldate_first}) is not in the dataset time range ({date_first} - {date_last}). '
                f'Note that the warmup period is added after the lower date range in the train set.'
            )

        self.warmup_first = warmup_first
        self.seldate_first = seldate_first
        self.seldate_last = seldate_last

        date_range = pd.date_range(date_first, date_last)

        self.num_warmup = len(ds_time.sel(time = slice(warmup_first, seldate_first))[:-1])
        self.start_t = date_first
        self.end_t = date_last
        self.seq_len = len(ds_time)

    def get_time_range(self):
        return self.start_t, self.end_t

    def __repr__(self):
        s = (
            f'TimeSlice object\n\n'
            f'Partition set: {self.partition_set}\n'
            f'Sample lenth: {"full sequence"}d\n\n'
            f'   warmup period: {self.num_warmup:5d}steps        sample period: {self.seq_len-self.num_warmup:5d}steps\n'
            f'|-------------------------|------------------------------|\n'
            f'|                         |                              | \n'
            f'{self.warmup_first.strftime("%Y-%m-%d")}            '
            f'{self.seldate_first.strftime("%Y-%m-%d")}                '
            f'{self.seldate_last.strftime("%Y-%m-%d")}\n\n'
            f'warmup start: {self.warmup_first.strftime("%Y-%m-%d")}\n'
            f'sample start: {self.seldate_first.strftime("%Y-%m-%d")}\n'
            f'sample end:   {self.seldate_last.strftime("%Y-%m-%d")}\n'

        )
        return s
