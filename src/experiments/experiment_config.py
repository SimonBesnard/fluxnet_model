import ConfigSpace as CS


def decode_config_name(config_name):
    def raise_format_error():
        msg = (
            'expected `config_name` of format '
            '`(nee | gpp | reco).(w | n)_perm` but got '
            f'`{config_name}`.')
        raise ValueError(msg)

    s = config_name.split('.')
    if len(s) != 2:
        raise_format_error()

    if s[1][0] == 'w':
        is_temporal = False
    elif s[1][0] == 'n':
        is_temporal = True
    else:
        raise_format_error()

    return is_temporal


def get_search_space(config_name):
    """Get search space.

    Parameters
    ----------
    config_name (str):
        The configuration name, one of
        - '(nee | gpp | reco).n_perm': temporal model.
        - '(nee | gpp | reco).w_perm': static model.

    Returns
    ----------
    Hyperparameter search space.

    """

    is_temporal = decode_config_name(config_name)

    config_space = CS.ConfigurationSpace()

    if not is_temporal:
        # Model args, non-temporal case.
        config_space.add_hyperparameters([
            # Dense hidden size.
            CS.UniformIntegerHyperparameter(
                'dense_hidden_size', lower=50, upper=600, q=50),
            # Dense number of layers.
            CS.UniformIntegerHyperparameter(
                'dense_num_layers', lower=2, upper=6, q=1),
            # Dropout probability for input.
            CS.UniformFloatHyperparameter(
                'dropout_in', lower=0.0, upper=0.5, q=0.1),
            # Dropout probability for dense layers.
            CS.UniformFloatHyperparameter(
                'dropout_linear', lower=0.0, upper=0.5, q=0.1)
        ])
    else:
        # Model args, temporal case.
        config_space.add_hyperparameters([
            # LSTM hidden size.
            CS.UniformIntegerHyperparameter(
                'lstm_hidden_size', lower=50, upper=300, q=50),
            # LSTM number of layers.
            CS.UniformIntegerHyperparameter(
                'lstm_num_layers', lower=1, upper=3, q=1),
            # Dense hidden size.
            CS.UniformIntegerHyperparameter(
                'dense_hidden_size', lower=50, upper=300, q=50),
            # Dense number of layers.
            CS.UniformIntegerHyperparameter(
                'dense_num_layers', lower=2, upper=6, q=1),
            # Dropout probability for LSTM.
            CS.UniformFloatHyperparameter(
                'dropout_lstm', lower=0.0, upper=0.5, q=0.1),
            # Dropout probability for input.
            CS.UniformFloatHyperparameter(
                'dropout_in', lower=0.0, upper=0.5, q=0.1),
            # Dropout probability for dense layers.
            CS.UniformFloatHyperparameter(
                'dropout_linear', lower=0.0, upper=0.5, q=0.1)
        ])

    # Optim args.
    config_space.add_hyperparameters([
    # The learning rate.
    CS.CategoricalHyperparameter(
        'learning_rate', choices=[1e-2, 1e-3, 1e-4, 1e-5]),
    # Weight decay (L2 regularization).
    CS.CategoricalHyperparameter(
        'weight_decay', choices=[1e-2, 1e-3, 1e-4])
    ])

    return config_space


def get_config(config_name):
    """Get hard-coded experiment configuration.

    This configuration involves
    - experiment-independent settings like number of GPUs per trial.
    - configurations for different experiments that are not parameters to
      be tuned, e.g. whether to add noise to the features.

    Parameters
    ----------
    config_name (str):
        The configuration name, one of
        - '(nee | gpp | reco).n_perm': temporal model.
        - '(nee | gpp | reco).w_perm': static model.

    Returns
    ----------
    Dict of configurations.

    """

    is_temporal = decode_config_name(config_name)

    # This is the default config, all other configurations need to build upon this.
    global_config = {
        'experiment_name': config_name,
        # Directory for logging etc.
        'store': '/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/experiments/cross_validation/',
        # The name of the metric to MINIMIZE. Make shure this is part of
        # the dict returned by 'YourTrainable._train'.
        'metric': 'loss_eval',
        # The optimizer to use.
        'optimizer': 'Adam',
        # The loss function to use.
        'loss_fn': 'MSE',
        # BOHB hyperband parameters (it is best to not change this):
        # - max_t: Maximum resources (in terms of epochs).
        # - Successive halving factor.
        # - num_samples (https://github.com/ray-project/ray/issues/5775)
        'max_t': 100,
        'halving_factor': 3,
        'num_samples': 134,  # 81 + 27 + 9 + 6 + 5,
        # Early stopping arguments: This applies to model tuning only, where
        # the best configuration from BOHB is used.
        # - grace_period: Minimum number of epochs.
        # - patience: After 'grace_period' number of epochs, training is stopped if
        #   more than 'patience' epochs have worse performance than current best, than
        #   predicitons are made.
        'grace_period': 50,
        'patience': 30,
        # Number of CPUs to use per run.
        'ncpu_per_run': 20,
        # Number of GPUs to use per run (0, 1].
        'ngpu_per_run': 0,
        # Number of workers per data loader.
        'num_workers': 0,
        # Batch size is not part of the hyperparaeter search as we use
        # a batch size that optimizes training performance.
        'batch_size': 64,
        # Number of batches per training epoch. This is equivalent to setting a logging frequency.
        'train_sample_size': 600,
        # Whether to pin memory; see torch.utils.data.dataloader:Dataloader.
        'pin_memory': False,
        # Validation fraction
        'valid_frac': 0.3,
        #Number of cross-validation fold
        'nfolds':10,
        'training_procedure': 'global',
        # Data configuration:
        'data_path': '/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/input_data/fluxnet_qc',
        'cv_path': '/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/input_data/folds',
        'output_path': '/Net/Groups/BGI/work_3/LSTM_CO2flux_upscaling/lstm_fluxnet/output_data',        
        'input_vars': [
            "NIRv",
            "NDWI_band6",
            "EVI",
            "LST_Day_1km",
            "LST_Night_1km",
            "TA",
            "P",
            "SW_IN",
            "VPD"
        ],
        'input_vars_static': [
            "tree_cover"
        ],
        'target_var': 'NEE',
        'qc_var': 'NEE_QC',
        'qc_threshold': 0.85,
        'time': {
            # Time range of the time series
            "date_range":{"start_date": "2000-01-01","end_date": "2015-12-31"},
            # Number of warmup years.
            'warmup': 1
        },
        # If true, use LSTM, else FC.
        'is_temporal': False
    }

    if is_temporal:
        global_config['is_temporal'] = True
    else:  
        global_config['time']['warmup'] = 0
    
        
    if config_name.split('.')[0] == 'nee':
        pass
    elif config_name.split('.')[0] == 'gpp':
        # Change target variable.
        global_config['target_var'] = 'GPP_DT'
    elif config_name.split('.')[0] == 'reco':
        # Change target variable.
        global_config['target_var'] = 'RECO_DT'
    else:
        raise ValueError(
            f'Argument `config_name`: {config_name} not a valid configuration.'
        )
    return global_config