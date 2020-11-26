import torch
from torch import jit
from utils.loggers import EpochLogger
import os
from torch import Tensor
import numpy as np

def nanmae(
        pred: Tensor,
        target: Tensor) -> Tensor:
    """Mean absolute error (MAE) loss, handels NaN values.

    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.

    Returns
    ----------
    loss: torch.Tensor
        The mae loss, a tensor of one element.
    """

    mask = torch.isnan(target)
    target[mask] = pred[mask]
    cnt = torch.sum(~mask, dtype=target.dtype)

    mae = torch.abs(pred - target).sum() / cnt

    return mae

def nanmse(
    input_:Tensor,
    target_:Tensor)-> Tensor:
    """Mean squared error (MSE) loss, handels NaN values.
    
    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.
    
    Returns
    ----------
    loss: torch.Tensor
        The mse loss, a tensor of one element.
    """
    # Compute loss at daily scale
    mask = torch.isnan(target_)
    target_[mask] = input_[mask]
    cnt = torch.sum(~mask, dtype=target_.dtype)
    loss_daily = torch.pow(input_ - target_, 2).sum() / cnt 
    
    # Compute loss at spatial scale
    #pred_spatial = torch.sum(input_, axis=1) / input_.shape[1]
    #target_spatial = torch.sum(target_, axis=1) / target_.shape[1]
    #mask = torch.isnan(target_spatial)
    #cnt = torch.sum(~mask, dtype=target.dtype)
    #loss_spatial = torch.pow(target_spatial - pred_spatial, 2).sum() / cnt 
    return loss_daily #+ loss_spatial.mean()

def nanrmse(
        pred: Tensor,
        target: Tensor) -> Tensor:
    """root mean squared error (RMSE) loss, handels NaN values.

    Parameters
    ----------
    pred: torch.Tensor
        Prediction tensor of shape (batch_size, sequence_length, 1).
    target: torch.Tensor
        Target tensor of shape (batch_size, sequence_length, 1), same
        shape as ``pred``.
    clamp: Iterable[float]
        Optional values used to cut clamp[0] < values < clamp[1] and set
        them to the respective threshold.

    Returns
    ----------
    loss: torch.Tensor
        The rmse loss, a tensor of one element.
    """

    return torch.sqrt(nanmse(pred, target))
