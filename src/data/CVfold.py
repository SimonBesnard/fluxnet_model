#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:28:56 2018

@author: simonbesnard
"""
import numpy as np

def create_folds(data, nfolds=10):
    sites = data.site.values
    np.random.shuffle(sites)
    _folds = np.array_split(sites, nfolds)
    return _folds

        