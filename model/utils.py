import os
import h5py
import time
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np 
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Reshape, Input, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import pickle as pk
import gc
import ray
import multiprocessing
from ray import tune
from data import *
from ray import tune
import sys
import logging
import traceback
import pandas as pd
import h5py
from scoring import loss_score
from models import dec, create_ae_from_encoder, SelfAttention
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.utils.util import unflatten_dict
from filelock import FileLock
from hashlib import sha1
from scoring import *
import inspect

logging.basicConfig(level=logging.INFO)

CONFIG_DEC = {
    "arch_hash": (str, None),
    "net_hash": (str, None),
    "ae_hash": (str, None),
    "batch_size": (int, 32),
    "window": (int, 100),
    "l1": (float, 1e-3),
    "l2": (float, 1e-3),
    "dropout": (float, 0.9),
    "batch_normalization": (bool, True),
    "lr": (float, 1e-3),
    
    "variational_mode": (bool, True),
    "dec_blocks": (int, 4),
    "dec_activation" : ('activation', 'relu'),
    "eta": (float, 0.5),

    "block_size": (int, 4),
    "nblocks": (int, 4),
    "msblocks": (int, 4),

    "conv_activation": ('activation', 'relu'),
    "kernel_size": ('kernel', (3,3)),
    "filters" : (int, 32),
    "dilation_rate": (int, 1),
    
    "dense_activation": ('activation', 'relu'),
    "fc1": (int, 64),
    "fc2": (int, 100),
    "f1": (int, 5),
    "f2": (int, 10),
    "f3": (int, 15),
    "pretrain": (bool, True),
    "freeze": (bool, False),
    
    "ae_loss": (float, None),
    "ae_rec_loss": (float, None),
    "ae_kl_loss": (float, None),
    "net_score": (float, None),
    "net_mae": (float, None),
    "net_mse": (float, None),
    "net_NASA_score": (float, None),
    "status": (str, None),
    "ae_time": (float, None),
    "net_time": (float, None),
    
 
}



CNNEMB_CONFIG_DEC = {
    "arch_hash": (str, None),
    "net_hash": (str, None),

    "batch_size": (int, 32),
    "window": (int, 100),
    "l1": (float, 1e-3),
    "l2": (float, 1e-3),
    "dropout": (float, 0.9),
    "batch_normalization": (bool, True),
    "lr": (float, 1e-3),
    
    "block_size": (int, 4),
    "nblocks": (int, 4),
    "msblocks": (int, 4),

    "conv_activation": ('activation', 'relu'),
    "kernel_size": ('kernel', (3,3)),
    "filters" : (int, 32),
    "dilation_rate": (int, 1),
    
    "dense_activation": ('activation', 'relu'),
    "fc1": (int, 64),
    "fc2": (int, 100),
    "f1": (int, 5),
    "f2": (int, 10),
    "f3": (int, 15),
    
    "net_score": (float, None),
    "net_mae": (float, None),
    "net_mse": (float, None),
    "net_NASA_score": (float, None),
    "status": (str, None),
    "net_time": (float, None),
    
    "step_min": (int, 100),
    "step_range": (int, 100),
    "channels": (int, 3),
    
}


class CustomBayesOptSearch(BayesOptSearch):
    
    def __init__(self, grid_params = [], *args, **kwargs):
        super(CustomBayesOptSearch, self).__init__(*args, **kwargs)
        
        self.__last_point = None
        self.__grid_params = grid_params
        self.__igrid = 0
    
    def suggest(self, trial_id: str):
        if self.__last_point is None or self.__igrid == len(self.__grid_params):
            config = super(CustomBayesOptSearch, self).suggest(trial_id)
            if config is None:
                return None
            self.__igrid = 0
            self.__last_point = unflatten_dict(config)
        
        
        config = self.__last_point 
        
        for key, value in self.__grid_params[self.__igrid].items():
            config[key] = value
            
        self.__igrid += 1
            
            
        return config
    

def decconf(config, key, dtype, default=None, pop=True):
    if key not in config:
        return default
    
    if pop:
        value = config.pop(key)
    else:
        value = config[key]
    
    value = dec(value, dtype)
    
    return value

def decconfull(config, defaults=CONFIG_DEC):
    
    for key, (dtype, default) in defaults.items():
        
        config[key] = decconf(config, key, dtype, default=default, pop=False)
        
        if hasattr(config[key], '__dict__'):         # is a user defined class
            config[key] = type(config[key]).__name__
        
    return config
        

def confighash(config, exclude=[]):
    if exclude is not None and len(exclude) > 0:
        config = config.copy()
        for key in exclude:
            if key in config:
                del config[key]
    
    return sha1(repr(sorted(config.items())).encode()).hexdigest()
        

def log_train(config):
    with FileLock('log_train.lock') as lock:
        try:
            log_path = 'log_train.csv'
            if os.path.exists(log_path):
                log = pd.read_csv(log_path)
                log = log.append(config, ignore_index=True)
            else:
                log = pd.DataFrame(data=[config])

            logging.info("Saving log train csv")
            log.to_csv(log_path, index=False)
        finally:
            lock.release()

def valid_params(config, func):
    config = config.copy()
    params = inspect.getargspec(func).args
    config = {k:v for k,v in config.items() if k in params}
    return config
            
HASH_EXCLUDE = ["ae_loss", "ae_rec_loss", "ae_kl_loss", "net_score", "net_mae", "net_mse", "net_NASA_score", 
                "net_hash", "arch_hash", "ae_hash", "status", "ae_time", "net_time"]

