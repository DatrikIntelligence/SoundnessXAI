import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
from models import create_mscnn_model
import pickle as pk
import tqdm
from ray import tune
import ray



# Data generator 
class FASTCHARGESequence(tf.keras.utils.Sequence):

    def __init__(self, data, batches_per_epoch=1000, batch_size=32, split_channel=False):
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.units = self.data.keys()
        self.data = data
        self.split_channel = split_channel
        D = data

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx):
        D = self.data
      
        X = np.zeros(shape=(self.batch_size, 9, 512, 1))
        Y = np.zeros(shape=(self.batch_size,))
        ncycles = len(self.units)
        for i in range(self.batch_size):
            unit, cycle = list(self.units)[random.randint(0, ncycles-1)]
            Db = self.data[(unit, cycle)][:,:]
            L = Db.shape[1] 
            
            k = random.randint(0, L-512) 
  
            X[i, :, :, 0] = Db[:-1, k:k+512]
            Y[i] = max(0, Db[-1, k:k+512].min())
            
        return X, Y
        

ray.shutdown()
ray.init(num_cpus=4, num_gpus=2)



def train(config):
     

    
    train_units = ['b1c25', 'b1c17', 'b1c15', 'b1c27', 'b1c41', 'b1c23', 'b1c22', 'b1c46', 'b1c11',
                   'b1c35', 'b1c28', 'b1c20', 'b1c30', 'b1c40', 'b1c43', 'b1c3', 'b1c13', 'b1c32', 
                   'b1c6', 'b1c1', 'b1c34', 'b1c18', 'b1c5', 'b1c39', 'b1c44', 'b1c10', 'b1c2', 'b1c0', 
                   'b1c31', 'b1c19', 'b1c37', 'b1c38']
    test_units = ['b1c12', 'b1c14', 'b1c16', 'b1c21', 'b1c24', 'b1c26', 'b1c29', 'b1c33', 'b1c36', 
                  'b1c4', 'b1c42', 'b1c45', 'b1c7', 'b1c8', 'b1c9']
  
    X = pk.load(open('/home/dasolma/papers/xai/data/fast_charge.pk', 'rb'))

    X_train = {k: v for k, v in X.items() if k[0] in train_units}
    X_test =  {k: v for k, v in X.items() if k[0] in test_units}
    
    def norm(values, i, _min, _max):
        values[i] = (values[i] - _min) / (_max - _min)
        return values

    for i in range(9):
        values = np.hstack([d[i] for k, d in X_train.items()])
        _min, _max = values.min(), values.max()

        X_train = {k: norm(v, i, _min, _max) for k, v in X_train.items()}
        X_test = {k: norm(v, i, _min, _max) for k, v in X_test.items()}

    gen_train = FASTCHARGESequence(X_train, batches_per_epoch=2500)
    gen_val = FASTCHARGESequence(X_test, batches_per_epoch=5000)
    
    epochs = config.pop("epochs")
    
    m = create_mscnn_model((9,512,1),**config)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
    history = m.fit(gen_train, validation_data=gen_val,
                    batch_size=32, epochs=epochs, verbose=0,
                   callbacks=[es, rlr])
    history = history.history
    tune.report(score=history['val_loss'][-1])
    

from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler
space = {
    "block_size": (1.51, 4.5),
    #"msblocks": (-0.51, 4.5),
    "nblocks": (2.51, 6.5),
    "l1": (0, 1e-3),
    "l2": (0, 1e-3),
    "dropout": (0, 0.9),
    "lr": (1e-5, 1e-3),
    "fc1": (64, 256),
    "fc2": (0, 1),
    "conv_activation": (-0.51, 2.5),
    "dense_activation": (-0.51, 2.5),
    "dilation_rate": (0.51, 10.49),
    "kernel_size": (-0.51, 1.5),
    #"f1": (2.51, 15.5),
    #"f2": (2.51, 15.5),
    #"f3": (2.51, 15.5),
}


bayesopt = BayesOptSearch(space=space, mode="min", metric="score")
scheduler=ASHAScheduler(metric="score", mode="min", max_t=3600, time_attr='training_iteration')

analysis = tune.run(
    train,
    config={
        "epochs": 100,
        "input_folding_size": 128,
        "msblocks": 0,
    },
    resources_per_trial={'gpu': 1},
    num_samples=30,
    search_alg=bayesopt,
    log_to_file=False,
    scheduler=scheduler
    )


pk.dump(analysis._checkpoints, open('tune_checkpoint_fast_charge.pk', 'wb'))
