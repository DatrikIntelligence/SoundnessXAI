import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
from model.models import create_mscnn_model, SplitTS
from methods.utils import *
import methods
from scoring import *
import pickle as pk
import tqdm
from ray import tune
import ray
import time
import gc
import multiprocessing


ray.shutdown()
ray.init(num_cpus=4, num_gpus=2)


def validate(model, samples, targets, time_dimension, feature_dimension, **kwargs):


    results = {}
   
    layers = list(kwargs.keys())
    factors = list(kwargs.values())

    explainer = methods.GradCAM_LW(model, layers, factors, 
                                        feature_dimension=feature_dimension,
                                        time_dimension=time_dimension)
    
    
    for proxy in ['selectivity', 'cs', 'identity','stability', 'separability', 'acumen']:
                
        if proxy == 'identity':
            validation_function = lambda model, explainer: methods.validate_identity(model, explainer, samples, verbose=False)

        elif proxy == 'selectivity':
            validation_function = lambda model, explainer: methods.validate_selectivity(model, explainer, samples, samples_chunk=1, verbose=False)

        elif proxy == 'stability':
            validation_function = lambda model, explainer: methods.validate_stability(model, explainer, samples, verbose=False)

        elif proxy == 'separability':
            validation_function = lambda model, explainer: methods.validate_separability(model, explainer, samples, verbose=False)

        elif proxy == 'cs':
            validation_function = lambda model, explainer: methods.validate_coherence(model, explainer, samples, targets, verbose=False)

        elif proxy == 'acumen':
            validation_function = lambda model, explainer: methods.validate_acumen(explainer, samples, verbose=False)

        r = validation_function(model, explainer)
        if isinstance(r, dict):
            results.update(r)
        else:
            results[proxy] = r

    return results



def train(config):
    
    model_path = config.pop('model')
     
    samples = pk.load(open(os.path.join(model_path, 'samples.pk'), 'rb')).astype(np.float32)
    targets = pk.load(open(os.path.join(model_path, 'targets.pk'), 'rb')).astype(np.float32)
    print("Samples readed")
    
    
    model = tf.keras.models.load_model(os.path.join(model_path, 'model.h5'), 
                                   custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU,
                                                  'NASAScore': NASAScore,
                                                  'SplitTS': SplitTS,
                                                  'PHM21Score': PHM21Score})
    
    layers = [l.name for l in model.layers if 'conv2d' in l.name]
    
    result = validate(model, samples, targets, **config)
    
    result['score'] = np.mean(list(result.values()))
    
    print(config)
    print(result)
   
    tune.report(**result)
    

from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    # Adding optional argument
    parser.add_argument("-m", "--Model", help = "Model directory", required=True)
    parser.add_argument("-s", "--samples", help = "Number of search samples", required=False, default=100, type=int)

    # Read arguments from command line
    args = parser.parse_args()

    
    def get_layers(return_dict):
        model = tf.keras.models.load_model(os.path.join(args.Model, 'model.h5'), 
                                   custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU,
                                                  'NASAScore': NASAScore,
                                                  'SplitTS': SplitTS,
                                                  'PHM21Score': PHM21Score})
    
        layers = [l.name for l in model.layers if 'conv2d' in l.name]
        return_dict['layers'] = layers
    
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process_eval = multiprocessing.Process(target=get_layers, args=(return_dict,))
    process_eval.start()
    process_eval.join()

    layers = return_dict['layers']
    
    points_to_evaluate = [
        {"conv2d_8": 0.0, "conv2d_9": 0.0, "conv2d_10": 1.0, "conv2d_11": 0.0, "conv2d_12": 0.0, "conv2d_13": 0.0, "conv2d_14": 0.0, "conv2d_15": 0.0, "feature_dimension": 0.5, "time_dimension": 0.5},
        {"conv2d_8": 0.0, "conv2d_9": 0.1, "conv2d_10": 0.0, "conv2d_11": 0.0, "conv2d_12": 0.0, "conv2d_13": 0.0, "conv2d_14": 0.0, "conv2d_15": 0.0, "feature_dimension": 0.5, "time_dimension": 0.5},
        {"conv2d_8": 0.1, "conv2d_9": 0.0, "conv2d_10": 0.0, "conv2d_11": 0.0, "conv2d_12": 0.0, "conv2d_13": 0.0, "conv2d_14": 0.0, "conv2d_15": 0.0, "feature_dimension": 0.5, "time_dimension": 0.5},
        
    ]
    
    

    space = {
        "time_dimension": (0, 1),
        "feature_dimension": (0, 1),   
    }
    
    for l in layers:
        space.update({l: (0,1)})
    
    
    
    bayesopt = BayesOptSearch(space=space, mode="max", metric="score",
                              random_search_steps=30,
                              points_to_evaluate=points_to_evaluate)
    scheduler=ASHAScheduler(metric="score", mode="max", max_t=3600, time_attr='training_iteration')

    analysis = tune.run(
        train,
        config={
            "model": os.path.join(os.path.dirname(os.path.realpath(__file__)), args.Model),
        },
        resources_per_trial={'gpu': 1},
        num_samples=args.samples,
        search_alg=bayesopt,
        log_to_file=False,
        scheduler=scheduler
        )


    #pk.dump(analysis._checkpoints, open('tune_checkpoint_fast_charge.pk', 'wb'))
