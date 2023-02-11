import tensorflow as tf
import cv2 
import numpy as np
import matplotlib as mpl
from itertools import combinations
import ruptures as rpt
import random
from scipy.special import comb
import sklearn
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cosine
from methods.utils import *

class Lime:
    
    def __init__(self, model, nsegments=5, nsamples=1000, feature_faker=lambda _min, _max, _mean, _std, _size: 0,
                verbose=False):
        self.model = model
        self.nsegments = nsegments
        self.nsamples = nsamples
        self.feature_faker = feature_faker
        self.random_state = 666
        self.verbose = verbose
    
    def _get_weights(self, data):
        
        def distance_fn(x):
            ref = x[0]
            distance = np.zeros((x.shape[0],))
            for i in range(x.shape[0]):
                distance[i] = cosine(x[i], ref)
            return distance
        
        distances = distance_fn(data)

    
    def explain(self, input_array):
        
        mean_pred = self.model.predict(input_array)[0][0]
        
        if len(input_array.shape) > 2:
            input_array = input_array.squeeze()
        
        # compute the mean prediction
        #mean_pred = self.model.predict(np.array([mean_sample(input_array)]))[0][0]
        
        
        # create the mask to train the linear model
        nsamples = self.nsamples
        mask = []
        predictions = []
        segments = segment(input_array, self.nsegments)
        segments = sorted(segments, key=(lambda x: x[1]))
        samples = sampling(input_array, segments, n=nsamples, feature_faker=self.feature_faker)
        zs = np.array([s[1] for s in samples])
        predictions = self.model.predict(zs, batch_size=128)
        for zprime, z in samples:
            mask.append(zprime)

 
        # weights  of the masks
        weights = self._get_weights(np.array(mask))
        
        # train a ridge model to predict the predictions from the masks: 
        #    which relation exists beetween the permuted samples and the predictions?
        model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        model_regressor.fit(mask, predictions, sample_weight=weights)
        
        prediction_score = model_regressor.score(mask, predictions, sample_weight=weights)
        local_pred = model_regressor.predict(mask[0:1])

        if self.verbose:
            print('Prediction score', prediction_score)
            print('Intercept', model_regressor.intercept_)
            print('Prediction_local', local_pred,)

        heatmap = np.zeros(input_array.shape, np.float32)
        for (i, s, e), imp in zip(segments, model_regressor.coef_[0]):
            heatmap[i,s:e] = imp
            
        #heatmap = cv2.resize(heatmap.T, dsize=input_array.shape, interpolation=cv2.INTER_CUBIC).T
        
        return heatmap


    def display(self, input_array, explanation):
        
        s = input_array[0].squeeze()
        aux_hm = cv2.resize(explanation.T, dsize=s.shape, interpolation=cv2.INTER_CUBIC).T
 
        aux_hm =  (aux_hm - aux_hm.min()) / (aux_hm.max() - aux_hm.min())
        importances = [[(i, j, j+1, aux_hm[i,j]) for j in range(s.shape[1])] for i in range(s.shape[0])]

        display(s.T, importances)    
