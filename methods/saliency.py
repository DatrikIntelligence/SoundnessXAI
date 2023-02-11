import tensorflow as tf
import cv2 
import numpy as np
import matplotlib as mpl
from tensorflow.keras import backend as K
import cv2 
import numpy as np
from methods.utils import *

class Saliency:
    
    def __init__(self, model):
        self.model = model

    def explain(self, input_array, class_idx=-1):
        
        input_ = tf.Variable(input_array)
        with tf.GradientTape() as tape:
            tape.watch(input_)
            predictions = self.model(input_)

            loss = predictions[:, class_idx]

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_)

        # take maximum across channels
        gradient = tf.reduce_max(gradient, axis=-1)

        # convert to numpy
        gradient = gradient.numpy()

        # normalization between 0 and 1
        min_val, max_val = np.min(gradient), np.max(gradient)
        heatmap = (gradient - min_val) / (max_val - min_val + tf.keras.backend.epsilon())

        return heatmap[0]
    
    def display(self, input_array, explanation):
        
        s = input_array[0].squeeze()
        aux_hm = cv2.resize(explanation.squeeze().T, dsize=s.shape, interpolation=cv2.INTER_CUBIC).T
 
        aux_hm =  (aux_hm - aux_hm.min()) / (aux_hm.max() - aux_hm.min())
        importances = [[(i, j, j+1, aux_hm[i,j]) for j in range(s.shape[1])] for i in range(s.shape[0])]

        display(s.T, importances)
        
     
    