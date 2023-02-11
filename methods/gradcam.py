import tensorflow as tf
import cv2 
import numpy as np
import matplotlib as mpl
from tensorflow.python.ops import gen_nn_ops
from tensorflow.keras import backend as K
from methods.utils import *

class GRADCAM:
    
    def __init__(self, model, conv_layer_name, time_dimension=False, 
                 feature_dimension=False):
        self.model = model
        self.conv_layer_name = conv_layer_name
        self.time_dimension = time_dimension
        self.feature_dimension = feature_dimension

    def explain(self, input_array):

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.conv_layer_name).output, self.model.output]
        )

        # Then, we compute the gradient of the top prediction for our input 
        # with respect to the activations the conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(input_array)

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the conv layer
        grads = tape.gradient(preds, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Compute the time grads and feature grads.
        time_grads = tf.reduce_mean(grads, axis=(0, 1, 3))
        feature_grads = tf.reduce_mean(grads, axis=(0, 2, 3))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top prediction
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = tf.squeeze(last_conv_layer_output @ pooled_grads[..., tf.newaxis])
        
        if self.time_dimension:
            td = 1.0 if self.time_dimension == True else float(self.time_dimension)
                
            heatmap = heatmap + (td * time_grads)

        if self.feature_dimension:
            fd = 1.0 if self.feature_dimension == True else float(self.feature_dimension)
            heatmap = tf.transpose(tf.transpose(heatmap) + fd * feature_grads) 
            
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap.T, dsize=input_array.shape[1:3], interpolation=cv2.INTER_CUBIC).T
        
        return heatmap
    
    def display(self, input_array, explanation):
        
        s = input_array[0].squeeze()
        aux_hm = cv2.resize(explanation.squeeze().T, dsize=s.shape, interpolation=cv2.INTER_CUBIC).T
 
        aux_hm =  (aux_hm - aux_hm.min()) / (aux_hm.max() - aux_hm.min())
        importances = [[(i, j, j+1, aux_hm[i,j]) for j in range(s.shape[1])] for i in range(s.shape[0])]

        display(s.T, importances)
        
        
class GRADCAM2:
    
    def __init__(self, model, conv_layer_names, factors, time_dimension=False, 
                 feature_dimension=False):
        
        self.explainers = [GRADCAM(model, l, time_dimension, feature_dimension) 
                           for l in conv_layer_names]
        self.factors = factors
        
        
  
    def explain(self, input_array):
        
        heatmap = None
        for explainer, factor in zip(self.explainers, self.factors):
            if heatmap is None:
                heatmap = explainer.explain(input_array) * factor
            else:
                heatmap = heatmap + explainer.explain(input_array) * factor
                
        return heatmap
        
        