import tensorflow as tf
import cv2 
import numpy as np
import matplotlib as mpl
from tensorflow.python.ops import gen_nn_ops
from tensorflow.keras import backend as K
from methods.utils import *

# Based on: https://github.com/atulshanbhag/Layerwise-Relevance-Propagation
class LRP:

    def __init__(self, model, alpha=2, epsilon=1e-7, debug=False, mode='original'):
        self.model = model
        self.alpha = alpha
        self.beta = 1 - alpha
        self.epsilon = epsilon
        self.debug = debug
        self.mode = mode

        self.names, self.activations, self.weights, self.layers = get_model_params(self.model)
        self.num_layers = len(self.names)

        self.relevance = self.compute_relevances()
        self.lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])

    def compute_relevances(self):
        r = self.model.output
        outputs = [r*0+4, r]
        for i in range(self.num_layers-2, -1, -1):
            name = self.names[i + 1]
            layer = self.layers[i + 1]
            if 'fc' in name or 'dense' in name or 'predictions' in name:
                r = self.backprop_fc(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
                outputs.append(tf.zeros_like(r))
                outputs.append(r)
            elif 'flatten' in name:
                r = self.backprop_flatten(self.activations[i], r)
                outputs.append(tf.ones_like(r))
                outputs.append(r)
            elif 'pool' in name:
                r = self.backprop_max_pool2d(self.activations[i], r, layer)
                outputs.append(tf.ones_like(r)*2)
                outputs.append(r)
            elif 'conv' in name:
                r = self.backprop_conv2d(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r, layer)
                outputs.append(tf.ones_like(r)*3)
                outputs.append(r)
            elif 'batch' in name:
                continue
                #gamma, beta, moving_mean, moving_variance = self.weights[i + 1]
                #r = (((r * moving_variance) + moving_mean) / gamma) - beta
                
                #outputs.append(r)
            elif 'activation' in name or 'dropout' in name:
                continue
            else:
                raise Exception(f'Layer {name} not recognized!')

        return r if not self.debug else outputs
    
    def backprop_fc(self, w, b, a, r):
        
        if self.mode == 'source':
            w_p = K.maximum(w, 0.)
            b_p = K.maximum(b, 0.)
            z_p = K.dot(a, w_p) + b_p + self.epsilon
            s_p = r / z_p
            c_p = K.dot(s_p, K.transpose(w_p))

            w_n = K.minimum(w, 0.)
            b_n = K.minimum(b, 0.)
            z_n = K.dot(a, w_n) + b_n - self.epsilon
            s_n = r / z_n
            c_n = K.dot(s_n, K.transpose(w_n))

            return a * (self.alpha * c_p + self.beta * c_n)
        else:
            w = K.constant(w)
            b = K.constant(b)
            z = self.epsilon + K.dot(a, w) + b
            s = r / z
            c = K.dot(s, K.transpose(w))

            return a * c        
        

    def backprop_flatten(self, a, r):
        shape = a.get_shape().as_list()
        shape[0] = -1
        return K.reshape(r, shape)

    def backprop_max_pool2d(self, a, r, layer):
        
        ksize = (1, layer.pool_size[0], layer.pool_size[1], 1)
        strides =  (1, layer.strides[0], layer.strides[1], 1)
        padding = layer.padding.upper()
        
        r = tf.clip_by_value(r, -self.epsilon, self.epsilon)
        if self.mode == 'source':
            
            z = K.pool2d(a, pool_size=ksize[1:-1], strides=strides[1:-1], padding=padding.lower(), pool_mode='max')

            z_p = K.maximum(z, 0.) + self.epsilon
            s_p = r / z_p
            c_p = gen_nn_ops.max_pool_grad_v2(a, z_p, s_p, ksize, strides, padding=padding)

            z_n = K.minimum(z, 0.) - self.epsilon
            s_n = r / z_n
            c_n = gen_nn_ops.max_pool_grad_v2(a, z_n, s_n, ksize, strides, padding=padding)

            rn =  a * (self.alpha * c_p + self.beta * c_n)
            
            rn = tf.clip_by_value(rn, -self.epsilon, self.epsilon)

             # scale to keep conservation of the relevances
            ratio =  tf.math.reduce_sum(r) /  tf.math.reduce_sum(rn)

            rn = rn * ratio
            rn = tf.clip_by_value(rn, -self.epsilon, self.epsilon)
            return rn

        else:
            z = K.pool2d(a, pool_size=ksize[1:-1], strides=strides[1:-1], 
                         padding=padding.lower(), pool_mode='max') + self.epsilon
            s = r / z
            c = gen_nn_ops.max_pool_grad_v2(a, z, s, ksize, strides, padding=padding)
            
            rn = a * c
            rn = tf.clip_by_value(rn, -self.epsilon, self.epsilon)
            
             # scale to keep conservation of the relevances
            ratio =  tf.math.reduce_sum(r) /  tf.math.reduce_sum(rn)

            rn = rn * ratio
            rn = tf.clip_by_value(rn, -self.epsilon, self.epsilon)
            return rn
           


    def backprop_conv2d(self, w, b, a, r, layer, is_input=False):
        ksize = (1, layer.kernel_size[0], layer.kernel_size[1], 1)
        strides =  (1, layer.strides[0], layer.strides[1], 1)
        padding = layer.padding.upper()
        
        if self.mode == 'source':
            w_p = K.maximum(w, 0.)
            b_p = K.maximum(b, 0.)
            z_p = K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding='same') + b_p + self.epsilon
            s_p = r / z_p
            c_p = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_p, s_p, strides, padding='SAME')

            w_n = K.minimum(w, 0.)
            b_n = K.minimum(b, 0.)
            z_n = K.conv2d(a, kernel=w_n, strides=strides[1:-1], padding='same') + b_n - self.epsilon
            s_n = r / z_n
            c_n = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_n, s_n, strides, padding='SAME')

            rn =  a * (self.alpha * c_p + self.beta * c_n)

            
             # scale to keep conservation of the relevances
            ratio =  tf.math.reduce_sum(r) /  tf.math.reduce_sum(rn)

            return rn * ratio
            
        else:
            if not is_input:
                w = K.constant(w)
                b = K.constant(b)
                z = K.conv2d(a, kernel=w, strides=strides[1:-1], padding=padding.lower()) + b + self.epsilon
                s = r / z
                c = tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w, s, strides, padding=padding)

                rn = a * c

                # scale to keep conservation of the relevances
                ratio =  tf.math.reduce_sum(r) /  tf.math.reduce_sum(rn)

                return rn * ratio
            else:

                w_p = K.maximum(w, 0.)
                w_n = K.minimum(w, 0.)

                lb = a * 0 - 1
                hb = a * 0 + 1

                z = (K.conv2d(a, kernel=w_p, strides=strides[1:-1], padding=padding.lower()) - 
                      K.conv2d(lb, kernel=w_p, strides=strides[1:-1], padding=padding.lower()) - 
                      K.conv2d(hb, kernel=w_n, strides=strides[1:-1], padding=padding.lower()) + b + self.epsilon)


                s = r / z
                c,cp,cn  = (tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w, s, strides, padding=padding),
                            tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_p, s, strides, padding=padding),
                            tf.compat.v1.nn.conv2d_backprop_input(K.shape(a), w_n, s, strides, padding=padding))

                return a * c - lb*cp - hb * cn

    
    def explain(self, input_array):
 
        e = self.lrp_runner([input_array, ])[0]

        if self.debug:
            for i in range(0, len(e)-1, 2):
                j = int(e[i].flatten()[0])
                layer = ['dense', 'flatten', 'pool', 'conv', 'output'][j]
                print(layer, e[i+1].sum(),  e[i+1].max(),  e[i+1].min())
                
            e = e[-1]
        
        e = gamma_correction(e)[0]
        r = reduce_channels(e, axis=-1, op='sum')
        r = project_image(r)
        
        return r
    
    def display(self, input_array, explanation):
        
        s = input_array[0].squeeze()
        aux_hm = cv2.resize(explanation.T, dsize=s.shape, interpolation=cv2.INTER_CUBIC).T
 
        aux_hm =  (aux_hm - aux_hm.min()) / (aux_hm.max() - aux_hm.min())
        importances = [[(i, j, j+1, aux_hm[i,j]) for j in range(s.shape[1])] for i in range(s.shape[0])]

        display(s.T, importances)       
        
   