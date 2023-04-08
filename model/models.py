from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Reshape, Input, Dropout
from tensorflow.keras.layers import BatchNormalization, Lambda, Conv2DTranspose, Add
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints
import tensorflow.keras.backend as K
import numpy as np
import inspect
from scoring import *
from tensorflow.keras.layers.experimental.preprocessing import Resizing

ACTIVATIONS = ['relu', tf.keras.layers.LeakyReLU(alpha=0.1), 'tanh']
KERNELS = [(3,3), (1, 10), (5, 10), (10, 1), (10, 5)]      
SCORERS = [loss_score, 'mean_squared_error']


class SplitTS(tf.keras.layers.Layer):
    def __init__(self, chunk_size=128, *args, **kwargs):
        super(SplitTS, self).__init__(*args, **kwargs)
        
        self.chunk_size = chunk_size
 
    def build(self, input_shape):
        pass

    
    def call(self, x):
        L = x.shape[2]
        n = L // self.chunk_size
        chunks = tf.split(x, n, axis=2)
        x = tf.keras.backend.concatenate(chunks, axis=-1)
        return x
    
    
    def get_config(self):
        config = super().get_config()
        config.update({"chunk_size": self.chunk_size})
        return config


def dec(value, dtype):
    if dtype == int:
        value = int(round(value))
    elif dtype == bool:
        value = False if round(value) == 0 else True
    elif dtype == float:
        value = round(float(value), 15)
    elif dtype == 'scorer':
        value = SCORERS[dec(value, int)]
    elif dtype == 'activation':
        value = ACTIVATIONS[dec(value, int)]
    elif dtype == 'kernel':
        value = KERNELS[dec(value, int)]
    elif dtype == 'rnn_cell':
        value = tf.keras.layers.LSTMCell if int(round(value)) == 0 else tf.keras.layers.GRUCell
    elif dtype == str:
        value = str(value)
    elif dtype == tuple:
        value = value
    else:
        raise Exception("dtype %s does not found for decoding value" % dtype)
    
    return value

PREPROCESS_LAYERS = {
    'pronostia': SplitTS,
}

def create_mscnn_model(input_shape, block_size=2, nblocks=2, kernel_size=1, l1=1e-5, l2=1e-4, msblocks=2,
                       f1=10, f2=15, f3=20, dropout=0.5, lr=1e-3, filters=64,
                       fc1=256, fc2=128, conv_activation=2, dense_activation=2, 
                       dilation_rate=1, batch_normalization=1, scorer=1, pooling_kernel=(2, 2),
                       input_folding_size=False):
    block_size = dec(block_size, int)
    scorer = dec(scorer, 'scorer') 
    nblocks = dec(nblocks, int)
    msblocks = dec(msblocks, int)
    fc1 = dec(fc1, int)
    if fc2 <= 1:
        fc2 = int(fc1 * fc2)
    else:
        fc2 = dec(fc2, int)
    
    f1 = dec(f1, int)
    f2 = dec(f2, int)
    f3 = dec(f3, int)
    ms_kernel_size = [f1, f2, f3]
    
    dilation_rate = dec(dilation_rate, int)
    conv_activation = dec(conv_activation, 'activation')
    dense_activation = dec(dense_activation, 'activation')
    kernel_size = dec(kernel_size, 'kernel')
    batch_normalization = dec(batch_normalization, bool)

    input_tensor = Input(input_shape)
    x = input_tensor
    if input_folding_size:
        x = SplitTS(input_folding_size)(x)
        
    for i, _ in enumerate(range(msblocks)):

        cblock = []
        for k in range(3):
            output_shape = x.shape
            f = ms_kernel_size[k]
  
            b = Conv2D(filters, kernel_size=(f, 1), padding='same', 
                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                       kernel_initializer='he_uniform',
                       name='MSConv_%d%d_%d' % (i, k, f),
                       dilation_rate=dilation_rate)(x)

            if batch_normalization:
                b = BatchNormalization()(b)
            b = Activation(conv_activation)(b)

            cblock.append(b)

        x = Add()(cblock)
        if dropout > 0:
            x = Dropout(dropout)(x)
    
    
    for i, n_cnn in enumerate([block_size] * nblocks):
        for j in range(n_cnn):
            x = Conv2D(filters*2**min(i, 2), kernel_size=kernel_size, padding='same', 
                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                       kernel_initializer='he_uniform',
                       dilation_rate=dilation_rate)(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            x = Activation(conv_activation)(x)

        if (x.shape[2] == 1 and pooling_kernel[1] > 1) or (x.shape[1] == 1 and pooling_kernel[0] > 1):
            break
            
        x = MaxPooling2D(pooling_kernel)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)           

            
    x = Flatten()(x)
    
    # FNN
    x = Flatten()(x)
    x = Dense(fc1, 
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(fc2, tf.keras.layers.LeakyReLU(alpha=0.1),
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Activation(dense_activation)(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Dense(1, activation='relu', name='predictions')(x) 
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss=scorer, optimizer=Adam(lr=lr), 
                  metrics=[NASAScore(), PHM21Score(), tf.keras.metrics.MeanAbsoluteError(name="MAE")])
    
    return model
