import argparse
import methods
import imp
imp.reload(methods)
import tensorflow as tf
import pickle as pk
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import os
import numpy as np
from scoring import *
from tensorflow.python.framework.ops import disable_eager_execution

def validate(validation_function):
    feature_faker_list = {
        'mean': lambda _min, _max, _mean, _std, _size: np.random.normal(_mean, _std, _size),
        'normal_noise': lambda _min, _max, _mean, _std, _size: np.random.normal(_mean, _std, _size),
        'uniform_noise': lambda _min, _max, _mean, _std, _size: np.random.uniform(_min, _max, _size),
        'zero': lambda _min, _max, _mean, _std, _size: 0,
        'one': lambda _min, _max, _mean, _std, _size: 1,   
    }

    results = []
   
    # Lime
    print("Lime")
    for ff_name, feature_faker in feature_faker_list.items():

        explainer = methods.Lime(model, nsamples=1000, nsegments=5, 
                               feature_faker=feature_faker)

        score =  validation_function(model, explainer)
        results.append({'method': 'Lime', 
                        'feature faker': ff_name, 
                        'score': score})    
        
    print("Saliency")
    explainer = methods.Saliency(model)

    score = validation_function(model, explainer)
    results.append({'method': 'Saliency', 
                    'score': score})    


 
        
    # Lawer Wise Relevance Propagation (LRP)
    print("LRP")
    for mode  in ['source', 'none']:

        explainer = methods.LRP(model, mode=mode)

        score =  validation_function(model, explainer)
        results.append({'method': 'LRP', 'mode': mode, 
                        'score': score})    


    # GRAD CAM
    print("GRAD CAM")
    layers = [l.name for l in model.layers if 'conv2d' in l.name]
    for last_conv_layer_name in layers:
        for feature_dimension in [True, False]:
            for time_dimension in [True, False]:
                #last_conv_layer_name = f"conv2d_{layer}"
                explainer = methods.GRADCAM(model, last_conv_layer_name, 
                                                    feature_dimension=feature_dimension,
                                                    time_dimension=time_dimension)

                score = validation_function(model, explainer)

                results.append({'method': 'GRADCAM', 
                                'time_dimension': time_dimension,
                                'feature_dimension': feature_dimension,
                                'layer': last_conv_layer_name, 
                                'score': score})

    # Kernel SHAP
    print("Kernel SHAP")
    for ff_name, feature_faker in feature_faker_list.items():

        explainer = methods.KernelSHAP(model, nsamples=1000, nsegments=5, 
                               feature_faker=feature_faker)

        score = validation_function(model, explainer)
        results.append({'method': 'Kernel SHAP', 'feature faker': ff_name, 
                        'score': score})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # Adding optional argument
    parser.add_argument("-d", "--Device", help = "CUDA DEVICE")
    parser.add_argument("-p", "--Proxy", help = "{identity, stability, }", required=True)

    # Read arguments from command line
    args = parser.parse_args()
    
    if args.Device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.Device
    
    
    samples2 = pk.load(open('data/samples2.pk', 'rb')).astype(np.float32)
    targets2 = pk.load(open('data/targets2.pk', 'rb')).astype(np.float32)
    print("Samples readed")
    
    
    seed = 666
    model = tf.keras.models.load_model('model/cnn2_6.43_%d.h5' % seed, 
                                   custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU,
                                                  'NASAScore': NASAScore,
                                                  'PHM21Score': PHM21Score})
    print("Model readed")
    
    if args.Proxy == 'identity':
        validation_function = lambda model, explainer: methods.validate_identity(model, explainer, samples2)
        results = validate(validation_function)
        pk.dump(results, open('identity.pk', 'wb'))
        
    elif args.Proxy == 'selectivity':
        validation_function = lambda model, explainer: methods.validate_selectivity(model, explainer, samples2)
        results = validate(validation_function)
        pk.dump(results, open('selectivity.pk', 'wb'))

    elif args.Proxy == 'stability':
        validation_function = lambda model, explainer: methods.validate_stability(model, explainer, samples2)
        results = validate(validation_function)
        pk.dump(results, open('stability.pk', 'wb'))
        
    elif args.Proxy == 'separability':
        validation_function = lambda model, explainer: methods.validate_separability(model, explainer, samples2)
        results = validate(validation_function)
        pk.dump(results, open('separability.pk', 'wb'))
        
    elif args.Proxy == 'cs':
        validation_function = lambda model, explainer: methods.validate_coherence(model, explainer, samples2, targets2)
        results = validate(validation_function)
        pk.dump(results, open('cs.pk', 'wb'))   
        
    elif args.Proxy == 'acumen':
        validation_function = lambda model, explainer: methods.validate_acumen(explainer, samples2)
        results = validate(validation_function)
        pk.dump(results, open('acumen.pk', 'wb'))           
