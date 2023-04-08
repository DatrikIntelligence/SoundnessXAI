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
from itertools import product
from model.models import SplitTS

def validate(validation_function, feature_dimension, time_dimension, method_name, mode=1):
    feature_faker_list = {
        'mean': lambda _min, _max, _mean, _std, _size: np.random.normal(_mean, _std, _size),
        'normal_noise': lambda _min, _max, _mean, _std, _size: np.random.normal(_mean, _std, _size),
        'uniform_noise': lambda _min, _max, _mean, _std, _size: np.random.uniform(_min, _max, _size),
        'zero': lambda _min, _max, _mean, _std, _size: 0,
        'one': lambda _min, _max, _mean, _std, _size: 1,   
    }

    results = []
   
 
    # GRAD CAM
    if mode == 1:
        print("GRAD CAM")
        layers = [l.name for l in model.layers if 'conv2d' in l.name]
        for last_conv_layer_name in layers:
            #last_conv_layer_name = f"conv2d_{layer}"
            explainer = methods.GRADCAM(model, last_conv_layer_name, 
                                                feature_dimension=feature_dimension,
                                                time_dimension=time_dimension)

            score = validation_function(model, explainer)

            if type(score) is dict:
                for key, value in score.items():
                    results.append({'method': 'GRADCAM', 
                        'time_dimension': time_dimension,
                        'feature_dimension': feature_dimension,
                        'layer': last_conv_layer_name, 
                        'method': key,
                        'score': value})


            else:
                results.append({'method': 'GRADCAM', 
                                'time_dimension': time_dimension,
                                'feature_dimension': feature_dimension,
                                'layer': last_conv_layer_name, 
                                'method': method_name,
                                'score': score})
    
    elif mode == 2:
        print("GRAD CAM 2")
        layers = [l.name for l in model.layers if 'conv2d' in l.name][1:3]
        #last_conv_layer_name = f"conv2d_{layer}"
        explainer = methods.GRADCAM2(model, layers, [1, 0], 
                                            feature_dimension=feature_dimension,
                                            time_dimension=time_dimension)

        score = validation_function(model, explainer)

        if type(score) is dict:
            for key, value in score.items():
                results.append({'method': 'GRADCAM', 
                    'time_dimension': time_dimension,
                    'feature_dimension': feature_dimension,
                    'layer': layers, 
                    'method': key,
                    'score': value})


        else:
            results.append({'method': 'GRADCAM', 
                            'time_dimension': time_dimension,
                            'feature_dimension': feature_dimension,
                            'layer': layers, 
                            'method': method_name,
                            'score': score})

    return results


def exists_result(results, td, fd, proxy):
    if proxy == 'cs':
        proxy = 'coherence'
        
    for r in results:
        if ((r['time_dimension'] == td) and 
            (r['feature_dimension'] == fd) and
            (r['method'] == proxy)):
            return True
            
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # Adding optional argument
    parser.add_argument("-d", "--Device", help = "CUDA DEVICE")
    parser.add_argument("-c", "--SamplesChunk", help = "Selectivity sample chunk", default=1, required=False, type=int)
    parser.add_argument("-m", "--Mode", help = "GradCam mode", default=1, required=False, type=int)
    parser.add_argument("-mo", "--Model", help = "Model directory", required=True)
    parser.add_argument("-fd", "--FeatureDimension", help = "Feature Dimiension", default=None, required=False, type=float)
    parser.add_argument("-tds", "--TimeDimensionStart", help = "Time Dimiension Start", default=None, required=False, type=float)
    parser.add_argument("-fds", "--FeatureDimensionStart", help = "Feature Dimiension Start", default=None, required=False, type=float)
    parser.add_argument("-td", "--TimeDimension", help = "Time Dimiension", default=None, required=False, type=float)
    parser.add_argument("-o", "--Output", help = "Result file name", default=None, required=False, type=str)
    

    # Read arguments from command line
    args = parser.parse_args()
    
    if args.Device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.Device
    
    
    samples2 = pk.load(open(os.path.join(args.Model, 'samples.pk'), 'rb')).astype(np.float32)
    targets2 = pk.load(open(os.path.join(args.Model, 'targets.pk'), 'rb')).astype(np.float32)
    print("Samples readed")
    
    
    model = tf.keras.models.load_model(os.path.join(args.Model, 'model.h5'), 
                                   custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU,
                                                  'NASAScore': NASAScore,
                                                  'SplitTS': SplitTS,
                                                  'PHM21Score': PHM21Score})
    print("Model readed")
    results = []
    
    if args.Output is None:
        output_file = os.path.join(args.Model, 'results', f'gradcam{args.SamplesChunk}_{args.Mode}.pk')
    else:
        output_file = os.path.join(args.Model, 'results', args.Output)

    
    if os.path.exists(output_file):
        results = pk.load(open(output_file, 'rb'))
        
    
    if args.TimeDimension is None:
        if args.TimeDimensionStart is None:
            tds = [float(i)/10 for i in range(10)]
        else:
            tds = [float(i)/10 for i in range(int(args.TimeDimensionStart*10), 10)]
    else:
        tds = [args.TimeDimension]
        
    if args.FeatureDimension is None:
        if args.FeatureDimensionStart is None:
            fds = [float(i)/10 for i in range(10)]
        else:
            fds = [float(i)/10 for i in range(int(args.FeatureDimensionStart*10), 10)]
    else:
        fds = [args.FeatureDimension]
        
        
    print(tds, fds)
       
    for td, fd in product(tds, fds):

            for proxy in [ 'selectivity', 'cs', 'identity','stability', 'separability', 'acumen']:
                
                if exists_result(results, td, fd, proxy):
                    print(proxy, td, fd, "Done")
                    continue
                    
                print(proxy, td, fd, "Computing.....")                    

                if proxy == 'identity':
                    validation_function = lambda model, explainer: methods.validate_identity(model, explainer, samples2)
                
                elif proxy == 'selectivity':
                    validation_function = lambda model, explainer: methods.validate_selectivity(model, explainer, samples2, samples_chunk=args.SamplesChunk)
                
                elif proxy == 'stability':
                    validation_function = lambda model, explainer: methods.validate_stability(model, explainer, samples2)
                
                elif proxy == 'separability':
                    validation_function = lambda model, explainer: methods.validate_separability(model, explainer, samples2)
                
                elif proxy == 'cs':
                    validation_function = lambda model, explainer: methods.validate_coherence(model, explainer, samples2, targets2)
 
                elif proxy == 'acumen':
                    validation_function = lambda model, explainer: methods.validate_acumen(explainer, samples2)
                
                results += validate(validation_function, fd, td, proxy, mode=args.Mode)
                
                pk.dump(results, open(output_file, 'wb'))
                    
