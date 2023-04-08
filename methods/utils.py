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
from scipy.spatial.distance import euclidean
from scipy import stats
from tensorflow.python.ops import gen_nn_ops
import tensorflow as tf
from tensorflow.keras import backend as K
import os 
import tempfile
from keras.models import load_model
import tqdm

EPS = 1e-7

def reduce_channels(image, axis=-1, op='sum'):
    if op == 'sum':
        return image.sum(axis=axis)
    elif op == 'mean':
        return image.mean(axis=axis)
    elif op == 'absmax':
        pos_max = image.max(axis=axis)
        neg_max = -((-image).max(axis=axis))
    return np.select([pos_max >= neg_max, pos_max < neg_max], [pos_max, neg_max])


def gamma_correction(image, gamma=0.4, minamp=0, maxamp=None):
    c_image = np.zeros_like(image)
    image -= minamp
    if maxamp is None:
        maxamp = np.abs(image).max() + EPS
    image /= maxamp
    pos_mask = (image > 0)
    neg_mask = (image < 0)
    c_image[pos_mask] = np.power(image[pos_mask], gamma)
    c_image[neg_mask] = -np.power(-image[neg_mask], gamma)
    c_image = c_image * maxamp + minamp
    return c_image

def project_image(image, output_range=(0, 1), absmax=None, input_is_positive_only=False):
    if absmax is None:
        absmax = np.max(np.abs(image), axis=tuple(range(1, len(image.shape))))
    absmax = np.asarray(absmax)
    mask = (absmax != 0)
    if mask.sum() > 0:
        image[mask] /= np.expand_dims(absmax[mask], axis=-1)
    if not input_is_positive_only:
        image = (image + 1) / 2
    image = image.clip(0, 1)
    projection = output_range[0] + image * (output_range[1] - output_range[0])
    return projection

def get_model_params(model):
    names, activations, weights, layers = [], [], [], []
    for layer in model.layers:
        name = layer.name 
        names.append(name)
        activations.append(layer.output)
        weights.append(layer.get_weights())
        layers.append(layer)
        
    return names, activations, weights, layers

def display(
    signal,
    true_chg_pts,
    computed_chg_pts=None,
    computed_chg_pts_color="k",
    computed_chg_pts_linewidth=3,
    computed_chg_pts_linestyle="--",
    computed_chg_pts_alpha=1.0,
    **kwargs
):
    """Display a signal and the change points provided in alternating colors.
    If another set of change point is provided, they are displayed with dashed
    vertical dashed lines. The following matplotlib subplots options is set by
    default, but can be changed when calling `display`):
    - figure size `figsize`, defaults to `(10, 2 * n_features)`.
    Args:
        signal (array): signal array, shape (n_samples,) or (n_samples, n_features).
        true_chg_pts (list): list of change point indexes.
        computed_chg_pts (list, optional): list of change point indexes.
        computed_chg_pts_color (str, optional): color of the lines indicating
            the computed_chg_pts. Defaults to "k".
        computed_chg_pts_linewidth (int, optional): linewidth of the lines
            indicating the computed_chg_pts. Defaults to 3.
        computed_chg_pts_linestyle (str, optional): linestyle of the lines
            indicating the computed_chg_pts. Defaults to "--".
        computed_chg_pts_alpha (float, optional): alpha of the lines indicating
            the computed_chg_pts. Defaults to "1.0".
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.
    Returns:
        tuple: (figure, axarr) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MatplotlibMissingError(
            "This feature requires the optional dependency matpotlib, you can install it using `pip install matplotlib`."
        )

    if type(signal) != np.ndarray:
        # Try to get array from Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    # let's set a sensible defaut size for the subplots
    matplotlib_options = {
        "figsize": (10, 2 * n_features),  # figure size
    }
    # add/update the options given by the user
    matplotlib_options.update(kwargs)

    # create plots
    fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
    if n_features == 1:
        axarr = [axarr]

    cmap = mpl.colormaps['RdYlGn']
    for i, (axe, sig) in enumerate(zip(axarr, signal.T)):
        # plot s
        axe.plot(range(n_samples), sig)

        # color each (true) regime
        bkps = [0] + sorted(true_chg_pts)
        alpha = 0.4  # transparency of the colored background

        for (_, start, end, imp) in true_chg_pts[i]:
            axe.axvspan(max(0, start - 0.5), end - 0.5, color=cmap(imp), alpha=alpha)
        

    fig.tight_layout()

    return fig, axarr

def usegment(signal, n_bkps=5):
    """
    Compute the segmentation of the signal uniform
    """
    # Calcula el número de períodos completos en la señal x
    period = len(signal) // n_bkps

    # Divide la señal x en tantos períodos completos como sea posible
    result = []
    for i in range(signal.shape[0]):
        result += [(i, p, p + period) for p in range(0, len(signal), period)]
        
    return result

def segment(signal, n_bkps=5):
    """
    Compute the segmentation of the signal 
    """
    result = []
    for i in range(signal.shape[0]):
        algo = rpt.Dynp(model="l2").fit(signal[i].T)
        r = [0] + algo.predict(n_bkps)
        result += [(i, l, r) for (l,r) in zip(r, r[1:])]
        
    
    return result
        
def sampling(signal, segments, feature_faker, n=10, k=3):
    ranges = [(signal[i].min(), signal[i].max()) for i in range(signal.shape[0])]
    mean_std = [(signal[i].mean(), signal[i].std()) for i in range(signal.shape[0])]
    
    zprimes = []
    for kk in range(1, k):
        for seg in combinations(segments, kk):
            zprime = [1 if s in seg else 0 for s in segments]
            z = np.copy(signal)
            
            for zj, (i, start, end) in zip(zprime, segments):
                if zj == 0:
                   # z[i, start:end] = np.random.normal(mean_std[i][0],mean_std[i][1],end-start)
                  z[i, start:end] = feature_faker(*ranges[i], *mean_std[i], end-start)
                

            n -= 1
            zprimes.append((zprime, z))
            if n <= 0:
                break
        
        if n <= 0:
            break

    for _ in range(n):
        nsegments = random.randint(1, len(segments)-1)
        seg = random.sample(segments, nsegments)
        zprime = [1 if s in seg else 0 for s in segments]
        z = np.copy(signal)
        for zj, (i, start, end) in zip(zprime, segments):
            if zj == 0:
                #z[i, start:end] = np.random.normal(mean_std[i][0],mean_std[i][1],end-start)
                z[i, start:end] = feature_faker(*ranges[i], *mean_std[i], end-start)
 
                
        zprimes.append((zprime, z))
        
    return zprimes
        




def mean_sample(signal):
    """
    Creates a new sample 
    
    a normal distribution with mean and std of each 
    feature
    """
    mean_std = [(signal[i].mean(), signal[i].std()) for i in range(signal.shape[0])]
    
    result = np.zeros(signal.shape)
    for i in range(signal.shape[0]):
        #result[i] = np.random.normal(mean_std[i][0],mean_std[i][1], signal.shape[1])
        result[i] = mean_std[i][0]
       

    return result
    
    
def validate_acumen(explainer, samples, iterations=100, nstd=1.5, top_features=100, verbose=True):
    """
    Se eleccionan n puntos de los que tienen mayor score y se perturban creando una nueva
    serie (tp). También se crea otra serie (tr) metiendo ruido en otros n puntos aleatorios. 
    La importancia qm de puntos debe cumplir las siguiente regla qm(t) >= qm(tr) > qm(tp)
    """
    
    ranking = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1] 
        base_exp = explainer.explain(xi)

        if not np.isnan(base_exp).any():
            # compute the thresold using mean + n*std 
            _mean, _std = base_exp.mean(), base_exp.std()
            theshold = _mean + nstd*_std
            nsamples = (base_exp > theshold).sum()
            nsamples = min(nsamples, top_features)
            aux1 = base_exp.flatten()

            top_mask = aux1.argsort()[-nsamples:]

            # tc
            tc = np.copy(xi).reshape(base_exp.shape)
            tc[base_exp >= theshold] = 0
            tc_exp = explainer.explain(tc.reshape(xi.shape))

            # obtengo las n muestra más imporantes de x'
            aux = tc_exp.flatten()
            aux_max = aux.argsort().max()

            # ranking
            ranking.append((1- (np.argsort(aux).argsort()[top_mask] / aux_max)).mean())
        
    return np.nanmean(ranking)

def validate_coherence(model, explainer, samples, targets, nstd=1.5, top_features=100, verbose=True):
    explains = []
    valid_idx = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1] 
        exp = explainer.explain(xi)

        # compute the thresold using mean + n*std 
        _mean, _std = exp.mean(), exp.std()
        theshold = _mean + nstd*_std
        nsamples = (exp > theshold).sum()
        nsamples = min(nsamples, top_features)
        aux = exp.flatten()
        theshold = aux[aux.argsort()][-nsamples]
        indexes = np.argwhere(exp.flatten() < theshold)

        # remove that features
        exp[exp < theshold] = 0
        xic = np.copy(xi).flatten()
        xic[indexes] = 0
        xic = xic.reshape(xi.shape)

        if not np.isnan(exp).any():
            valid_idx.append(i)
            explains.append(xic)

    samples = samples[valid_idx]
    targets = targets[valid_idx]

    tmax = targets.max()
    targets = targets / tmax

    pred = model.predict(samples) / tmax
    pred = pred.reshape(targets.shape)
    errors = 1 - (pred - targets) ** 2

    exp = np.array(explains).reshape(samples.shape)

    explains = np.array(explains).reshape(samples.shape)
    exp_pred = model.predict(explains) / tmax
    exp_errors = 1- (exp_pred - targets) ** 2
    
    coherence_i = np.abs(errors - exp_errors)
    coherence = np.mean(coherence_i)
    
    return {
            'coherence': coherence, 
            'completeness':np.mean(exp_errors / errors),
            'congruency': np.sqrt(np.mean((coherence_i - coherence)**2))
           }
    

def validate_identity(model, explainer, samples, verbose=True):
    """
    The principle of identity states that identical objects should receive identical explanations. This is 
    a measure of the level of intrinsic non-determinism in the method:
    
                                d(xa , xb ) = 0 => ∀a,b d(εa , εb ) = 0
                                
    """
    errors = []
    for i, sample in tqdm.tqdm(enumerate(samples), total=samples.shape[0], disable=not verbose):
        exp_a = explainer.explain(samples[i:i+1])
        exp_b = explainer.explain(samples[i:i+1])
        
        if not np.isnan(exp_b).any() and not np.isnan(exp_b).any():
            errors.append(1 if np.all(exp_a == exp_b) else 0)
                               
    return np.nanmean(errors)
                              

def validate_separability(model, explainer, samples, verbose=True):
    """
     Non-identical objects can not have identical explanations:
    
                 ∀a,b; d(xa , xb ) ̸= 0 =>  d(εa , εb ) > 0

     This proxy is based on the assumption that every feature has 
     a minimum level of importance, positive or negative, in the 
     predictions. The idea is that if a feature is not actually 
     needed for the prediction, then two samples that differ only 
     in that feature will have the same prediction. In this 
     scenario, the explanation method could provide the same 
     explanation, even though the samples are different.

    """
    explains = []
    samples_aux = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1] 
        exp = explainer.explain(xi)
        
        if not np.isnan(exp).any():
            samples_aux.append(xi)
            explains.append(exp)
    
    samples = np.array(samples_aux)
    
    errors = []
    for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):
        
        for j in range(i+1, len(samples)-1):
            
            if i == j:
                continue
                
            exp_a = explains[i] #explainer.explain(samples[i:i+1])
            exp_b = explains[j] #explainer.explain(samples[i+1:i+2])

            assert np.any(samples[i] != samples[j])
            #assert np.sum((exp_a - exp_b)**2) > 0

            errors.append(1 if np.sum((exp_a - exp_b)**2) > 0 else 0)

    return np.nanmean(errors)


def validate_stability(model, explainer, samples, verbose=True):
    """
    Similar objects must have similar explanations. This is built 
    on the idea that an explanation method should only return 
    similar explanations for slightly different objects. The 
    spearman correlation ρ is used to define this
      
      ∀i;
      ρ({d(xi , x0 ), d(xi , x1 ), ..., d(xi , xn )}, 
        {d(εi , ε0 ), d(εi , ε1 ), ..., d(εi , εn )}) = ρi > 0
                                
    """
    explains = []
    samples_aux = []
    for i in tqdm.tqdm(range(len(samples)), total=len(samples), disable=not verbose):
        xi = samples[i:i+1] 
        exp = explainer.explain(xi)
        
        
        if not np.isnan(exp).any():
            samples_aux.append(xi)
            explains.append(exp)
            
    samples = np.array(samples_aux)
                    
    errors = []
    for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):
        dxs, des = [], []
        xi = samples[i:i+1]          
        for j in range(len(samples)):    
            if i==j:
                continue
                
            xj = samples[j:j+1]
            #exp_i = explainer.explain(xi)
            #exp_j = explainer.explain(xj)
            exp_i = explains[i]
            exp_j = explains[j]
            
            if np.isnan(exp_i).any() or np.isnan(exp_j).any():
                continue
            
            dxs.append(euclidean(xi.flatten(), xj.flatten()))
            des.append(euclidean(exp_i.flatten(), exp_j.flatten()))


        errors.append(stats.spearmanr(dxs, des).correlation)
        
                               
    return np.nanmean(errors)


def validate_selectivity(model, explainer, samples, samples_chunk=1, verbose=True):
    """
    The elimination of relevant variables must affect 
    negatively to the prediction. To compute the selectivity 
    the features are ordered from most to lest relevant. 
    One by one the features are removed, set to zero for 
    example, and the residual errors are obtained to get the 
    area under the curve (AUC).
    """

    errors = []
    for i in tqdm.tqdm(range(len(samples)-1), total=len(samples), disable=not verbose):
        dxs, des = [], []
        xi = samples[i:i+1]
        ei = explainer.explain(xi)
        if np.isnan(ei).any():
            continue
            
        idxs = ei.flatten().argsort()[::-1]
        xi = xi[0]
        xs = [xi]
        xprime = xi.flatten()
        l = idxs.shape[0]
        if samples_chunk >= 1:
            idxs = np.split(idxs, int(l/samples_chunk))
        
        for i in idxs:
            xprime[i] = 0
            xs.append(xprime.reshape(xi.shape))
            xprime = np.copy(xprime)
            
        preds = model.predict(np.array(xs), batch_size=32)[:,0]   
        e = np.abs(preds[1:] - preds[:-1]) / (preds[0] + 1e-12)
        e = np.cumsum(e)
        e = 1 - (e / (e.max() + 1e-12))
        score = 1 - np.mean(e)
        
        errors.append(score)
        
    return np.nanmean(errors)
        

    
def apply_modifications(model, custom_objects=None):
    """
    Aplicamos las modificaciones realizadas en el modelo creando un nuevo grafo de
    computación. Para poder modificar el grafo tenemos que grabar y leer el modelo.
    """
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)