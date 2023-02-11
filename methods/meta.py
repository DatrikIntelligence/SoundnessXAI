from tsfresh.feature_extraction import feature_calculators as fc
import tsfresh
import numpy as np
from scipy import signal, stats

def compute_metaattributes():
    attributes = []
    samples = []
    isample = 0
    for i in tqdm.tqdm(range(len(train_gen))):
        batch = train_gen.__getitem__(i)

        for x, y in zip(*batch):

            x = x[:, :, 0]
            samples.append(s)
            segments = usegment(x)

            for ksegment, (i, start, end) in enumerate(segments):
                s = x[i, start:end]

                att = get_atributes(s)

                assert not any([np.isnan(v) for k,v in att.items()])
                att['y'] = y[0]
                att['sample'] = isample
                att['feature'] = i
                att['segment'] = ksegment
                att['segment_start'] = start
                att['segment_end'] = end
                attributes.append(att)
                #print('.', end='')

            isample += 1
        
    return samples, attributes


    
# variación
def calculate_coefficient_of_variation(x):
    # Calcula la media y la desviación estándar de la señal x
    mean, std = np.mean(x), np.std(x)

    # Calcula el coeficiente de variación de la señal x
    return std / (mean + 1e-12)


def calculate_entropy(x):
    # Calcula la distribución de probabilidad de los valores de la señal x
    prob_distribution, _ = np.histogram(x, density=True)

    # Evita que la distribución de probabilidad contenga valores nulos
    prob_distribution += 1e-12

    # Calcula la entropía de la señal x
    entropy = stats.entropy(prob_distribution)

    return entropy

# nivel de ruido: 0 es ruido puro
def signaltonoise(a):
    m = a.mean()
    sd = a.std()
    return float(np.where(sd == 0, 0, m/sd))


# complejidad
def complexity(s):
    # Normalizamos la señal
    s = (s - s.min()) / (s.max() - s.min() + 1E-7)
    
    # Calcula la transformada de Fourier de la señal
    X = np.fft.fft(s)

    # Calcula la magnitud de la transformada de Fourier
    magnitude = np.abs(X) 
   
    # Calcula la complejidad de la señal como la suma de la magnitud de la transformada de Fourier
    # por unidad de tiempo
    complexity = np.sum(magnitude)  
 
    return complexity


def calculate_oscillation(x):
    # Calcula la desviación estándar de la señal
    std = np.std(x)

    # Calcula la media aritmética de la señal
    _mean = np.mean(x)

    # Calcula el nivel de oscilación como el cociente entre la desviación estándar y la media
    oscillation = std / (_mean + 1E-7)

    return np.abs(oscillation)


def calculate_stability(x):
    # Calcula la regresión lineal de la señal x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y=np.arange(0, len(x)))

    # Calcula la estabilidad como el valor absoluto del coeficiente de determinación
    stability = np.abs(r_value**2)

    return stability

def evaluate_periodicity(x, num_periods):
    # Calcula el número de períodos completos en la señal x
    period = len(x) // num_periods

    # Divide la señal x en tantos períodos completos como sea posible
    periods = np.split(x[:period*num_periods], num_periods)

    # Calcula la similitud entre cada par de períodos consecutivos
    similarities = []
    for i in range(num_periods-1):
        l, r = periods[i], periods[i+1]
        if np.all(l == r):
            similarity = 1
        elif np.all(l == l[0]) and np.all(r == r[0]):
            similarity = 1
        elif np.all(l == l[0]) or np.all(r == r[0]):
            similarity = 0
        else:
            similarity = np.corrcoef(periods[i], periods[i+1])[0, 1]
        similarities.append(similarity)
    
    # Calcula la periodicidad como la media de las similitudes entre períodos consecutivos
    periodicity = np.nanmean(np.abs(similarities))

    return periodicity

def get_atributes(s):
    
    return {
        'periodicity': evaluate_periodicity(s, 4),
        'stability': calculate_stability(s),
        'oscilatlion': calculate_oscillation(s),
        'complexity': complexity(s),
        'noise': signaltonoise(s),
        'informative': calculate_entropy(s),
        'variability': calculate_coefficient_of_variation(s),
        'estability': fc.standard_deviation(s),
        'peculiarity': fc.kurtosis(s),
        'dynamic_range': abs(s.max() - s.min()),
        'simetry': abs(fc.skewness(s)),
        'peaks': fc.number_cwt_peaks(s, 10),
        'slope': fc.linear_trend(s, [{'attr': 'slope'}])[0][1],
        'max_value': s.max(),
        'min_value': s.min(),
    }


def classify_distribution(data, alpha):
    """
    Clasifica la distribución de un conjunto de datos utilizando la comparación de las curvas de distribución acumulada (CDF).

    Parameters
    ----------
    data : array_like
        Conjunto de datos a clasificar.

    Returns
    -------
    probabilities : dict
        Diccionario con las probabilidades de que los datos sigan cada una de las distribuciones teóricas proporcionadas.
    """
    distributions = [stats.norm, stats.expon, stats.weibull_max, stats.weibull_min, stats.pareto, 
                     stats.gamma, stats.beta]
    
    # Calcular la CDF de los datos
    data_cdf = np.sort(data)
    data_cdf = np.array(range(len(data_cdf))) / float(len(data_cdf) - 1)

    # Inicializar diccionario de probabilidades
    probabilities = []
    for distribution in distributions:
        # Ajustar distribución teórica a los datos y calcular su CDF
        params = distribution.fit(data)
        theoretical_cdf = distribution.cdf(np.sort(data), *params)

        # Calcular la correlación entre la CDF de los datos y la CDF teórica
        correlation = np.corrcoef(data_cdf, theoretical_cdf)[0, 1]

        # Asignar la correlación como probabilidad de que los datos sigan la distribución teórica
        probabilities.append({'dist': distribution.name,
                             'params': params,
                             'coor': correlation,
                             'confidence_intervals': distribution.interval(alpha, *params),
                             'deciles': [distribution.ppf(q, *params) for q in np.arange(0.1, 1.0, 0.1)]
                             })

    return sorted(probabilities, key=lambda x: -x['coor'])


def modify_signal_slope(s, slope):
    return np.cumsum(np.ones(s.shape)) * slope + s

def modify_signal_correlation(s, r):
    r2 = r
    ve = 1-r2
    std = math.sqrt(ve)
 
    # Genera una señal aleatoria
    e = np.random.normal(0, std, len(s))

    # Modifica la señal aleatoria para que tenga el mismo nivel de correlación de Pearson que s
    ss = r * s + e 
    
    return ss
