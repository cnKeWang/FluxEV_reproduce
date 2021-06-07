import numpy as np 

from math import log
from POT import grimshaw
from POT import mom

def pot(data:np.array, risk,  init_level):
    ''' Peak-over-Threshold Alogrithm

    References: 
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform
    
    Returns:
        z: threshold searching by pot
        t: init threshold 
    '''
    # Set init threshold
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    # Grimshaw
    gamma, sigma = mom.mom(peaks)

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        thf = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else: 
        thf = t - sigma * log(r)

    return thf, t, peaks
    