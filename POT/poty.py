import numpy as np

from math import log
from POT import grimshaw
from POT import mom

def poty(data:np.array, risk, num_candidates, t):

    # gamma, sigma = grimshaw.grimshaw(peaks=data,
    #                                  threshold=t,
    #                                  num_candidates=num_candidates)

    gamma, sigma = mom.mom(data)
    # Calculate Threshold
    r = data.size * risk / num_candidates
    if gamma != 0:
        thf = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else:
        thf = t - sigma * log(r)

    return thf