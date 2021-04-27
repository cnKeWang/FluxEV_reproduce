import numpy as np

from math import log
from POT import grimshaw
from POT import mom

def poty(data:np.array, risk:float=3e-3, num_candidates:int=120, epsilon:float=1e-8) -> float:
    t = np.sort(data)[int(0.98 * data.size)]
    gamma, sigma = grimshaw.grimshaw(peaks=data,
                                     threshold=t,
                                     num_candidates=num_candidates)

    # Calculate Threshold
    r = data.size * risk / num_candidates
    if gamma != 0:
        thf = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else:
        thf = t - sigma * log(r)

    return thf, t