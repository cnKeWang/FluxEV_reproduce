import extandsmooth.fluctuation_extraction as fe
from POT import pot, poty
from extandsmooth import smooth
import numpy as np
from data_preprocess import dataloader
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def SD_FluxEV(X, p, s, d, l, k, risk):
    risk = 3e-3
    n = len(X)
    a = 2 * s + d + l * (p - 1)
    E, f, m, S = fe.ExtAndSmooth(X, s, p, d, l)

    r = [None for _ in range(n)]
    #先用S算初始阈值
    thf, t, y = pot.pot(np.array((S[a + 1: a + k])), risk = risk, init_level=0.98)
    y = list(y)
    nt = len(y)
    #对每个i > k,更新阈值
    for i in range(a + k + 1, n):
        r[i] = 0
        E[i], f[i], S[i], m[i - d] = smooth.calfeats(X, E, f, m, S, s, i, d, l, p)
        if S[i] > thf:
            r[i] = 1
        elif S[i] > t:
            yi = S[i] - t
            y.append(yi)
            nt +=  1
            k += 1
            thf, t = poty.poty(np.array(y), risk=risk, num_candidates=nt)
        else:
            k += 1
    #如果是异常点，该点的特征舍去
        if r[i] == 1:
            f[i] = 0
            m[i-d] = max(f[i - 2 * d: i - 1])
    return r


data_path = "../dataset/AIOps2018/decomposed/1th_ts_train.csv"
value = dataloader(data_path)
r = SD_FluxEV(X=value, s=10, p=5, d=2, l=288, k=120, risk=3e-3)

fig = make_subplots(rows = 1, cols = 1)
length = len(r)
fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=r,
        mode='lines',
        name='r', marker={"line": {"color": "blue"}}),
        row=1, col=1)
fig.show()
