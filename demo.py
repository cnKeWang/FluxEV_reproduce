from plotly.subplots import make_subplots
import plotly.graph_objects as go
from data_preprocess import dataloader
import numpy as np
from extandsmooth.fluctuation_extraction import ExtAndSmooth


def firstdemo(X):
    length = len(X)
    #提取和平滑
    E, f, m, S = ExtAndSmooth(X, s=10, p=5, d=2, l=288)

    #画图
    fig = make_subplots(rows=5, cols=1,
                        shared_xaxes=True)
    fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=X,
        mode='lines',
        name='原始数据', marker={"line": {"color": "blue"}}),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=E,
        mode='lines',
        name='(a)', marker={"line": {"color": "red"}}),
        row=2, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=f,
        mode='lines',
        name='(b)', marker={"line": {"color": "green"}}),
        row=3, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=m,
        mode='lines',
        name='(c)', marker={"line": {"color": "black"}}),
        row=4, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=S,
        mode='lines',
        name='(d)', marker={"line": {"color": "yellow"}}),
        row=5, col=1)
    fig.show()


#数据路径
data_path = "../dataset/AIOps2018/decomposed/1th_ts_train.csv"
#读取数据
value = dataloader(data_path)

firstdemo(value)