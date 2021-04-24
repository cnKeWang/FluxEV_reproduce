from plotly.subplots import make_subplots
import plotly.graph_objects as go
from data_preprocess import dataloader
import numpy as np
from fluctuation_extraction import ExtAndSmooth

data_path = "../dataset/AIOps2018/decomposed/1th_ts_train.csv"
timestamp, value, label = dataloader(data_path)
E, f, S, m = ExtAndSmooth(value, s=10, p=5, d=2, l=288)


def stl_visualization(X):
    length = len(X)
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

stl_visualization(X=value)
