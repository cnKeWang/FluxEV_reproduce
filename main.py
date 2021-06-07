import extandsmooth.fluctuation_extraction as fe
from POT import pot, poty
from extandsmooth import smooth
import numpy as np
from data_preprocess import dataloader
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# 该函数参考FluxEV算法3，加的一些list以及返回值是为了可视化
def SD_FluxEV(X, p, s, d, l, k, risk):
    """

    :param X: 输入经过预处理的数据
    :param p: p period
    :param s: window sizes
    :param d: half of the window sizes to handle data drift
    :param l: period length
    :param k: window size
    :param risk: risk coefficient
    :return: 主要的是r，若r[i]为1，则该点异常；S为平滑后的数据
    """

    n = len(X)
    list1 = []
    list2 = []
    thf_list = [None for _ in range(n)]
    t_list = [None for _ in range(n)]
    a = 2 * s + d + l * (p - 1)
    E, f, m, S = fe.ExtAndSmooth(X, s, p, d, l)

    r = [None for _ in range(n)]
    # 先用S算初始阈值
    thf, t, y = pot.pot(np.array((S[a + 1: a + k + 1])), risk = risk, init_level=0.98)
    list1.append(thf); list2.append(t)
    y = list(y)
    nt = len(y)
    # 对每个i > k,更新阈值
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
            thf = poty.poty(np.array(y), risk=risk, num_candidates=nt, t=t)

        else:
            k += 1

    # 如果是异常点，该点的特征舍去
        if r[i] == 1:
            f[i] = 0
            m[i-d] = max(f[i - 2 * d: i])
        list1.append(thf)

    n1 = len(list1)
    n2 = len(list2)
    thf_list[n-n1 : n] = list1[0: n1]
    t_list[n-n2: n] = list2[0: n2]
    return r, thf_list, t_list, S



# 参数如下
s = 10; p = 5; d = 2; l = 288; k = 120
# a值就是前面用来计算初始 t, thf 数据个数
a = 2 * s + d + l * (p - 1) + k + 1

if __name__ == "__main__" :
    data_path = "E:/项目组/STL分解数据/separate_train_data/1th_ts_train.csv"
    value, label = dataloader(data_path, a, l)
    r, thf_list, t_list, S = SD_FluxEV(X=value, s=s, p=p, d=d, l=l, k=k, risk=1e-4)


# 下面是计算precision, recall, F1
# 计算检测出来的异常点占总的异常点的比例
length = len(r)
r1 = [None for _ in range(length)]
r1[0:length] = r[0:length]

r1[0:2 * s + d + l * (p - 1)+k+1] = [0 for _ in range(2 * s + d + l * (p - 1) + k+1)]
# 按定义算precision, recall, F1
TP = r1 + label
for i in range(len(TP)):
    if TP[i] != 2:
        TP[i] = 0
TP = TP / 2
precision = sum(TP) / sum(r1)
recall = sum(TP) / sum(label)
print('precision:', precision)
print('recall:', recall)
f1 = 2 * precision * recall / (precision + recall)
print('f1:', f1)

df = pd.DataFrame(value)

df['r'] = r
df['label'] = label

value = df['value'].values
label = df['label'].values
r = df['r'].values

df1 = pd.DataFrame(S)
df1['thf'] = thf_list
df1['t'] = t_list
S = df1[0].values
thf = df1['thf'].values
t = df1['t'].values


# 可视化
fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True)


fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=value,
        mode='lines',
        name='value', marker={"line": {"color": "blue"}}),
        row=1, col=1)


real_ad = np.where(label == 1)
fig.add_trace(go.Scatter(
        x=real_ad[0],
        y=value[real_ad],
        mode='markers',
        name='real_ad', marker={"symbol": "circle", "opacity": 0.5}),
        row=1, col=1)
estimate_segment_ad = np.where(r == 1)
fig.add_trace(go.Scatter(
        x=estimate_segment_ad[0],
        y=value[estimate_segment_ad],
        mode='markers',
        name='estimate', marker={"symbol": "circle","color": "black", "opacity": 0.5}),
        row=1, col=1)

fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=S,
        mode='lines',
        name='r', marker={"line": {"color": "blue"}}),
        row=2, col=1)

fig.add_trace(go.Scatter(
        x=np.arange(0, length),
        y=thf,
        mode='lines',
        name='thf', marker={"line": {"color": "red"}}),
        row=2, col=1)


fig_html = "11th_train_FluxEV.html"
fig.update_layout(title_text='segment threshold:' )

path = 'E:/项目组/FluxEV复现/show'
pio.write_html(fig, file=fig_html)
