import  matplotlib.pyplot as plt
import data_preprocess as dp
import pandas as pd
data_path = "../dataset/AIOps2018/decomposed/1th_ts_train.csv"
df = pd.read_csv(data_path)
timestamp = df['timestamp']
value = df['value']


plt.plot(timestamp, value)
plt.show()

