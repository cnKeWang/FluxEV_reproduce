import numpy as np
import pandas as pd
import csv
from delete_anomalies import change

def dataloader(data_path, a, l):
    df = pd.read_csv(open(data_path,'r'))
    df = change(df, a, l)
    timestamp = df['timestamp']
    value = df['value']
    label = df['label']
    return value, label