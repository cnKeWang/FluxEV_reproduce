import numpy as np
import pandas as pd
import csv

def dataloader(data_path):
    df = pd.read_csv(open(data_path,'r'))
    timestamp = df['timestamp']
    value = df['value']
    label = df['label']
    return value