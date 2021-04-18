import numpy as np
import pandas as pd

def dataloader(data_path):
    f = pd.read_csv(data_path)

    data = f[:, "origin"]


    return data