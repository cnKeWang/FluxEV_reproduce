import numpy as np

def calculate(X:np.array, i, s, alpha):
    #计算ewma(X[i - s: i - 1])
    ewmaX = 0
    divide = 0
    for j in range(s):
        ewmaX += ((1 - alpha)**j) * X[i-j-1]
        divide += ((1 - alpha)**j)

    return ewmaX / divide
