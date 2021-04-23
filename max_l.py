
def max_l(X, l, p):
    maxm = []
    for j in range(0, p-1, l):
        maxm.append(X[j])
    return max(maxm)