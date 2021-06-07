
def max_l(X, l, p):
    maxm = []
    n = len(X)
    for j in range(0, n, l):
        maxm.append(X[j])
    return max(maxm)