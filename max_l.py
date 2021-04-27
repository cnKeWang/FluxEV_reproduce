
def max_l(X, l, p):
    maxm = []
    for j in range(0, (p - 1) * l , l):
        k = int(j / l)
        maxm.append(X[k])
    return max(maxm)