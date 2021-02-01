import numpy as np
import cvxpy as cp
import sympy as sym
import pandas as pd
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt


def Plambda(v1, sigma):
    print(np.sum(v1))
    vtemp = np.sort(v1, 0)[::-1]
    inds = np.argsort(v1, 0)[::-1]
    j = 0
    while j < len(vtemp):
        tau = (sum(vtemp[0:j+1]) - sigma)/(j+1)
        if vtemp[j] > tau:
            j = j + 1
        else:
            break
    tau = (sum(vtemp[0:j]) - sigma)/(j)
    vtemp2 = vtemp - tau
    for x in range(len(vtemp2)):
        if vtemp2[x] < 0:
            vtemp2[x] = 0
    v2 = np.zeros(len(v1))
    v2[inds] = vtemp2
    print(np.sum(v2))

    return(v2)


def GSSP(w, l, k):
    # INPUT: vector x and (scalar) parameter lambda and sparsity k
    # OUTPUT: w in simplex.
    N = len(w)
    indices = np.argsort(w, 0)[::-1]
    S = indices[0:k]
    ws = w[S]
    utemp1 = Plambda(w[S], l)
    u = np.zeros((N, 1))
    a = 0
    for x in S:
        u[x] = utemp1[a]
        a += 1
    return(u)
