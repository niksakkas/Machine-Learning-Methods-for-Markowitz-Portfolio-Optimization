import numpy as np
from GSSPf import Plambda
from GSSPf import GSSP

def f(Phi, w, w_ew):
#     x = sym.MatrixSymbol('x', 48, 1)
#     y = np.zeros((1,48))
#     print("x shape:", x.shape)
#     print("y shape:", y.shape)
    fx = np.dot(np.transpose(np.dot(Phi, w) - w_ew), (np.dot(Phi, w) - w_ew))
    return fx

def g(Phi, w, w_ew):
#     x = sym.MatrixSymbol('x', 48, 1)
    gx = np.dot(np.transpose(Phi), np.dot(Phi,w) - w_ew)
    return gx

def p(w, k):
    px = GSSP(w,1,k)
    return px
