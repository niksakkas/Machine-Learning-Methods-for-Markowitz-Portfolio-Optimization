import numpy as np
from Functs import f
from Functs import g
from Functs import p


def PGMB(x0,s,alpha,beta,epsilon, max_iter, Phi, w_ew, k):
    x = x0
    grad = g(Phi,x,w_ew)
    it = 0
    while np.linalg.norm(grad) > epsilon and it < max_iter:
        fun_val = f(Phi,x,w_ew)
        it+=1
        t=s
        while (fun_val - f(Phi,x-t*grad,w_ew)) < alpha*t*(np.linalg.norm(grad))**2:
            t *= beta
        grad = g(Phi,x,w_ew)
        x = p(x-t*grad, k)
#         print("iter number = ", it, "norm of grad = ", np.linalg.norm(grad),"X = ",x[1])
    return x
