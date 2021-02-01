import numpy as np
import cvxpy as cp
import sympy as sym
import pandas as pd
from sklearn import preprocessing
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors


train_data = pd.read_csv(r'data.csv', header=None)
train_data.columns=np.arange(train_data.shape[1])

RR = train_data/100
Phi = RR[60:120]
rho = 0.1/12; # `annualised' return of 10%
y = rho*np.ones(60);
x0 = np.zeros((48,1))
x0[0] = 1/2
x0[1] = 1/2
W = np.zeros((3,48))

for j in range(3):
    if j >= 1:
        Phi2 = Phi + 0.01*np.random.randn(60,48)
    else:
        Phi2= Phi
    w = cp.Variable(48)
    cost = cp.sum_squares(Phi2.to_numpy() @ w - y)
    prob = cp.Problem(
                        cp.Minimize(cost),
                        [cp.sum(w) == 1,
                        w >= 0,]
                        )
    result = prob.solve()
    W[j] = w.value

figure(num=None, figsize=(13, 8), dpi=80, facecolor='w', edgecolor='k')

for j in range(3):
    w_current = W[j]
    w_current[w_current == 0] = np.nan
    plt.plot(w_current, 'o')

legend = ['X', 'X_1', 'X_2']
plt.legend(legend ,prop={'size': 14})

plt.xlabel('Stocks')
plt.ylabel('Weights')

plt.show()
