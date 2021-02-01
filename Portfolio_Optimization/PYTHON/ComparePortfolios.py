import numpy as np
import cvxpy as cp
import sympy as sym
import pandas as pd
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors


from PGMBf import PGMB
# from GSSPf import *

train_data = pd.read_csv(r'data.csv', header=None)
train_data.columns=np.arange(train_data.shape[1])

RR = train_data/100
numportfolios = 4
MonthlyReturns = np.zeros((numportfolios,360));
AnnualReturns = np.zeros((numportfolios,30));
rho = 0.1/12; # `annualised' return of 10%
y = rho*np.ones(60);

for i in range (60,409,12):
    portfolios = np.zeros((numportfolios,train_data.shape[1]));
    Phi = RR[i-60:i]

    #equally weighted portfolio
    portfolios[0] = np.ones(48)/48
    AnnualReturns[0][i//12 - 5] = sum(np.dot(RR[i:i+12],portfolios[0]))
    MonthlyReturns[0][i-60:i-48] = np.dot(RR[i:i+12],portfolios[0])

    #CVX based optimization
    w = cp.Variable(48)
    cost = cp.sum_squares(Phi.to_numpy() @ w - y)
    prob = cp.Problem(
                        cp.Minimize(cost),
                        [cp.sum(w) == 1,
                        w >= 0,]
                        )

    result = prob.solve()
    portfolios[1] = (w.value)
#     print(portfolios[2])
    AnnualReturns[1][i//12 - 5] = sum(np.dot(RR[i:i+12],portfolios[1]))
    MonthlyReturns[1][i-60:i-48] = np.dot(RR[i:i+12],portfolios[1])
    # sparse-convex
    w_ew = np.dot(Phi,np.ones((48,1)))/48
    # k=5
    x0=np.zeros((48,1))
    x0[0] = 1/2
    x0[1] = 1/2
    w = PGMB(x0,0.1,1,0.25,0.001, 200, Phi, w_ew, 5)
    portfolios[2] = w.T
    AnnualReturns[2][i//12 - 5] = sum(np.dot(RR[i:i+12],portfolios[2]))
    MonthlyReturns[2][i-60:i-48] = np.dot(RR[i:i+12],portfolios[2])

    # k=10
    x0=np.zeros((48,1))
    x0[0] = 1/2
    x0[1] = 1/2
    w = PGMB(x0,0.1,1,0.25,0.001, 200, Phi, w_ew, 10)
    portfolios[3] = w.T
    AnnualReturns[3][i//12 - 5] = sum(np.dot(RR[i:i+12],portfolios[3]))
    MonthlyReturns[3][i-60:i-48] = np.dot(RR[i:i+12],portfolios[3])

    # Calculate the cumulative returns
CumulativeReturns = np.zeros(MonthlyReturns.shape);
for j in range(numportfolios):
    CumulativeReturns[j] = np.cumsum(MonthlyReturns[j])
CumulativeReturns

figure(num=None, figsize=(13, 8), dpi=80, facecolor='w', edgecolor='k')
plt.plot(CumulativeReturns[0],color='black')
plt.plot(CumulativeReturns[1],color='red')
plt.plot(CumulativeReturns[2],color='blue')
plt.plot(CumulativeReturns[3],color='green')
plt.margins(0)

legend = ['Equally Weighted', 'Least Squares', 'Sparse k=5', 'Sparse k=10']
plt.legend(legend ,loc = "lower right",prop={'size': 20})
plt.xlabel('Months since July 1976')
plt.ylabel('Value of porfolio')

plt.show()
