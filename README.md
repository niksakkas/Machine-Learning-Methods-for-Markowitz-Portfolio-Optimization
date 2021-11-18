# Machine Learning Methods for Markowitz Portfolio Optimization

## About

This Project was completed in Athens, 2021. It has been written to fulfill the graduation requirements of the Department of Informatics and Telecommunications undergraduate
program. I was engaged in researching and writing it from December 2019 to November 2021. It is mainly addressed to Machine Learning and Stock Investing enthusiasts. I hope
you will find its contents interesting and useful.

## Abstract

In this thesis, we consider the problem of Markowitz Portfolio Optimization. It is defined as attempting to minimize the variance of a diversified investment’s returns. We use several conventional Machine Learning techniques to solve it, namely CVXpy, CVXpy-layers, Proximal and Projected Gradient Descent. We also propose a Deep Learning approach, which uses an LSTM unit. As investment units to train our models, we use the historic returns of 48 industry sector portfolios from 2019 to 2021(FF48 daily returns ). Four of our models including our Deep Learning approach manage to surpass the performance of the equally weighted portfolio which is considered a tough benchmark in this problem. Finally, we propose modifications for further improvements.

## Contents

The thesis is written on the Machine_Learning_Methods_for_Portfolio_Optimization.pdf file. It goes over theory, literature review and all our methodologies and models created. The rest of the files contain the code used to create and train our models:

 ###📄 48_Industry_Portfolios_Daily.csv 
 
 Contains our Dataset

 ### :gear: Functs.ipynb
 
 Contains the basic functions of our problem.
 
 ### :gear: Proximal.ipynb
 Contains the Proximal Gradient Descent algorithms of our LASSO approach.
 
 ### :gear: GSSP.ipynb and PGMB.ipynb
 Contain the Projected Gradient Descent algorithms of the sparse-convex approach.

 ### :gear: LSTM.ipynb
 
 Creates the LSTM_Predictions.csv file. This file contains the predicted future values of our features. Each window of future values is calculated using an LSTM unit.

 ### :gear: ComparePortfolios.ipynb
 
 This file executes and compares all our models using the rest of the .ipynb and .csv files. It also contains the CVXpy and CVXpylayers algorithms.
