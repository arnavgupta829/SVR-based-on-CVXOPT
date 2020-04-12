import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn import metrics

from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target 
raw = []
for i in data.columns:
    raw.append(data[i])
raw = np.asarray(raw).transpose()

'''
First performing support vector regression using the Sklearn library LIBSVM
and then using a custom optimization package i.e. CVXOPT
'''

#sklearn LIBSVM:
#For each of the kernels used, a graph has been plotted
#for predicted prices vs. actual prices.
#The closer the graph is to the line Y=X,
#the better is our choice of kernel.

from sklearn import svm

#Linear Kernel
clf = svm.SVR(kernel = 'linear')
y_lin = clf.fit(raw[:, 0:13], raw[:, 13:14]).predict(raw[:, 0:13])
plt.scatter(raw[:, 13:14], y_lin, color='darkorange', label='data')

#Gaussian Kernel
clf = svm.SVR(kernel = 'rbf')
y_gaus = clf.fit(raw[:, 0:13], raw[:, 13:14]).predict(raw[:, 0:13])
plt.scatter(raw[:, 13:14], y_gaus, color='darkorange', label='data')

#Polynomial Kernel
clf = svm.SVR(kernel = 'poly', degree = 2)
y_poly = clf.fit(raw[:, 0:13], raw[:, 13:14]).predict(raw[:, 0:13])
plt.scatter(raw[:, 13:14], y_poly, color='darkorange', label='data')

#CVXOPT:

import cvxopt
import cvxopt.solvers

#Defining different types of kernels here
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

#Creating an SVM class
class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None): #if kernel is not mentioned, default is linear, if C is notmentioned, default is C
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
        
        #Here K is the inner product of the feature space (if non linear kernel is used, it is computed using the kernel functions)

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))     #If C is not given => hard margin, G = -1 diagonal vector of size m*m
            h = cvxopt.matrix(np.zeros(n_samples))                  #                                  h = 0 vector of size m*1
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        #P = m*m square matrix, for which, P(i,j) = y(i)y(j)*K
        #q = -1 vector of size m*1
        #A = y (target) of size m*1
        #b = 0 (scalar)
        #If C is given => soft margin
        #G = -1 diagonal vector of size m*m concat with a unity matrix of size m*m
        #h = 0 vector of size m*1 concat with a constant C matrix of size m*1

        '''
            Recall that the dual problem is expressed as
            max{ sum(i = 1,m)alpha(i) - 0.5(sum(i,j = 1,m)y(i)y(j)alpha(i)alpha(j)*K) }
            subject to:
                alpha(i) >= 0
                sum(i = 1,m)alpha(i)y(i) = 0

            writing this with the same notation as mentioned in the CVXOPT notation, we get:
            min{ 0.5(alpha(T)*P*alpha) + q(T)*alpha}
            subject to:
                0 <= alpha(i) <= C
                A*(alpha) = b
        '''

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])
        
        sv = a > 1e-05                                              #Since CVXOPT does not give us exact solutions, a threshold has been kept for identifying the non zero solutions
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        self.b = 0                                                  #calculating the bias term
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        if self.kernel == linear_kernel:                            #calculating the weights
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None                                           #to make the code less complicated, if the kernel is non linear, the weights are calculated directly when the prediction function is called

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:                                                       #else clause executed when the kernel is non linear
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return (self.project(X))

X = raw[:, 0:13]
Y = raw[:, 13:14]
for i in range(0, len(X[0])):
    X[:, i] = (X[:, i] - np.mean(X[:, i]))/np.amax(X[:, i])
meanPrice = np.mean(Y)
maxPrice = np.amax(Y)
Y = (Y-np.mean(Y))/np.amax(Y)

clf = SVM(C=0.01)
clf.fit(X,Y)
y_pred = clf.predict(X)
y_pred = (y_pred*maxPrice)+meanPrice
Y = (Y*maxPrice)+meanPrice
plt.scatter(Y, y_pred, color='darkorange', label='data')
print('R^2:',metrics.r2_score(Y, y_pred))
print('MAE:',metrics.mean_absolute_error(Y, y_pred))
print('MSE:',metrics.mean_squared_error(Y, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y, y_pred)))