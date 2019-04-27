# newton2.py 牛顿法对实际数据进行分类
import argparse
from math import exp, log

import matplotlib.pyplot as plt
import numpy as np

# data fields
x = None
y = None


def main():
    global x, y
    parser = argparse.ArgumentParser(
        description='Using Newton method to solve the logistic regression problem.')
    parser.add_argument('--factor', '-f', type=float,
                        default=0.0, help='Regular factor. Default is 0')
    args = parser.parse_args()
    # load the dataset
    data = np.loadtxt('glass.csv', delimiter=',')
    x = data[:, 0:-1]
    y = data[:, -1]
    w, like, k = newton(args.factor)
    print('Optimized result: ' + str(w))
    print('Likelihood: ' + str(like))
    print('Steps: ' + str(k))


def sigmoid(b):
    # Truncating big values to avoid overflow
    den = 1.0 + exp(-b)
    return 1.0 / den


def grad_func(w, fac):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        temp = 0
        for l in range(len(y)):
            xl = np.concatenate(([1], x[l]))
            p = sigmoid(np.dot(w, xl))
            temp += xl[i]*(y[l]-p)
        grad[i] = temp
    return grad-fac*w


def likelihood(w, fac):
    lw = 0
    for l in range(len(y)):
        xl = np.concatenate(([1], x[l]))
        pr = np.dot(xl, w)
        lw += y[l]*pr-log(1+exp(pr)) # truncate again
    return lw - (fac/2)*np.linalg.norm(w)**2


def hessian(w, fac):
    n = len(w)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            temp = 0
            for l in range(len(y)):
                xl = np.concatenate(([1], x[l]))
                sim = sigmoid(np.dot(w, xl))
                temp += xl[i]*xl[j]*sim*(1-sim)
            H[i][j] = -temp
    return H - fac * np.eye(n)


def newton(fac):
    w = np.zeros(len(x[0])+1)
    grad = grad_func(w, fac)
    like = likelihood(w, fac)
    hess = hessian(w, fac)
    k = 0
    while not np.all(np.abs(grad) <= 1e-4):
        w -= np.dot(np.linalg.inv(hess), grad)
        grad = grad_func(w, fac)
        like = likelihood(w, fac)
        hess = hessian(w, fac)
        k += 1
    return w, like, k


if __name__ == '__main__':
    main()
