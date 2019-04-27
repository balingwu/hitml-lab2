# graddesc.py 用梯度下降法进行求解
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
        description='Using gradient descending method to solve the logistic regression problem.')
    parser.add_argument('--alpha', '-a', type=float, default=0.1,
                        help='Learning rate(the step length). Default is 0.1')
    parser.add_argument('--factor', '-f', type=float,
                        default=0.0, help='Regular factor. Default is 0')
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error('Expect alpha to be positive')
    # load the dataset
    data = np.loadtxt('rawdata.csv', delimiter=',')
    x = data[:, 0:-1]
    y = data[:, -1]
    w, like, k = grad_asc(args.factor, args.alpha)
    print('Optimized result: ' + str(w))
    print('Likelihood: ' + str(like))
    print('Steps: ' + str(k))
    # Drawing
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(x[pos, 0], x[pos, 1], marker='o', color='b', label='Positive')
    plt.scatter(x[neg, 0], x[neg, 1], marker='x', color='r', label='Negative')
    plx = np.linspace(-3, 3)
    ply = -(w[1]/w[2])*plx-w[0]/w[2]
    plt.plot(plx, ply, color='g', label='Result')
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.legend()
    plt.show()


def sigmoid(b):
    den = 1.0 + exp(-1.0*b)
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
        pl = np.dot(xl, w)
        lw += y[l]*pl-log(1+exp(pl))
    return lw - (fac/2)*np.linalg.norm(w)**2


def grad_asc(fac, alpha):
    w = np.zeros(len(x[0])+1)
    grad = grad_func(w, fac)
    like = likelihood(w, fac)
    k = 0
    while not np.all(np.abs(grad) <= 1e-4):
        w += alpha * grad
        if(likelihood(w, fac)) < like:
            # Make alpha smaller if likilihood don't get bigger
            alpha *= 0.5
        grad = grad_func(w, fac)
        # print(grad)
        like = likelihood(w, fac)
        k += 1
    return w, like, k


if __name__ == '__main__':
    main()
