# rndgen2.py 破坏了独立性假设的随机数生成器
import argparse
import csv

import numpy.random as npr


def main():
    parser = argparse.ArgumentParser(
        description='Generate random (X1,X2,Y) tuples for Logistic Regression.')
    parser.add_argument('-n', type=int, default=20,
                        help='The number of random tuples. Default is 20.')
    parser.add_argument('-p', type=float, default=0.5,
                        help='The probability of Y=1. Default is 0.5.')
    args = parser.parse_args()
    if args.n <= 0:
        parser.error('Expect n to be a positive integer')
    # generate random number pairs
    n1 = round(args.n*args.p)
    x1 = [npr.normal(loc=1.0, scale=1.0) for n in range(n1)]
    n2 = args.n-n1
    x2 = [npr.normal(loc=-1.0, scale=1.0) for n in range(n2)]
    # store the data to a file
    with open('rawdata.csv', 'w', newline='') as csvfile:
        cwrite = csv.writer(csvfile)
        for xi in x1:
            cwrite.writerow([str(xi), str(xi*xi), "1"])
        for xj in x2:
            cwrite.writerow([str(xj), str(2*xj), "0"])


if __name__ == '__main__':
    main()
