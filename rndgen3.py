# rndgen3.py 破坏了方差与分类无关的假定
import argparse
import csv
from random import gauss


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
    x1 = [gauss(2.0, 2.0) for n in range(n1)]
    y1 = [gauss(3.0, 3.0) for n in range(n1)]
    n2 = args.n-n1
    x2 = [gauss(-1.0, 1.0) for n in range(n2)]
    y2 = [gauss(1.0, 2.0) for n in range(n2)]
    # store the data to a file
    with open('rawdata.csv', 'w', newline='') as csvfile:
        cwrite = csv.writer(csvfile)
        for xi, yi in zip(x1, y1):
            cwrite.writerow([str(xi), str(yi), "1"])
        for xj, yj in zip(x2, y2):
            cwrite.writerow([str(xj), str(yj), "0"])


if __name__ == '__main__':
    main()
