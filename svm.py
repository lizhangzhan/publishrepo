#! /usr/bin/python
# -*- encoding:utf-8 -*-
import math
import random

"""
__author__ = "zhan lizhang"
__copyright__ = "Copyright (c) 2014 zhan lizhang"
__licence__ = "MIT licence"
__version__ = 1.0
__email__ = "lizhangzhan@outlook.com"

Reference:
    [1] svm note, http://cs229.stanford.edu/notes/cs229-notes3.pdf
    [2] a simple vesion smo, http://cs229.stanford.edu/materials/smo.pdf
    [3] smo paper, http://research.microsoft.com/apps/pubs/default.aspx?id=69644
"""

class SVM(object):
    class option(object):
        """
        It is simple storage struct, some factors used in svm model training.
        kernel      -   a kernel function, usually four kinds of kernel
                        function, linear, rbf, polynomial, sigmoid.Here, just
                        the former two ones are support.
        
        alpha_tol   -   nuberical tolerance of alpha, a condition to stop
                        updating alpha value

        tol         -   numberical error tolerance of each point loss function
                        It is used in checking KKT condition

        max_skip_iter - maximum iteration times without alpha parameters update
                        to stop training in advance.
                        
        max_iter    -   training maximum iteration to stop training

        regular     -   a regular factor to control model generalization, 
                        Smaller it is, more generalization the model has, usually
                        [0.75, 1.0] is better. Default is 1.0
        """
        def __init__(self, kernel, alphaTol, tol, maxSkipPass, maxIter, regular = 1.0):
            self.kernel_fun = kernel
            self.tol = tol
            self.alpha_tol = alphaTol
            self.max_iter = maxIter
            self.max_skip_pass = maxSkipPass
            self.regular = regular

    def __init__(self):
        self.points = []
        self.labels = []
        self.alphas = []
        self.b = 0.0
        self.kernel_fun = None
        self.kernel_pair = []

    def randomParameters(self):
        #self.alphas = [random.random() for i in range(len(self.points))]
        self.alphas = [0.0 for i in range(len(self.points))]
    
    # Karush-Kuhn-Tucker condition
    # alpha(i) = 0      y(i) * f(x_i) > = 1
    # 0< alpha(i) < C   y(i) * f(x_i)   = 1
    # alpha(i) =c       y(i) * f(x_i) < = 1
    def KKTcondition(self, error, label, alpha, opt):
        if error * label < -opt.tol and alpha < opt.regular:
            return True
                
        if error * label > opt.tol and alpha > 0.0:
            return True

        return False

    def computeKernalPair(self):
        self.kernel_pair = []
        for i in range(len(self.points)):
            for j in range(i, len(self.points)):
                self.kernel_pair.append(self.kernel_fun(self.points[i], self.points[j]))

    def getKernelPair(self, i, j, n):
        return self.kernel_pair[i * n + j - i * (i + 1) // 2 ]

    def __setting__(self, points, labels, option):
        assert len(points) == len(labels)

        N = len(labels)
        self.points = points
        self.labels = labels
        self.kernel_fun = option.kernel_fun
        self.regular = option.regular
    
        self.randomParameters()
        self.computeKernalPair()
    """
     Three components in smo algorithm:
     1. An analytic method to solve for the two Lagrange multipliers
     2. A heuristic for choosing which multipliers to optimize
     3. A method for computing bias term, b
    """
    def train(self, points, labels, opt):
        C = opt.regular
        skip_num = 0

        self.__setting__(points, labels, opt)
        alphas = self.alphas
        N = len(alphas)
        for t in range(opt.max_iter):
            changed = False

            # A potential contrained condtion,sum_i(alpha_i * label_i) = 0
            # At every step, just select two Lagrange multipliers to optimize
            for i in range(len(alphas)):
                error_i = self.computeMargin(points[i]) - labels[i]
                if self.KKTcondition(error_i, labels[i], alphas[i], opt):
                    j = i
                    # Randomly choose a alpha parameter j, j != i
                    while j == i: j = random.randint(0, len(alphas)-1)
                    error_j = self.computeMargin(points[j]) - labels[j]
                    # Get the current alpha[i] and alpha[j] value
                    # They might be updated in next.
                    alpha_i = alphas[i]
                    alpha_j = alphas[j]

                    if labels[i] == labels[j]:
                       low = max(0, alpha_i + alpha_j - C)
                       high = min(C, alpha_i + alpha_j)
                    else:
                       low = max(0, alpha_j - alpha_i)
                       high = min(C, C + alpha_j - alpha_i)

                    if abs(high - low) < opt.alpha_tol: continue
                    # The second derivative of the object function along the
                    # diagonal line
                    derivate = 2 * self.getKernelPair(i, j, N) - \
                                self.getKernelPair(i, i, N) - \
                                self.getKernelPair(j, j, N)
                    if derivate >= 0: continue
                    new_alpha_j = alpha_j - labels[j] * \
                                (error_i - error_j) / derivate
                    if new_alpha_j > high: new_alpha_j = high
                    if new_alpha_j < low: new_alpha_j = low

                    if abs(alpha_j - new_alpha_j) < opt.alpha_tol: continue
                    alphas[j] = new_alpha_j
                    new_alpha_i = alpha_i + labels[i] * labels[j] * \
                                    (alpha_j - new_alpha_j)
                    alphas[i] = new_alpha_i
                    
                    # alpha[i] and alpha[j] obey a linear equality constraint
                    assert alpha_i+alpha_j-new_alpha_i-new_alpha_j < opt.tol
                    # update the bias term b
                    b1 = self.b - error_i - \
                        labels[i] * (new_alpha_i - alpha_i) * \
                        self.getKernelPair(i, i, N) - \
                        labels[j] * (new_alpha_j - alpha_j) * \
                        self.getKernelPair(i, j, N)
                    b2 = self.b - error_j - \
                        labels[i] * (new_alpha_i - alpha_i) *\
                        self.getKernelPair(i, j, N) - \
                        labels[j] * (new_alpha_j - alpha_j) * \
                        self.getKernelPair(j, j, N)
                    self.b = 0.5 * (b1 + b2)

                    if new_alpha_i > 0 and new_alpha_i < C: self.b = b1
                    if new_alpha_j > 0 and new_alpha_j < C: self.b = b2
                    
                    changed = True;
                # end if
            # end for alpha update
            if not changed: skip_num = skip_num + 1
            else:           skip_num = 0

            if skip_num >= opt.max_skip_pass: break
        # end for iteration

        # save support vectors and valid lagrange multipliers
        self.alphas = []
        self.points = []
        self.labels = []

        # if the kernel function is a linear
        # self.weight = sum_i(alpha[i] * labels[i] * points[i])
        # here, it is suitable for non-linear kernel function
        for i in range(len(alphas)):
            if alphas[i] > opt.alpha_tol:
                self.alphas.append(alphas[i])
                self.points.append(points[i])
                self.labels.append(labels[i])


    def predict(self, point):
        if self.computeMargin(point) > 0:
            return 1
        else:
            return -1

    def predicts(self, points):
        for point in points:
            yield self.predict(point)

    def computeMargin(self, point):
        return sum([self.alphas[i] * self.labels[i] * \
            self.kernel_fun(point, self.points[i]) for \
            i in range(len(self.points))]) + self.b
    
def linearKernel(x1, x2):
    assert len(x1) == len(x2)
    return sum([x1[i] * x2[i] for i in range(len(x1))])

def RBFKernel(x1, x2, sigma = 0.5):
    assert len(x1) == len(x2)
    v = sum([pow(x1[i] - x2[i], 2) for i in range(len(x1))]) / (-2.0 * sigma * sigma)
    return math.exp(v)

def main():
    tol = 1e-4
    alpha_tol = 1e-5
    max_skip_pass = 10
    max_iter = 1000
    regular = 1.0
    
    # Nonlinear Separation
    points = [[0,0], [1, 0], [0, 1], [1, 1]]
    labels = [-1, 1, 1, -1]
    
    opt = SVM.option(RBFKernel, alpha_tol, tol, max_skip_pass, max_iter, regular)
    #opt = SVM.option(linearKernel, alpha_tol, tol, max_skip_pass, max_iter, regular)

    svm = SVM()
    svm.train(points, labels, opt)
    for i in range(len(points)):
        assert svm.predict(points[i]) == labels[i]
    print "pass"

if __name__ == "__main__":
    main()
