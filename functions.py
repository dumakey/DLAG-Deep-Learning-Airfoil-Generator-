from math import sqrt
import numpy as np
import math

class shape_parameters():

    def __call__(self,Theta,b):
        return np.array(Theta * b.reshape(b.shape[0],1))

class class_function():
    '''
    Class that gives airfoil shape to the estimated airfoil z-coordinate distribution
    '''

    def __call__(self,x,a,b,c,d):
        g = list(map(lambda x: a*x**b * (1 - x) + c*x + d*(1 - x),x))
        return np.reshape(np.array(g),(x.shape[0],1))

class shape_functions():
    '''
    Class whose instances are shape functions used to generate the estimation of the airfoil z-coordinate
    When called, it instantiates and generates a matrix of the shape functions particularized at the airfoil streamwise
    domain range (x)
    :param x: domain where to evaluate the basis of polynomials
    :param n: order of the Bernstein polynomials
    '''

    def __call__(self,x,n):

        def Bernstein_poly(x_,n,i):
            '''
            Shape function polynomial: Bernstein polynomials
            :param x: domain of definition
            :param n: order of polynomial
            :return: function handle
            '''
            return lambda x_,n,i: binomial(n,i) * x_**i * (1 - x_)**(n-i)

        def binomial(a,b):
            '''
            Function to compute binomial coefficient
            '''
            try:
                binom = math.factorial(a) // math.factorial(b) // math.factorial(a-b)
            except ValueError:
                binom = 0
            return binom

        def eval_shape_functions(shape_functions, x):

            m = len(shape_functions)
            s = len(x)
            # Function handle evaluation
            P = np.zeros([m,s])
            for i in range(m):
                P[i,:] = shape_functions[i](x,m-1,i)

            return np.matrix(P)

        '''
        def interp(x):
            finterp = np.zeros_like(x)
            for i in range(len(x)):
                for k in range(n + 1):
                    finterp[i] = finterp[i] + math.sin(k / n) * Bernstein_poly(x[i], k)(k, x[i])
    
            return finterp
        '''

        # Function handle generation
        shape_functions = []
        for k in range(n+1):
            shape_functions.append(Bernstein_poly(x,n,k))

        return eval_shape_functions(shape_functions,x)

