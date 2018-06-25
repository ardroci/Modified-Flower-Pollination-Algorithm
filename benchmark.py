#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import mpmath as mp
import math
def first_function(data):
    """
    first function
    """
    return abs(data[0] + data[1])

def second_function(data):
    """
    first function
    """
    return data[0]

def rosenbrock_2D(data):
    """
    Rosenbrock’s function in the 2D case.
    Whose global minimum f∗ = 0 occurs at x∗ = (1, 1, ..., 1)
    in the domain −5 ≤ xi ≤ 5 where i = 1, 2, ..., n.
    """
    return (data[0]-1)**2 + 100*(data[1]-data[0]**2)**2

def rosenbrock(data):
    """
    Rosenbrock’s function in the 2D case.
    Whose global minimum f∗ = 0 occurs at x∗ = (1, 1, ..., 1)
    in the domain −5 ≤ xi ≤ 5 where i = 1, 2, ..., n.
    """
    aux = 0
    for i in range(len(data)-1):
        aux += (data[i]-1)**2 + 100*((data[i+1]-(data[i]**2))**2)
    return aux

# TIRAR ESTE CICLO FOR
def rastrigin(data):
    """
    Rastrigin’s function
    Whose global minimum is f∗ = 0 at (0, 0, ..., 0). This function is highly multimodal.
    """
    # aux = 0
    # for i in range(len(data)):
    #     aux += data[i]**2 - 10 * np.cos(2 * np.pi * data[i])
    # res1 = 10 * len(data) + aux
    res2 = 10 * len(data) + np.sum(np.square(data)-10*np.cos(2*np.pi*data))
    # assert res1 == res2
    return res2

def schwefel(data):
    """
    Schwefel’s function
    Whose global minimum f∗ ≈ −418.9829n occurs at xi = 420.9687 where i = 1, 2, ..., n.
    """
    alpha = 418.9828872724339
    # fitness = 0
    # for i in range(len(data)):
    #     fitness -= data[i]*math.sin(math.sqrt(math.fabs(data[i])))
    # return float(fitness) + alpha*len(data)

    # dimensions = len(data)
    # aux = 0
    # for i in range(len(data)):
    #     aux += data[i]*mp.sin(mp.sqrt(mp.fabs(data[i])))
    # return float(alpha*dimensions-aux)

    # data = np.asarray_chkfinite(data)
    # n = len(data)
    # return 418.9829*n - np.sum( data * np.sin( np.sqrt( np.abs( data ))))

    aux1 = alpha*len(data)-np.sum(np.multiply(data,np.sin(np.sqrt(np.abs(data)))))
    # print(aux)
    # assert aux1 == -aux
    return aux1

def easom(data):
    """
    Easom’s function
    Whose global minimum is f∗ = −1 at x∗ = (π, π) within −100 ≤ x, y ≤ 100. It has many
local minima.
    """
    return float(-mp.cos(data[0])*mp.cos(data[1])*mp.exp(-(data[0]-mp.pi)**2 - (data[1]-mp.pi)**2))

def shubert(data, n = 5):
    """
    Shubert’s function
    Which has 18 global minima f ∗ ≈ −186.7309 for n = 5 in the search domain −10 ≤ x, y ≤ 10.
    """
    aux, aux1 = 0, 0
    for i in range(1,n+1,1):
        aux += i * np.cos(i + (i+1)*data[0])
        aux1 += i * np.cos(i + (i+1)*data[1])
    return aux * aux1

def yang(data, dimensions = 16):
    """
    Yang’s forest-like function has a global minimum f∗ = 0 at (0, 0, ..., 0).
    """
    # aux, aux1 = 0, 0
    # for i in range(dimensions):
    #     aux += abs(data[i])
    #     aux1 += mp.sin(data[i]**2)

    # return aux*mp.exp(-aux1)
    return float(np.sum(abs(data)) * mp.exp(-(np.sum(np.sin(np.square(data))))))

def michaelwicz(data):
    aux = 0
    for i in range(0, len(data),1):
        # aux += - np.sum(np.sin(data) * np.sin(j*np.square(data)/np.pi)**(20))
        aux += np.sin(data[i]) * np.sin(i+1*np.square(data[i])/np.pi)**20
    return - aux

def ackley(data, a = 20, b = 0.2, c = 2*np.pi ):
    data = np.asarray_chkfinite(data)  # ValueError if any NaN or Inf
    n = len(data)
    s1 = np.sum(np.square(data))
    s2 = np.sum(np.cos(c*data))
    return -a * np.exp(-b*np.sqrt(s1/n)) - np.exp(s2/n)+a+np.exp(1)


def setup_benchmark_function(args):
    # if args.dimensions is None:
    dimensions = args.dimensions
    # rastrigin
    if args.rastrigin:
        # __function = [benchmark, 'rastrigin']
        benchmark, optimum_solution, domain = rastrigin, 0, [-5.12, 5.12]
    # rosenbrock
    elif args.rosenbrock:
        # __function = [benchmark, 'rosenbrock']
        benchmark, optimum_solution, domain = rosenbrock, 0, [-5,5]
    # easom
    elif args.easom:
        # __function = [benchmark, 'easom']
        benchmark, optimum_solution, domain = easom, -1, [-100,100]
    # shubert
    elif args.shubert:
        # __function = [benchmark, 'shubert']
        benchmark, optimum_solution, domain = shubert, -186.7309, [-10,10]
    # yang
    elif args.yang:
        # __function = [benchmark, 'yang']
        benchmark, optimum_solution, domain = yang, 0, [-2*np.pi,2*np.pi]
    # michaelwicz
    # elif args.michaelwicz:
        __function = [benchmark, 'michaelwicz']
        # benchmark, optimum_solution, domain = michaelwicz, -1.803, [0, np.pi]
    # ackley
    elif args.ackley:
        # __function = [benchmark, 'ackley']
        benchmark, optimum_solution, domain = ackley, 0, [-32.768, 32.768]
    # schwefel
    else:
        # __function = [benchmark, 'schwefel']
        benchmark, optimum_solution, domain = schwefel, 0, [-500,500]
    return dimensions, benchmark, optimum_solution, domain
