#!/usr/bin/env python
# -*- coding: utf-8 -*-
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from math import gamma, log, exp
import math, sys
from fpa import init_population_chaotic_mapping
"""
E04: Tension/compression spring design optimization problem

This problem minimizes the weight of a tension/compression spring, subject to constraints of minimum deflection, shear stress, surge frequency, and limits on outside diameter and on design variables.
There are three design variables: the wire diameter x1 , the mean coil diameter x2 , and the number of active coils x3.
The mathematical formulation of this problem is:
Minimize:
        f (x) = (x3 + 2)*x2*x1^2
subject to:
        g1 (x) = 1 − (x2^3*x3)/(7.178*x1^4) ≤ 0
        g2 (x) = (4x2^2 - x1*x2)/(12,566*(x2*x1^3)-x1^4) + 1/(5,108x1^2) ≤ 0
        g3 (x) = 1 − (140.45 * x1)/(x2^2*x3) ≤ 0
        g4 (x) = (x2+x1)/1.5 - 1 ≤ 0

with 0.05 ≤ x1 ≤ 2.0, 0.25 ≤ x2 ≤ 1.3, and 2.0 ≤ x3 ≤ 15.0.

"""

def problem_E04(data):
    """
    The optimum value of the objective function v∗ = f ∗ = XXXXXX is known and the values of the design variables x∗ i are as follow:
    x1 = XXXXX, x2 = XXXXX, x3 = XXXXX
    """
    return (data[2]+2)*data[1]*(data[0]**2)

def validate_solution_domain_E04(solution):
    if(solution[0]>=0.05 and solution[0]<=2.0):
        # print('1')
        if(solution[1]>=0.25 and solution[0]<=1.3):
            # print('2')
            if(solution[2]>=2.0 and solution[2]<=15.0):
                return True
    return False

def validate_problem_solution_E04(solution):
    # g1 (x) = 1 − (x2^3*x3)/(7.178*x1^4) ≤ 0
    if 1-(solution[1]**3*solution[2])/(7178*solution[0]**4) <=0:
        # g2 (x) = (4x2^2 - x1*x2)/(12,566*(x2*x1^3)-x1^4) + 1/(5,108x1^2) -1 ≤ 0
        if ( ( (4*solution[1]**2)-(solution[0]*solution[1]) ) / ((12566*(solution[1]*solution[0]**3)) -solution[0]**4) + (1/(5108*solution[0]**2) -1)) <=0:
            # g3 (x) = 1 − (140.45 * x1)/(x2^2*x3) ≤ 0
            if 1-(140.45*solution[0])/(solution[1]**2*solution[2]) <=0:
                # g4 (x) = (x2+x1)/1.5 - 1 ≤ 0
                if (solution[1]+solution[0])/1.5 -1 <=0:
                    return True
    return False



def init_problem_population_E04(size = 10, population = None,
                                x1_lower_bound = 0.05, x1_upper_bound = 2.0,
                                x2_lower_bound = 0.25, x2_upper_bound = 1.3,
                                x3_lower_bound = 2., x3_upper_bound = 15.):
    """
    Initialize population
    :param lower_bound:
    :param upper_bound:
    """
    x1_population = np.random.uniform(low = x1_lower_bound, high = x1_upper_bound, size = (size, 1))
    # x1_population = init_population_chaotic_mapping(size= size, variables = 1,
    #                                                lower_bound = x1_lower_bound,
    #                                                upper_bound = x1_upper_bound)
    x2_population = np.random.uniform(low = x2_lower_bound, high = x2_upper_bound, size = (size, 1))
    # x2_population = init_population_chaotic_mapping(size= size, variables = 1,
    #                                                lower_bound = x2_lower_bound,
    #                                                upper_bound = x2_upper_bound)
    x3_population = np.random.uniform(low = x3_lower_bound, high = x3_upper_bound, size = (size, 1))
    # x3_population = init_population_chaotic_mapping(size= size, variables = 1,
    #                                                lower_bound = x3_lower_bound,
    #                                                upper_bound = x3_upper_bound)
    new_population = np.concatenate((x1_population, np.concatenate((x2_population, x3_population), axis = 1)), axis=1)

    # print('new_population', new_population)
    # print(np.apply_along_axis(valide_solution_problem, 1, population))
    # return new_population

    # https://stackoverflow.com/questions/23911875/select-certain-rows-condition-met-but-only-some-columns-in-python-numpy
    aux =  new_population[np.apply_along_axis(validate_solution_domain_E04, 1, new_population) == True]
    # # print('new_population', aux)
    if population is None:
        print('aux')
        population = aux
    else:
        population = np.vstack((population, aux))

    if len(population[:,1]) >= size:
        # print('new_population', new_population)
        return population[:size]
    else:
        return init_problem_population_E04(size, population,
                                x1_lower_bound, x1_upper_bound,
                                x2_lower_bound, x2_upper_bound,
                                x3_lower_bound, x3_upper_bound)
