#!/usr/bin/env python
# -*- coding: utf-8 -*-
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from math import gamma, log, exp
import math, sys
"""
E02: Pressure Vessel design optimization problem

A compressed air storage tank with a working pressure of 3,000 psi and a
minimum volume of 750 ft3.
A cylindrical vessel is capped at both ends by hemispherical heads.
Using rolled steel plate, the shell is made in two halves that are joined by
teo longitudinal welds to form a cylinder.
The objective is minimize the total cost, including the cost of the materials
forming the welding.
The design variables are: thickness x1, thickness of the head x2, the inner
radius x3, and the length of the cylindrical section of the vessel x4.
The variables x1 and x2 are discrete val ues which are integer multiples of 0.0625 inch.

Then, the formal statement is:
Minimize:
    f (x) =0.6224*x1*x3*x4 + 1.7781*x2*x3^2 + 3.1661*x1^2*x4 + 19.84*x1^2*x3
subject to:
    g1 (x) = −x1 + 0.0193r ≤ 0
    g2 (x) = −x2 + 0.00954r ≤ 0
    g3 (x) = −πr^2L − (4*π)/3*r^3 + 1296000 ≤ 0
    g4 (x) = L − 240 ≤ 0.

with 0.0625 ≤ x1 , x2 ≤ 99 × 0.0625, 10.0 ≤ x3, and x4 ≤ 200.0.
"""

                  # 0   1 2 3
def problem_E02(data):# d1 d2 r L
    """
    The optimum value of the objective function v∗ = f ∗ = 5807.390 is known and the values of the design variables x∗ i are as follow:
    x∗ 1 = 0.7277, x∗ 2 = 0.3597, x∗ 3 = 37.70, x∗ 4 = 240.00.
    """
    return 0.6224*data[0]*data[2]*data[3]+1.7781*data[1]*(data[2]**2)+3.1661*(data[0]**2)*data[3]+19.84*(data[0]**2)*data[2]

def validate_solution_domain_E02(solution):
    if(solution[0]>=0.0625 and solution[0]<=99*0.0625):
        if(solution[1]>=0.0625 and solution[1]<=99*0.0625):
            if(solution[2]>=10 and solution[3]<=200):
            # if(solution[2]>=10):
                if(solution[3]>=10 and solution[3]<=200):
                # if(solution[3]<=200):
                    return True
    return False

def validate_problem_solution_E02(solution):
    # g1 (x) = −d1 + 0.0193r ≤ 0
    if(-solution[0] + 0.0193*solution[2])<=0:
        # g2 (x) = −d2 + 0.00954r ≤ 0
        if(-solution[1] + 0.00954*solution[2])<=0:
            # g3 (x) = −πr^2L − (4*π)/3*r^3 + 1296000 ≤ 0
            # if(-np.pi*(solution[2]**2)*(solution[3]**2) - ((4*np.pi)/3) * solution[2]**3 + 1296000)<=0:
            if(-np.pi*(solution[2]**2)*(solution[3]) - ((4*np.pi)/3) * solution[2]**3 + 1296000)<=0:
                # g4 (x) = L − 240 ≤ 0.
                if(solution[3]-240)<=0:
                    return True
    return False



def init_problem_population_E02(size = 10, population = None,
                                d_lower_bound = 0.0625, d_upper_bound = 99*0.0625,
                                r_lower_bound = 10, r_upper_bound = 70,
                                L_lower_bound = 0, L_upper_bound = 200):
    """
    Initialize population
    :param lower_bound:
    :param upper_bound:
    """
    d_population = np.random.uniform(low = d_lower_bound, high = d_upper_bound, size = (size, 2))
    r_population = np.random.uniform(low = r_lower_bound, high = r_upper_bound, size = (size, 1))
    L_population = np.random.uniform(low = L_lower_bound, high = L_upper_bound, size = (size, 1))
    new_population = np.concatenate((d_population, np.concatenate((r_population, L_population), axis = 1)), axis=1)
    # return new_population
    # repr(new_population, text ='new_population')
    # print(np.apply_along_axis(valide_solution_problem, 1, population))
    # https://stackoverflow.com/questions/23911875/select-certain-rows-condition-met-but-only-some-columns-in-python-numpy
    aux =  new_population[np.apply_along_axis(validate_problem_solution_E02, 1, new_population) == True]
    # repr(aux, text ='aux')
    if population is None:
        population = aux
    else:
        population = np.vstack((population, aux))

    if len(population[:,1]) >= size:
        # repr(population, text ='population')
        return population[:size]
    else:
        return init_problem_population_E02(size, population,
                            d_lower_bound, d_upper_bound,
                            r_lower_bound, r_upper_bound,
                            L_lower_bound, L_upper_bound)

