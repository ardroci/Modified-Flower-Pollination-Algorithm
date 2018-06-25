#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
from math import gamma, log, exp
import math, sys

from benchmark import *
from pressure_vessel import *
from tension_compression import *

def exponential_decay(iteration):
    # exponential decay
    p = 0.8*mp.exp(-iteration/50)
    if p < 0.5:
        return 0.5
    return p

def dynamic_switch_probability(p, iteration, max_iter = 1000):
    # x = p-((max_iter-iteration)/max_iter)*0.1
    # if x < 0.5:
    #     return 0.5
    # return p-((max_iter-iteration)/max_iter)*0.1 # OLD VERSION
    return 0.6-0.1*(max_iter-iteration)/max_iter

def pop_entropy(data, function):
   max_E = - np.log(1/len(data))
   fitness = np.apply_along_axis(function, 1, data)
   print(fitness)
   min_fitness = data[np.argmin(data)-1]
   print(min_fitness)
   max_fitness = data[np.argmax(data)-1]
   print(max_fitness)
   norm_data = np.subtract(fitness, min_fitness)
   # /np.subtract(max_fitness,min_fitness)
   print(norm_data)

def levy_paper(s, lamb = 1.5, dimensions = 2):
    """
    """
    gamma_dist = gamma(lamb)
    r = (lamb*np.sin(np.pi*lamb/2)/np.pi) * (1/(s**(1+lamb)))
    return np.full(shape = (1,dimensions), fill_value = gamma_dist * float(r))

def levy_paper_modified(lamb = 1.5, dimensions = 2):
    """
    """
    s = levy(dimensions=1)
    r = (lamb*np.sin(np.pi*lamb/2)/np.pi) * (1/(s**(1+lamb)))
    step_vector_L = np.array([gamma(lamb) * float(r)])
    for i in range(1,dimensions,1):
        s = levy(dimensions=1)
        r = (lamb*np.sin(np.pi*lamb/2)/np.pi) * (1/(s**(1+lamb)))
        step_vector_L = np.hstack((step_vector_L, gamma(lamb) * float(r)))
    return step_vector_L

def levy(lamb = 1.5, dimensions = 2):
    """
    Mantegna algorithm.
    It has been proved mathematically that the Mantegna algorithm can produce
    the random samples that obey the required distribution(Levy) correctly.
    :param lamb:
    :param dimensions:
    """
    # calculate the standard deviation used in U
    num = gamma(1+lamb)*np.sin(np.pi*lamb/2)
    den = gamma((1+lamb)/2)*lamb*2**((lamb-1)/2)
    sigma_u = (num/den)**(1/lamb)
    # Gaussian normal distribution U with mean=0 and std=sigma_u
    u = np.random.normal(loc=0.0, scale = sigma_u**2, size = dimensions)
    # Gaussian distribution V with mean=0 and std=1
    v = np.random.normal(loc=0.0, scale = 1, size = dimensions)
    # z = 0.01 * u/(abs(v)**(1/lamb))
    z = u/(np.abs(v)**(1/lamb))
    return z

# def levy_distro(mu):
# 	''' From the Harris Nature paper. '''
# 	# uniform distribution, in range [-0.5pi, 0.5pi]
# 	x = np.random.uniform(-0.5 * math.pi, 0.5 * math.pi)

# 	# y has a unit exponential distribution.
# 	y = -math.log(np.random.uniform(0.0, 1.0))

# 	a = math.sin( (mu - 1.0) * x ) / (math.pow(math.cos(x), (1.0 / (mu - 1.0))))
# 	b = math.pow( (math.cos((2.0 - mu) * x) / y), ((2.0 - mu) / (mu - 1.0)) )

# 	z = a * b
# 	return z

def elite_opposition_based_solution(population, best_solution):
    k = draw_random_uniform()
    elite_individual = best_solution
    da = population.min(axis=0)
    db = population.max(axis=0)
    elite_opposition_individual = k * np.subtract(np.add(da,db),
                                                  elite_individual)
    return elite_opposition_individual

def init_population_chaotic_mapping(size = 10, variables = 2, lower_bound = -2, upper_bound = 2):

    result = np.empty((size, variables), dtype=np.float64)
    for i in range(size):
        z = draw_random_uniform()
        # Make Logistic mapping
        # z = 4 * z * (1-z)
        # Make Sinus mapping
        # z = 2.3*z**(2*np.sin(np.pi*z))
        z = math.sin(math.pi*z)
        for j in range(variables):
            # Make Logistic mapping
            # z = 4 * z * (1-z)
            # Make Sinus mapping
            # z = 2.3*z**(2*np.sin(np.pi*z))
            z = math.sin(math.pi*z)
            result[i,j] = lower_bound+z*(upper_bound-lower_bound)
    return result

    # temp_population = np.random.uniform(low = 0, high = 1, size = (size, variables))
    # return lower_bound + (upper_bound - lower_bound) * temp_population

def init_population(size = 10, variables = 2, lower_bound = -2, upper_bound = 2):
    """
    Initialize population
    :param lower_bound:
    :param upper_bound:
    """
    # print(lower_bound, upper_bound)
    return np.random.uniform(low = lower_bound, high = upper_bound, size = (size, variables))

def repr(data, text):
    print('{}\n{}'.format(text, np.array2string(data)))

def fitness(data, function):
    """
    Fitness calculation
    """
    return np.apply_along_axis(function, 1, data)

def draw_random_uniform(low = 0.0, high = 1.0, size = 1):
    """
    Draw random number from uniform distribution
    :param low:
    :param high:
    :param size:
    """
    return np.random.uniform(low = low, high = high, size = size)

def find_best_solution(population, fitness_function = rastrigin,
                       goal = 'minimization'):
    """
    Calculate population fitness and find the best solution g∗ in the
    population.
    :param population:
    :param fitness_function:
    :return population fitness:
    :return best solution:
    """
    # print('POP',population)
    # calculate population fitness
    _fitness = fitness(data = population, function = fitness_function)
    # get position of the best fitness
    if goal == 'minimization':
        _best_fitness = np.argmin(_fitness)
    else:
        _best_fitness = np.argmax(_fitness)
    # solution with the best fitness
    _best_solution = population[_best_fitness]

    # print('{0:10} {1}'.format('BEST IND.',_best_solution))
    # print('{}\n{}'.format('FITNESS', _fitness))
    return _fitness, _best_solution

def find_best_solution_2(population, fitness_function = rastrigin,
                       goal = 'minimization'):
    """
    Calculate population fitness and find the best solution g∗ in the
    population.
    :param population:
    :param fitness_function:
    :return population fitness:
    :return best solution:
    """
    print('POP',population)
    # calculate population fitness
    _fitness = fitness(data = population, function = fitness_function)
    # get position of the best fitness
    if goal == 'minimization':
        _best_fitness = np.argmin(_fitness)
    else:
        _best_fitness = np.argmax(_fitness)
    # solution with the best fitness
    _best_solution = population[_best_fitness]

    # print('{0:10} {1}'.format('BEST IND.',_best_solution))
    # print('{}\n{}'.format('FITNESS', _fitness))
    return _fitness, _best_solution
# MELHOR ISTO # MELHOR ISTO # MELHOR ISTO
# MELHOR ISTO # MELHOR ISTO # MELHOR ISTO
def valide_solution(solution, lower_bound = -5., upper_bound = 5.):
    """
    Check if a given solution respects the domain boundaries of a given
    problem.
    note: only used in benchmark
    :param solution: solution to be evaluated
    :param lower bound:
    :param upper_bound:
    """
    for i in range(len(solution)):
        if not(solution[i]<=upper_bound and solution[i]>=lower_bound):
            return False
    return True
