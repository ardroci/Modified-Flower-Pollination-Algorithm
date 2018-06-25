#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import numpy as np
from fpa import draw_random_uniform

def constant_inertia_weight():
    return 0.7

def random_inertia_weight():
    return 0.5+(draw_random_uniform()/2)

def chaotic_random_inertia_weight(w1 = 0.9, w2 = 0.4):
    #  Select  a  random  number z in  the  interval of (0, 1)
    z = draw_random_uniform()
    # Make Logistic mapping
    z = 4 * z * (1-z)
    return 0.5*draw_random_uniform()+0.5*z

def chaotic_inertia_weight(max_iterations, iteration, w1 = 0.9, w2 = 0.4):
    #  Select  a  random  number z in  the  interval of (0, 1)
    z = draw_random_uniform()
    # Make Logistic mapping
    z = 4 * z * (1-z)
    return (w1-w2) * (max_iterations-iteration)/max_iterations + w2*z

def paper_inertia_weight(iteration, g1, g, w):
    """
    :param iteration: denotes the current iteration
    :param g1: is the best fitness value in the first iteration
    :param g: is the obtained best fitness value in the current iteration
    :param w: inertia weigth matrix matrix
    """
    w_min = w[np.argmin(w)]
    w_max = w[np.argmax(w)]
    # print(w_min, w_max)
    return w_min * (1 + (w_max/math.sqrt(iteration+1)) * math.tan(g/g1))
