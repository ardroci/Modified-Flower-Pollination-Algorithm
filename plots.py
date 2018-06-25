#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import matplotlib.font_manager as FontProperties
import matplotlib.ticker as mtick

import benchmark
import pressure_vessel
import tension_compression
from fpa import *
from file_handler import *

def mean_std_iterations(solutions):
    """
    Calculate mean and standard deviation of iterations
    necessary to solve a given problem.
    """
    return np.mean(solutions[:,0]), np.std(solutions[:,0])

def mean_std_error(solutions):
    """
    Calculate mean and standard deviation of the error
    obtained while solving a given problem.
    """
    return np.mean(solutions[:,-1]), np.std(solutions[:,-1])

def plot_levy_flights(dimensions = 2):
    """
    Use Mantegna Algorithm (Levy Flight) to draw 50 step sizes
    to form a consecutive 50 steps of Lévy Flights.
    :param dimensions: number of dimensions to be used
    """
    levy_flights = np.full(shape=(1,dimensions), fill_value = levy(dimensions = dimensions))
    aux = levy_flights
    for i in range(50):
        aux += levy(dimensions = dimensions)
        levy_flights = np.vstack((levy_flights, aux))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(levy_flights[0,0], levy_flights[0,1], 'r^')
    ax.plot(levy_flights[1:,0], levy_flights[1:,1], 'b-')
    ax.plot(levy_flights[-1,0], levy_flights[-1,1], 'g^')
    # remove axis
    plt.axis('off')
    plt.show()

def plot(data, function, filename = 'fpa_iterations_mean'):
    previous_solutions = read_file(filename = filename)
    mean, std = mean_std_iterations(solutions = previous_solutions)
    f_mean = function([mean])
    print(f_mean)
    f_std = function([std])
    y = function(data[0])-f_mean
    print(len(data))
    for i in range(1,len(data),1):
        val1 = function(data[i])-f_mean
        # val1 = function([data[i]])-function([previous_solutions[0,i]])
        y = np.vstack((y, val1))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(range(len(data)), y, '-')
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(-8,2))# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    # ax.ticklabel_format(style='sci',scilimits=(-8,8),axis='y')
    plt.xlabel('Iterations')
    plt.ylabel('D')
    plt.axis([0, len(data), float(y[np.argmin(y)]-f_std), float(y[np.argmax(y)]+f_std)])
    plt.show()

def plot_animated(data, function, filename = 'fpa_iterations_mean', save = False):
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], '-', animated=True)

    aux = np.apply_along_axis(function, 1, data)

    def init():
        ax.set_xlim(0, len(data), 100)
        ax.set_ylim(aux[np.argmin(aux)]-np.std(aux, axis=0),
                    aux[np.argmax(aux)]+np.std(aux, axis=0))
        return ln,

    def animate(y):
        # line.set_ydata(y)  # update the data
        xdata.append(y)
        # val1 = getattr(function[0], function[1])(data[y])
        val1 = function(data[y])
        ydata.append(val1)
        # ydata.append(y)
        ln.set_data(xdata,ydata) # update the data
        return ln,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,len(data),1), interval = 2, init_func=init, blit=True, repeat = False)
    if save:
        ani.save(filename+'.mp4', fps = 30, extra_args = ['-vcodec', 'libx264'])
    plt.show()

