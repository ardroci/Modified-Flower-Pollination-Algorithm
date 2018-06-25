#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import argparse
import sys
import logging
from scipy.stats import norm


from benchmark import *
from pressure_vessel import *
from tension_compression import *
from fpa import *
from plots import *
from file_handler import *
from inertia_weight import *


__author__ = "Ricardo Oliveira & Sérgio Silva"
__copyright__ = "Ricardo Oliveira & Sérgio Silva"
__license__ = "mit"

_logger = logging.getLogger(__name__)

class text_colors:
    MAGENTA = '\033[95m'
    BLUE= '\033[94m'
    ORANGE= '\033[33m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'

def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Flower Pollination Algorithm",
        epilog="usage example:\npython skeleton.py -y -p 50 -i 3000 -dsp --d 16"
    )
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO
    )
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG
    )
    parser.add_argument(
        '-p',
        '--population_size',
        dest = "pop_size",
        help = "Population size.",
        type = int,
        metavar = "INT",
        default = 25
    )
    parser.add_argument(
        '-i',
        '--iterations',
        dest = "max_iterations",
        help = "Number of maximum iterations.",
        type = int,
        metavar = "INT",
        default = 1000
    )
    parser.add_argument(
        '-dsp',
        '--dsp',
        dest = "dynamic_switch_probability",
        help = "Use dynamic swicth probability.",
        action ='store_true',
        default = False
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        # '-ciw',
        '--constant_inertia_weight',
        dest = "constant_inertia_weight",
        help = "Constant Inertia Weigth.",
        action = 'store_true',
        default = False
    )
    group.add_argument(
        '-riw',
        '--random_inertia_weight',
        dest = "random_inertia_weight",
        help = "Inertia Weight.",
        action = 'store_true',
        default = False
    )
    group.add_argument(
        '-criw',
        '--chaotic_random_inertia_weight',
        dest = "chaotic_random_inertia_weight",
        help = "Chaotic Random Inertia Weight.",
        action = 'store_true',
        default = False
    )
    group.add_argument(
        '-ciw',
        '--chaotic_inertia_weight',
        dest = "chaotic_inertia_weight",
        help = "Chaotic Random Inertia Weight.",
        action = 'store_true',
        default = False
    )
    parser.add_argument(
        '-d',
        '--dimensions',
        dest = "dimensions",
        help = "Number of dimensions.",
        type = int,
        metavar = "INT",
        default = 2
    )
    parser.add_argument(
        '-geols',
        '--geols',
        dest = "geols",
        help = "Global elite opposition-based learning strategy",
        action= "store_true",
        default = False
    )
    group_2 = parser.add_mutually_exclusive_group()
    group_2.add_argument(
        '-ra',
        '--rastrigin',
        dest="rastrigin",
        help="set benchmark to rastrigin function",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-ro',
        '--rosenbrock',
        dest="rosenbrock",
        help="set benchmark to rosenbrock function",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-e',
        '--easom',
        dest="easom",
        help="set benchmark to easom function",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-y',
        '--yang',
        dest="yang",
        help="set benchmark to yang's forest-like function.",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-sh',
        '--shubert',
        dest="shubert",
        help="set benchmark to shubert's function.",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-sc',
        '--schwefel',
        dest="schwefel",
        help="set benchmark to schwefel's function.",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-ac',
        '--ackely',
        dest="ackley",
        help="set benchmark to ackley's function.",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-pv',
        '--pvessel',
        dest="pressure_vessel",
        help="set to find best solution for pressure vessel problem",
        action='store_true',
        default=False
    )
    group_2.add_argument(
        '-tc',
        '--tension',
        dest="tension_compression",
        help="set to find the best solution for tension/compression problem",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-w',
        '--write',
        dest="write_to_file",
        help="Write solutions to file.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-plt',
        '--plot',
        dest="plot",
        help="Create plot with the solutions fitness.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        '-pa',
        '--plot_animated',
        dest="animated_plot",
        help="Create animated plot with the solutions fitness.",
        action='store_true',
        default=False
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")

    mp.dps = 300
    flag = False
    # Define a switch probability p ∈ [0, 1]
    # p = 0.8 works better for most applications. ~> PAPER
    p = 0.8
    # Define the inertia weight
    w = 1

    __error = 10**-4

    if args.max_iterations > 0:
        __max_iterations = args.max_iterations
    else:
        __max_iterations = 1000

    # População tem de ser >=2 para conseguir usar o local pollination
    if args.pop_size >= 2:
        sys.stdout.write("\r\033[33m%s\033[0m\033[K" % ('Changed Pop Size'))
        sys.stdout.flush()
        __pop_size = args.pop_size
    else:
        __pop_size = 50

    ############################################################################
    # Initialize a population of n flowers/pollen gametes with random solutions
    ############################################################################
    # PRESSURE VESSEL - E02
    if args.pressure_vessel:
        dimensions = 4
        __population = init_problem_population_E02(size = __pop_size)
        __benchmark, __optimum_solution = problem_E02, 6059.714 #5885.332 #5807.390 #6059.714
    # TENSION PRESSURE - E04
    elif args.tension_compression:
        dimensions = 3
        __benchmark, __optimum_solution = problem_E04, 0.012665
        __population = init_problem_population_E04(size = __pop_size)
    # BENCHMARK FUNCTIONS
    else:
        dimensions, __benchmark, __optimum_solution, __domain = setup_benchmark_function(args)

        # __population = init_population(size = __pop_size, variables = dimensions, lower_bound = __domain[0], upper_bound = __domain[1])
        __population = init_population_chaotic_mapping(size = __pop_size, variables = dimensions, lower_bound = __domain[0], upper_bound = __domain[1])
        # print(__population)
        # sys.exit(2)

    ############################################################################
    # Find the best solution g∗ in the initial population
    ##########################################################################
    __fitness , __best_solution = find_best_solution(population = __population, fitness_function = __benchmark)
    ##########################################################################

    solutions = __best_solution

    # g1 = __benchmark(__best_solution)
    # inertia_weights = np.array([1.,8.])

    sys.stdout.write("\r\033[33m%s\033[0m\033[K" % (__benchmark.__name__))
    sys.stdout.flush()

    for iteration in range(__max_iterations):
        for i in range(0,__pop_size):
            #########################
            # DYNAMYC SWITCH PROBABILITY
            if args.dynamic_switch_probability:
                p = dynamic_switch_probability(p=p, iteration = iteration, max_iter = __max_iterations)
                # p = exponential_decay(iteration = iteration)
            # DYNAMYC SWITCH PROBABILITY
            #########################

            if draw_random_uniform() < p:
            #########################
            # GLOBAL POLLINATION 
                #########################
                # LEVY FLIGHT
                # Draw a (d-dimensional) step vector L which obeys a Lévy distribution
                __levy = levy(dimensions = dimensions)
                # LEVY FLIGHT
                #########################

                #########################
                # INERTIA WEIHT
                # w = paper_inertia_weight(iteration = iteration, g1 = g1, g = __benchmark(__best_solution), w = inertia_weights)
                # inertia_weights = np.append(inertia_weights, w)
                if args.chaotic_inertia_weight:
                    w = chaotic_inertia_weight(max_iterations = __max_iterations, iteration = iteration)
                elif args.chaotic_random_inertia_weight:
                    w = chaotic_random_inertia_weight()
                elif args.random_inertia_weight:
                    w = random_inertia_weight()
                elif args.constant_inertia_weight:
                    w = constant_inertia_weight()
                # INERTIA WEIHT
                #########################

                # Global pollination via (x^t+1)_i = (x^t)_i + L(g∗ − (x^t)_i )
                __levy = __levy * w
                # __levy = np.multiply(__levy, w)
                aux_sub = np.subtract(__best_solution,  __population[i])
                aux_mult = np.matmul(__levy, aux_sub)
                __pollination = np.add(__population[i], aux_mult)

                temp_aux = elite_opposition_based_solution(__population,
                                                           __best_solution)
                #########################
                # Global elite opposition-based learning strategy
                if args.geols:
                    if __benchmark(temp_aux) < __benchmark(__pollination):
                        __pollination = temp_aux
                # Global elite opposition-based learning strategy
                #########################
            # GLOBAL POLLINATION 
            #########################

            else:

            #########################
            # LOCAL POLLINATION
                epsilon = draw_random_uniform(low = -1)
                # Randomly choose j and k among all the solutions
                __chosen_solution = __population[np.random.choice(__population.shape[0], 2, replace = False)]
               # Do local pollination via (x^(t+1))_i = (x^t)_i + epsilon((x^t)_j − (x^t)_k)
                aux_sub = np.subtract(__chosen_solution[0], __chosen_solution[1])
                __pollination = np.add(__population[i], epsilon * aux_sub)
            # LOCAL POLLINATION
            #########################

            # PRESSURE VESSEL 
            if args.pressure_vessel:
                if  validate_problem_solution_E02(__pollination):
                    if validate_solution_domain_E02(__pollination):
                        flag = True
            # TENSION PRESSURE
            elif args.tension_compression:
                if  validate_problem_solution_E04(__pollination):
                    if validate_solution_domain_E04(__pollination):
                        flag = True
            # BENCHMARK
            else:
                if valide_solution(__pollination, lower_bound = __domain[0], upper_bound = __domain[1]):
                    flag = True
            if flag:
                # Evaluate new solutions
                __new_fitness = __benchmark(__pollination)
                # If new solutions are better, update them in the population
                if __new_fitness < __fitness[i]: # PARA PROBLEMAS DE MINIMIZAÇÃO
                    __population[i] = __pollination
                flag = False

        __fitness , __best_solution = find_best_solution(population = __population, fitness_function = __benchmark)
        # solutions = np.hstack((solutions, __benchmark(__best_solution)))
        solutions = np.vstack((solutions, __best_solution))

        #########################
        # STOP CRITERIA
        y = abs(__optimum_solution-__benchmark(__best_solution))
        # if  y <= __error:
        if  __benchmark(__best_solution) <= __optimum_solution:
            sys.stdout.write("\r\033[33m%s\033[0m\033[K" % (str(iteration+1)+' ~> '+str(y)))
            sys.stdout.flush()
            break
        # STOP CRITERIA
        #########################

        sys.stdout.write("\r\033[33m%s\033[0m\033[K" % (str(iteration)+' '+str(__best_solution)))
        sys.stdout.flush()
    sys.stdout.write("\r\033[33m%s\033[0m\033[K" % (str(iteration)+' '+str(__best_solution)))
    sys.stdout.flush()

    # PRESSURE VESSEL 
    if args.pressure_vessel:
        aux = __population[np.apply_along_axis(validate_problem_solution_E02, 1, __population) == True]
        __fitness , __best_solution = find_best_solution(population = aux, fitness_function = __benchmark)
        _logger.info('{0}{1}{2}'.format(text_colors.ORANGE, problem_E02(__best_solution), text_colors.ENDC))
    # TENSION PRESSURE
    elif args.tension_compression:
        aux = __population[np.apply_along_axis(validate_problem_solution_E04, 1, __population) == True]
        __fitness , __best_solution = find_best_solution(population = aux, fitness_function = __benchmark)
        _logger.info('{0}{1}{2}'.format(text_colors.ORANGE, problem_E04(__best_solution), text_colors.ENDC))
    # BENCHMARK
    else:
        # print(__best_solution)
        _logger.info('{0}{1}{2}'.format(text_colors.ORANGE, __benchmark(__best_solution), text_colors.ENDC))

    #########################
    # WRITE TO FILE
    #########################
    if args.write_to_file:
        # iterations taken
        a = np.array([iteration+1])
        # mean solutions
        b = np.mean(solutions,axis = 0)
        c = np.hstack((a,b))
        # error
        d = np.hstack((c,y))
        write_to_file(data = d, filename = 'fpa_' + __benchmark.__name__, delimiter = ' ', append = True)

    #########################
    # PLOTS
    #########################
    if args.plot:
        plot(data = solutions, filename = 'fpa_' +  __benchmark.__name__, function = __benchmark)
    if args.animated_plot:
        plot_animated(data = solutions, filename = 'fpa_' + __benchmark.__name__, function = __benchmark, save = False)

    # MEANS & STD
    # print(mean_std_iterations(solutions = read_file(filename = 'fpa_rosenbrock')))
    # print(mean_std_error(solutions = read_file(filename = 'fpa_rosenbrock')))

def run():
    """
    Entry point for console_scripts
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    run()

