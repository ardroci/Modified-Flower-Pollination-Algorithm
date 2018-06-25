import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as FontProperties

import benchmark
import pressure_vessel
import tension_compression
from file_handler import *

def mean_std_iterations(solutions):
    # print(solutions[:,0])
    return np.mean(solutions[:,0]), np.std(solutions[:,0])
def mean_std_error(solutions):
    # print(solutions[:,0])
    return np.mean(solutions[:,-1]), np.std(solutions[:,-1])

# previous_solutions = read_file(filename = 'fpa_yang')
# print('iteracoes', previous_solutions[1][0])
# print('mean best solution', previous_solutions[1][1:])
# print(mean_std_iterations(solutions = previous_solutions))

folder = 'problems'
f = 'problem_E04'
print(f, mean_std_iterations(solutions = read_file(filename = 'results_'+folder+'/fpa_'+f)))
print(f, mean_std_error(solutions = read_file(filename = 'results_'+folder+'/fpa_'+f)))

def plot(data, function, filename = 'fpa_iterations_mean'):
    previous_solutions = read_file(filename = filename)
    mean = previous_solutions[1][1:]
    # fig, (ax1,ax2) = plt.subplots(1,2)
    y=np.zeros((1,1))
    for i in range(len(data)):
        val1 = function(data[i])-function(mean)
        y = np.vstack((y, val1))

    plt.plot(range(len(data)), y[1:], '-')
    plt.grid(1)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    # plt.axis([0,len(y)-1, np.argmin(y,axis=1), np.argmax(y,axis=1)])
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

