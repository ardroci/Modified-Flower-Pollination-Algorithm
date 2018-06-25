import numpy as np
import sys

def read_file(filename = 'fpa_iterations_mean', dtype = 'f8', delimiter = ' '):
    try:
        return np.loadtxt(filename + '.dat', dtype = dtype, delimiter = delimiter)
    except OSError as error:
        print('File not found!!')
        sys.exit(2)

def write_to_file(data, filename = 'fpa_iterations_mean', delimiter = ' ', append = True):
    if append:
        try:
            f = open(filename + '.dat', 'ab')
            np.savetxt(f, data[np.newaxis], delimiter =  delimiter, fmt = '%.8f', newline='\n')
            f.close()
        except OSError as error:
            print('File not found!!')
            np.savetxt(f, data, delimiter =  delimiter, fmt = '%.8f', newline='\n')
    else:
            np.savetxt(filename + '.dat', data, delimiter =  delimiter, fmt = '%.8f', newline='\n')

