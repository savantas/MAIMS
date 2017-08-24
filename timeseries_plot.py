#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Dries Verdegem <dries.verdegem@vib-kuleuven.be>
#
# License: BSD 3 clause

"""
timeseries_plot.py generates a pdf plot of time resolved
isotopologue profile MAIMS deconvolution data.

Copyright (C) 2017 Dries Verdegem and VIB - KU Leuven (BSD-3)

Usage:
  timeseries_plot.py -i <input_file> [-f <font_size>]
                     [-x <legend_x_pos>] [-y <legend_y_pos>]
  timeseries_plot.py (-h | --help)
  timeseries_plot.py --version

Options:
  -h --help           Output usage
  -i <input_file>     Provide the full path to the input contribution file.
                      (mandatory option)
                      The required format is a '####' separated list of times
                      and incorporation percentages.
  -f <font_size>      Determine the font size for the output image
                      [default: 20]
  -x <legend_x_pos>   Determine the x position of the legend.
                      (value between 0 and 1)
                      [default: 0]
  -y <legend_y_pos>   Determine the y position of the legend.
                      (value between 0 and 1)
                      [default: 0]
"""


from __future__ import print_function

__version__ = '1.0.1'
__author__ = "Dries verdegem"
__copyright__ = "Copyright 2017, VIB-KU Leuven (http://www.vib.be/en, http://www.kuleuven.be/english), Dries Verdegem"
__credits__ = ["Dries Verdegem", "Hunter NB Moseley", "Wesley Vermaelen",
               "Abel Acosta Sanchez", "Bart Ghesquière"]
__license__ = "BSD-3"
__maintainer__ = "Dries Verdegem"
__email__ = "dries.verdegem@vib-kuleuven.be"
__status__ = "Production"


import sys, os
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt


def parseopts():
    """
    Parse the command line arguments of the timeseries plot call.
    
    Returns:
        input_file    : path to the input file (Str)
        font_size     : font size (Int)
        legend_x_pos  : x position of the legend (Float)
        legend_y_pos  : y position of the legend (Float)
    """
    
    arguments = docopt(__doc__, version = __version__)
    print(arguments)
    
    # parsing input file name
    input_file = arguments['-i']
    # checking whether file exists
    if not os.path.isfile(input_file):
        print('')
        print('Unexisting file: %s' % input_file)
        print('Please check the -i argument.')
        print('')
        print(__doc__)
        sys.exit(2)

    # parsing the font
    try:
        font_size = int(arguments['-f'])
    except:
        print('')
        print('Invalid argument for the font size: %s' % arguments['-f'])
        print('Value should be integer number')
        print('Please check the -f argument.')
        print('')
        print(__doc__)
        sys.exit(2)
    
    # parsing the x position of the legend
    try:
        legend_x_pos = float(arguments['-x'])
    except:
        print('')
        print('Invalid argument for the x position of the legend: %s' % arguments['-x'])
        print('Value should be floating pointing number')
        print('Please check the -x argument.')
        print('')
        print(__doc__)
        sys.exit(2)

    # parsing the y position of the legend
    try:
        legend_y_pos = float(arguments['-y'])
    except:
        print('')
        print('Invalid argument for the y position of the legend: %s' % arguments['-y'])
        print('Value should be floating pointing number')
        print('Please check the -y argument.')
        print('')
        print(__doc__)
        sys.exit(2)


    return input_file, font_size, legend_x_pos, legend_y_pos


def parse_input_file(input_fn):
    """
    Parse the input file.
    
    Args:
        input_fn - Required  : full path to the input file (Str)
    
    Returns:
        measurements  : the measurement container (Dict)
    """
    
    time = None
    value = None
    measurements = {}
    
    with open(input_fn, 'r') as input_f:
        input_txt = input_f.read()
    
    measurements_txt = input_txt.split('####')
    for m_txt in measurements_txt:
        m_info = m_txt.split()
        for i in range(0,len(m_info),3):
            var = m_info[i]
            value = m_info[i + 2]
            if var == 'time':
                time = float(value)
            else:
                value = float(value[:-1]) / 100.
                if var not in measurements:
                    measurements[var] = {time: [value]}
                else:
                    if time not in measurements[var]:
                        measurements[var][time] = [value]
                    else:
                        measurements[var][time].append(value)
                    
    return measurements

def main():
    """
    main method
    """
    
    # defining colors
    _plot_colors = [('SeaGreen', '#2E8B57'),
                    ('Teal', '#008080'),
                    ('Maroon', '#800000'),
                    ('Crimson', '#DC143C'),
                    ('DarkOrchid', '#9932CC'),
                    ('SteelBlue', '#4682B4'),
                    ('DarkSlateGray', '#2F4F4F'),
                    ('DarkBlue', '#00008B'),
                    ('Green', '#008000'),
                    ('SlateGray', '#708090'),
                    ('SlateBlue', '#6A5ACD'),
                    ('DarkTurquoise', '#00CED1'),
                    ('CornflowerBlue', '#6495ED'),
                    ('BlueViolet', '#8A2BE2'),
                    ('GoldenRod', '#DAA520'),
                    ('LimeGreen', '#32CD32'),
                    ('DarkOliveGreen', '#556B2F')]
    
    # getting the script arguments
    input_fn, font_size, legend_x_pos, legend_y_pos = parseopts()
    
    measurements = parse_input_file(input_fn)
    
    fig, ax = plt.subplots()
    fig.canvas.draw()

    num_components = len(measurements)
    colors = [c for _, c in _plot_colors[:num_components]]
    components = list(measurements.keys())
    components.sort()
    time_points = set([])
    
    for color, component in zip(colors, components):
        component_info = measurements[component]
        data = []
        for time, incorporations in component_info.items():
            mean = np.mean(incorporations)
            std = np.std(incorporations)
            data.append((time, mean, std))
            time_points.add(time)
        data.sort(key = lambda x: x[0])
        time = [d[0] for d in data]
        mean = [d[1] for d in data]
        std_down = [max(0, d[1] - d[2]) for d in data]
        std_up = [min(1, d[1] + d[2]) for d in data]
        ax.plot(time, mean, linewidth=3, color=color, label=component)
        ax.fill_between(time, std_down, std_up, color=color, alpha=0.2)
    
    time_points = list(time_points)
    time_points.sort()
    plt.xlim([0,time_points[-1]])
    plt.xticks(time_points)
    plt.xlabel('time (h)')
    ax.xaxis.label.set_size(font_size)
    
    plt.ylim([0,1])
    new_yticklabels = [str(int(float(l.get_text()) * 100)) for l in ax.get_yticklabels()]
    ax.set_yticklabels(new_yticklabels)
    plt.ylabel('incorporation percentage (%)')
    ax.yaxis.label.set_size(font_size)
    
    plt.tick_params(axis='both', which='major', labelsize=font_size)
        
    ax.legend(loc=2, prop={'size':font_size},
              bbox_to_anchor=(legend_x_pos, legend_y_pos))
    #plt.show()
    plt.savefig('timeseries_plot.pdf', format='pdf')
    
if __name__ == '__main__':
    main()