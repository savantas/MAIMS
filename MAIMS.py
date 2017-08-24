#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Dries Verdegem <dries.verdegem@vib-kuleuven.be>
#
# License: BSD 3 clause

"""MAIMS.py deconvolutes the isotopologue profile of large metabolites upon
specific tracer administration into individual contributions of metabolic moieties.

A classical example is the deconvolution of the UDP-GlcNAc isotopologue profile
into glucose, ribose, acetyl and uracil contributions upon [U-13C]-glucose administration.

Copyright (C) 2017 Dries Verdegem and VIB - KU Leuven (BSD-3)

Usage:
  MAIMS.py -f <isotopologue_file> -m <model_file> [-n] [-v] [-s <random_seed>]
           [-c <HCS_max_missing_mass>] [-d <HCS_confidence_level>]
           [-i <min_num_iterations>] [-r <min_ROABM>]
           [--rel <mz_precision_rel>] [--abs <mz_precision_abs>]
           [-o <optimization_method>]
  MAIMS.py (-h | --help)
  MAIMS.py --version

Options:
  -h --help                   Output usage
  --abs <mz_precision_abs>    Specify the absolute mass error in Da of the MS machine.
                              This information is required when performing a natural
                              abundance correction on the isotopologue profile.
                              [default: 0.1] (low resolution)
  -c <HCS_max_missing_mass>   Set the maximum acceptable missing mass for the
                              high-confidence stopping rule (c parameter).
                              C can take floating point values in the interval: [0,1].
                              The deconvolution will stop when the missing mass is below a
                              threshold c with probability at least 1 - delta.
                              Bigger values mean faster results.
                              [default: 0.4]
  -d <HCS_confidence_level>   Set the confidence level for the high-confidence stopping rule (delta parameter).
                              Delta can take floating point values in the interval: [0,1].
                              The deconvolution will stop when the missing mass is below a
                              threshold c with probability at least 1 - delta.
                              Bigger values mean faster results.
                              [default: 0.4]
  -f <isotopologue_file>      Provide the full path to the input metabolite isotopologue profile file.
                              (mandatory option)
                              The required format is a tab separated file in which the first column
                              is a listing of the labels 'm0' up to 'mxx' (e.g. m17 in case of
                              UDP-GlcNAc) and the second column contains the actual experimental
                              values for these signals.
  -i <min_num_iterations>     Set the minimum number of iterations that should be performed
                              during the deconvolution process.
                              [default: 5000]
  -m <model_file>             Provide the full path to the metabolite-moiety model file.
                              (mandatory option)
                              The required format is an xml file containing metabolite,
                              moiety, mass and constraint information. For more detail, check
                              the example UDPGlcNAc.mdl file.
  -n                          Perform a natural abundance correction on the isotopologue profile
  -o <optimization_method>    Specify the local optimization method. The options are:
                              * L-BFGS-B (the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm)
                              * TNC (a truncated Newton algorithm)
                              * SLSQP (Sequential Least SQuares Programming)
                              [default: SLSQP]
  -r <min_ROABM>              Set the minimum required region of attraction (ROA) (i.e. number of times found)
                              of the current best local minimum (BM).
                              [default: 20]
  --rel <mz_precision_rel>    Specify the relative mass error in ppm of the MS machine.
                              This information is required when performing a natural
                              abundance correction on the isotopologue profile.
                              [default: 500] (low resolution)
  -s <random_seed>            Define the random seed value for the random number generation
                              [default: 0]
  -v --verbose                Set when extensive output information is required
  --version                   Output the version of the MAIMS script.
"""


from __future__ import print_function

__version__ = '2.0.1'
__author__ = "Dries verdegem"
__copyright__ = "Copyright 2017, VIB-KU Leuven (http://www.vib.be/en, http://www.kuleuven.be/english), Dries Verdegem"
__credits__ = ["Dries Verdegem", "Hunter NB Moseley", "Wesley Vermaelen",
               "Abel Acosta Sanchez", "Bart Ghesquière"]
__license__ = "BSD-3"
__maintainer__ = "Dries Verdegem"
__email__ = "dries.verdegem@vib-kuleuven.be"
__status__ = "Production"


import sys, os
import re
import copy
import math
import itertools
import time
import scipy.optimize
import numpy
import xml.etree.ElementTree
import operator
import functools
import sympy
from docopt import docopt
from sympy.parsing.sympy_parser import parse_expr

# show top 5 results in verbose mode
num_top_results = 5

# no progress should be shown when output is redirected
show_progress = os.fstat(0) == os.fstat(1) # stdin and stdout are the same


def parseopts():
    """
    Parse the command line arguments of the MAIMS call.
    
    Returns:
        random_seed                    : seed for random generator (Int)
        isotopologue_file              : path to isotopologue file (Str)
        model_file                     : path to model file (Str)
        natural_abundance_correction   : natural abundance correction flag (Bool)
        optimization_method            : optimization method (Str)
        min_num_iterations             : minimum number of iterations (Int)
        min_ROABM                      : minimum region of attraction of best minimum (Int)
        HCS_max_missing_mass           : maximum missing mass for HCS rule (Float)
        HCS_confidence_level           : confidence level for HCS rule (Float)
        mz_precision_rel               : MS relative mass error (Float)
        mz_precision_abs               : MS relative mass error (Float)
        verbose                        : verbose flag (Bool)
    """
    
    arguments = docopt(__doc__, version = __version__)
    
    # parsing random seed
    try:
        random_seed = int(arguments['-s'])
    except:
        print('WARNING: Invalid random seed was provided (%s)' % arguments['-s'])
        print('WARNING: Forcing random seed to 0')
        random_seed = 0
    
    # parsing isotopologue file name
    isotopologue_file = arguments['-f']
    # checking whether file exists
    if not os.path.isfile(isotopologue_file):
        print('')
        print('Unexisting file: %s' % isotopologue_file)
        print('Please check the -f argument.')
        print('')
        print(__doc__)
        sys.exit(2)
    
    # parsing model file name
    model_file = arguments['-m']
    # checking whether file exists
    if not os.path.isfile(model_file):
        print('')
        print('Unexisting file: %s' % model_file)
        print('Please check the -m argument.')
        print('')
        print(__doc__)
        sys.exit(2)
    
    # parsing the maximum missing mass for HCS rule
    correct_value = True
    try:
        HCS_max_missing_mass = float(arguments['-c'])
    except:
        HCS_max_missing_mass = -1
        correct_value = False
    if not (0. <= HCS_max_missing_mass <= 1.):
        correct_value = False
    if not correct_value:
        print('')
        print('Invalid argument for the maximum missing mass: %s' % arguments['-c'])
        print('Value should be floating point number in the interval: [0,1]')
        print('Please check the -c argument.')
        print('')
        print(__doc__)
        sys.exit(2)
    
    # parsing the confidence level for HCS rule
    correct_value = True
    try:
        HCS_confidence_level = float(arguments['-d'])
    except:
        HCS_confidence_level = -1
        correct_value = False
    if not (0. <= HCS_confidence_level <= 1.):
        correct_value = False
    if not correct_value:
        print('')
        print('Invalid argument for the confidence level: %s' % arguments['-d'])
        print('Value should be floating point number in the interval: [0,1]')
        print('Please check the -d argument.')
        print('')
        print(__doc__)
        sys.exit(2)
        
    # parsing the minimum number of iterations
    try:
        min_num_iterations = int(arguments['-i'])
    except:
        print('')
        print('Invalid argument for the minimum number of iterations: %s' % arguments['-i'])
        print('Value should be integer number')
        print('Please check the -i argument.')
        print('')
        print(__doc__)
        sys.exit(2)

    # parse natural abundance correction flag
    natural_abundance_correction = arguments['-n']

    # parse the optimization method
    optimization_method = arguments['-o']
    if optimization_method not in ['L-BFGS-B', 'TNC', 'SLSQP']:
        print('')
        print('Unknown optimization method: %s' % optimization_method)
        print('Please check the -o argument.')
        print('')
        print(__doc__)
        sys.exit(2)
        
    # parse the minimum region of attraction of best minimum
    try:
        min_ROABM = int(arguments['-r'])
    except:
        print('')
        print('Invalid argument for the minimum region of attraction: %s' % arguments['-r'])
        print('Value should be integer number')
        print('Please check the -r argument.')
        print('')
        print(__doc__)
        sys.exit(2)
    
    # parse the MS relative mass error
    try:
        mz_precision_rel = float(arguments['--rel'])
    except:
        print('')
        print('Invalid argument for the MS relative mass error: %s' % arguments['--rel'])
        print('Value should be floating point number')
        print('Please check the --rel argument.')
        print('')
        print(__doc__)
        sys.exit(2)
    
    # parse the MS relative mass error
    try:
        mz_precision_abs = float(arguments['--abs'])
    except:
        print('')
        print('Invalid argument for the MS absolute mass error: %s' % arguments['--abs'])
        print('Value should be floating point number')
        print('Please check the --abs argument.')
        print('')
        print(__doc__)
        sys.exit(2)
    #mz_precision_rel               : MS relative mass error (Float)
    #mz_precision_abs               : MS relative mass error (Float)
    #tracer_isotope                 : tracer isotope (Str)
    
    # parse the verbose argument
    verbose = arguments['--verbose']

    # report the arguments used
    if verbose:
        print('parameters settings:')
        print('- random seed: %s' % random_seed)
        print('- isotopologue file: %s' % isotopologue_file)
        print('- model file: %s' % model_file)
        print('- natural abundance correction: %s' % natural_abundance_correction)
        print('- optimization method: %s' % optimization_method)
        print('- minimum number of iterations: %s' % min_num_iterations)
        print('- minimum region of attraction of best minimum: %s' % min_ROABM)
        print('- maximum missing mass for HCS rule: %s' % HCS_max_missing_mass)
        print('- confidence level for HCS rule: %s' % HCS_confidence_level)
        print('- MS relative mass error: %s' % mz_precision_rel)
        print('- MS absolute mass error: %s' % mz_precision_abs)
        print('')

    return (random_seed, isotopologue_file, model_file,
            natural_abundance_correction, 
            optimization_method,
            min_num_iterations, min_ROABM,
            HCS_max_missing_mass, HCS_confidence_level,
            mz_precision_rel, mz_precision_abs,
            verbose)



def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    
    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def parse_isotopologue_data(isotopologue_fn):
    """
    Parse the isotopologue profile data file.
    
    Args:
        isotopologue_fn - Required  : full path to file (Str)
        
    Returns:
        profile  : the isotopologue profile (List)
    """
    
    m_regex = re.compile(r'[mM](\d+)')
    profiles_temp = {}
    with open(isotopologue_fn, 'r') as isotopologue_f:
        for line in isotopologue_f:
            line_list = line.strip().split('\t')
            mr = m_regex.search(line_list[0])
            m = int(mr.groups()[0])
            intensity = float(line_list[1])
            profiles_temp[m] = intensity

    profile = []
    for m in range(max(profiles_temp.keys()) + 1):
        intensity = profiles_temp[m]
        profile.append(intensity)
            
    return profile


def parse_model_data(model_fn):
    """
    Parse the model data file.
    
    Args:
        model_fn - Required  : full path to file (Str)
    
    Returns:
        metabolite_name  : metabolite name (Str)
        compound_formula : metabolite compound formula (Str)
        ion_formula      : metabolite ion formula (Str)
        moieties         : moiety information (Dict)
        constraints      : constraint information (List)
    """
    e = xml.etree.ElementTree.parse(model_fn).getroot()

    metabolite_name = e.get('name')
    compound_formula = e.get('compound_formula')
    ion_formula = e.get('ion_formula')
    moieties = {}
    constraints = []
    
    for moiety_list in e.iter('moietyList'):
        for moiety in moiety_list:
            mid = moiety.get('id')
            name = moiety.find('name').text
            position = moiety.find('position').text
            num_labels = int(moiety.find('numLabels').text)
            moieties[mid] = {'name': name,
                             'position': position,
                             'num_labels': num_labels}
            
    for constraint_list in e.iter('constraintList'):
        for constraint in constraint_list:
            constraints.append(constraint.text)
    
    return metabolite_name, compound_formula, ion_formula, moieties, constraints


def normalize_isotopologue_profile(isotopologue_profile):
    """
    Normalize the isotopologue profile.
    
    Args:
        isotopologue_profile - Required  : isotopologue profile (List)
        
    Returns:
        normalized_isotopologue_profile  : normalized isotopologue profile (List)
    """
    
    ip_sum = sum(isotopologue_profile)
    normalized_isotopologue_profile = [ip / ip_sum for ip in isotopologue_profile]
    return normalized_isotopologue_profile


class IsotopomerDistributionCorrector():
    """
    The isotopic mass data is from G. Audi, A. H. Wapstra:
    Nucl. Phys A. 1993, 565, 1-65
    and G. Audi, A. H. Wapstra: Nucl. Phys A. 1995, 595, 409-480.

    The percent natural abundance data is from the 1997 report
    of the IUPAC Subcommittee for Isotopic Abundance Measurements
    by K.J.R. Rosman & P.D.P. Taylor: Pure Appl. Chem. 1999, 71, 1593-1607.
    """

    def __init__(self):
        """
        Define the natural abundances for each element.
        """
        
        self.natural_abundances = {}

        self.natural_abundances['H']  = ((1,1.007825,99.9885),
                                         (2,2.014102,0.0115),
                                         (3,3.016049,0.))
        self.natural_abundances['He'] = ((3,3.016029,0.000137),
                                         (4,4.002603,99.999863))
        self.natural_abundances['Li'] = ((6,6.015122,7.59),
                                         (7,7.016004,92.41))
        self.natural_abundances['Be'] = ((9,9.012182,100),)
        self.natural_abundances['B']  = ((10,10.012937,19.9),
                                         (11,11.009305,80.1))
        self.natural_abundances['C']  = ((12,12.000000,98.93),
                                         (13,13.003355,1.07),
                                         (14,14.003242,0.))
        self.natural_abundances['N']  = ((14,14.003074,99.632),
                                         (15,15.000109,0.368))
        self.natural_abundances['O']  = ((16,15.994915,99.757),
                                         (17,16.999132,0.038),
                                         (18,17.999160,0.205))
        self.natural_abundances['F']  = ((19,18.998403,100),)
        self.natural_abundances['Ne'] = ((20,19.992440,90.48),
                                         (21,20.993847,0.27),
                                         (22,21.991386,9.25))
        self.natural_abundances['Na'] = ((23,22.989770,100),)
        self.natural_abundances['Mg'] = ((24,23.985042,78.99),
                                         (25,24.985837,10.00),
                                         (26,25.982593,11.01))
        self.natural_abundances['Al'] = ((27,26.981538,100),)
        self.natural_abundances['Si'] = ((28,27.976927,92.230),
                                         (29,28.976495,4.683),
                                         (30,29.973770,3.087))
        self.natural_abundances['P']  = ((31,30.973762,100),)
        self.natural_abundances['S']  = ((32,31.972071,94.93),
                                         (33,32.971458,0.76),
                                         (34,33.967867,4.29),
                                         (36,35.967081,0.02))
        self.natural_abundances['Cl'] = ((35,34.968853,75.78),
                                         (37,36.965903,24.22))
        self.natural_abundances['Ar'] = ((36,35.967546,0.3365),
                                         (38,37.962732,0.0632),
                                         (40,39.962383,99.6003))
        self.natural_abundances['K']  = ((39,38.963707,93.2581),
                                         (40,39.963999,0.0117),
                                         (41,40.961826,6.7302))
        self.natural_abundances['Ca'] = ((40,39.962591,96.941),
                                         (42,41.958618,0.647),
                                         (43,42.958767,0.135),
                                         (44,43.955481,2.086),
                                         (46,45.953693,0.004),
                                         (48,47.952534,0.187))
        self.natural_abundances['Sc'] = ((45,44.955910,100),)
        self.natural_abundances['Ti'] = ((46,45.952629,8.25),
                                         (47,46.951764,7.44),
                                         (48,47.947947,73.72),
                                         (49,48.947871,5.41),
                                         (50,49.944792,5.18))
        self.natural_abundances['V']  = ((50,49.947163,0.250),
                                         (51,50.943964,99.750))
        self.natural_abundances['Cr'] = ((50,49.946050,4.345),
                                         (52,51.940512,83.789),
                                         (53,52.940654,9.501),
                                         (54,53.938885,2.365))
        self.natural_abundances['Mn'] = ((55,54.938050,100),)
        self.natural_abundances['Fe'] = ((54,53.939615,5.845),
                                         (56,55.934942,91.754),
                                         (57,56.935399,2.119),
                                         (58,57.933280,0.282))
        self.natural_abundances['Co'] = ((59,58.933200,100),)
        self.natural_abundances['Ni'] = ((58,57.935348,68.0769),
                                         (60,59.930791,26.2231),
                                         (61,60.931060,1.1399),
                                         (62,61.928349,3.6345),
                                         (64,63.927970,0.9256))
        self.natural_abundances['Cu'] = ((63,62.929601,69.17),
                                         (65,64.927794,30.83))
        self.natural_abundances['Zn'] = ((64,63.929147,48.63),
                                         (66,65.926037,27.90),
                                         (67,66.927131,4.10),
                                         (68,67.924848,18.75),
                                         (70,69.925325,0.62))
        self.natural_abundances['Ga'] = ((69,68.925581,60.108),
                                         (71,70.924705,39.892))
        self.natural_abundances['Ge'] = ((70,69.924250,20.84),
                                         (72,71.922076,27.54),
                                         (73,72.923459,7.73),
                                         (74,73.921178,36.28),
                                         (76,75.921403,7.61))
        self.natural_abundances['As'] = ((75,74.921596,100),)
        self.natural_abundances['Se'] = ((74,73.922477,0.89),
                                         (76,75.919214,9.37),
                                         (77,76.919915,7.63),
                                         (78,77.917310,23.77),
                                         (80,79.916522,49.61),
                                         (82,81.916700,8.73))
        self.natural_abundances['Br'] = ((79,78.918338,50.69),
                                         (81,80.916291,49.31))
        self.natural_abundances['Kr'] = ((78,77.920386,0.35),
                                         (80,79.916378,2.28),
                                         (82,81.913485,11.58),
                                         (83,82.914136,11.49),
                                         (84,83.911507,57.00),
                                         (86,85.910610,17.30))
        self.natural_abundances['Rb'] = ((85,84.911789,72.17),
                                         (87,86.909183,27.83))
        self.natural_abundances['Sr'] = ((84,83.913425,0.56),
                                         (86,85.909262,9.86),
                                         (87,86.908879,7.00),
                                         (88,87.905614,82.58))
        self.natural_abundances['Y']  = ((89,88.905848,100),)
        self.natural_abundances['Zr'] = ((90,89.904704,51.45),
                                         (91,90.905645,11.22),
                                         (92,91.905040,17.15),
                                         (94,93.906316,17.38),
                                         (96,95.908276,2.80))
        self.natural_abundances['Nb'] = ((93,92.906378,100),)
        self.natural_abundances['Mo'] = ((92,91.906810,14.84),
                                         (94,93.905088,9.25),
                                         (95,94.905841,15.92),
                                         (96,95.904679,16.68),
                                         (97,96.906021,9.55),
                                         (98,97.905408,24.13),
                                         (100,99.907477,9.63))
        self.natural_abundances['Tc'] = ((98,97.907216,100),)
        self.natural_abundances['Ru'] = ((96,95.907598,5.54),
                                         (98,97.905287,1.87),
                                         (99,98.905939,12.76),
                                         (100,99.904220,12.60),
                                         (101,100.905582,17.06),
                                         (102,101.904350,31.55),
                                         (104,103.905430,18.62))
        self.natural_abundances['Rh'] = ((103,102.905504,100),)
        self.natural_abundances['Pd'] = ((102,101.905608,1.02),
                                         (104,103.904035,11.14),
                                         (105,104.905084,22.33),
                                         (106,105.903483,27.33),
                                         (108,107.903894,26.46),
                                         (110,109.905152,11.72))
        self.natural_abundances['Ag'] = ((107,106.905093,51.839),
                                         (109,108.904756,48.161))
        self.natural_abundances['Cd'] = ((106,105.906458,1.25),
                                         (108,107.904183,0.89),
                                         (110,109.903006,12.49),
                                         (111,110.904182,12.80),
                                         (112,111.902757,24.13),
                                         (113,112.904401,12.22),
                                         (114,113.903358,28.73),
                                         (116,115.904755,7.49))
        self.natural_abundances['In'] = ((113,112.904061,4.29),
                                         (115,114.903878,95.71))
        self.natural_abundances['Sn'] = ((112,111.904821,0.97),
                                         (114,113.902782,0.66),
                                         (115,114.903346,0.34),
                                         (116,115.901744,14.54),
                                         (117,116.902954,7.68),
                                         (118,117.901606,24.22),
                                         (119,118.903309,8.59),
                                         (120,119.902197,32.58),
                                         (122,121.903440,4.63),
                                         (124,123.905275,5.79))
        self.natural_abundances['Sb'] = ((121,120.903818,57.21),
                                         (123,122.904216,42.79))
        self.natural_abundances['Te'] = ((120,119.904020,0.09),
                                         (122,121.903047,2.55),
                                         (123,122.904273,0.89),
                                         (124,123.902819,4.74),
                                         (125,124.904425,7.07),
                                         (126,125.903306,18.84),
                                         (128,127.904461,31.74),
                                         (130,129.906223,34.08))
        self.natural_abundances['I']  = ((127,126.904468,100),)
        self.natural_abundances['Xe'] = ((124,123.905896,0.09),
                                         (126,125.904269,0.09),
                                         (128,127.903530,1.92),
                                         (129,128.904779,26.44),
                                         (130,129.903508,4.08),
                                         (131,130.905082,21.18),
                                         (132,131.904154,26.89),
                                         (134,133.905395,10.44),
                                         (136,135.907220,8.87))
        self.natural_abundances['Cs'] = ((133,132.905447,100),)
        self.natural_abundances['Ba'] = ((130,129.906310,0.106),
                                         (132,131.905056,0.101),
                                         (134,133.904503,2.417),
                                         (135,134.905683,6.592),
                                         (136,135.904570,7.854),
                                         (137,136.905821,11.232),
                                         (138,137.905241,71.698))
        self.natural_abundances['La'] = ((138,137.907107,0.090),
                                         (139,138.906348,99.910))
        self.natural_abundances['Ce'] = ((136,135.907144,0.185),
                                         (138,137.905986,0.251),
                                         (140,139.905434,88.450),
                                         (142,141.909240,11.114))
        self.natural_abundances['Pr'] = ((141,140.907648,100),)
        self.natural_abundances['Nd'] = ((142,141.907719,27.2),
                                         (143,142.909810,12.2),
                                         (144,143.910083,23.8),
                                         (145,144.912569,8.3),
                                         (146,145.913112,17.2),
                                         (148,147.916889,5.7),
                                         (150,149.920887,5.6))
        self.natural_abundances['Pm'] = ((145,144.912744,100),)
        self.natural_abundances['Sm'] = ((144,143.911995,3.07),
                                         (147,146.914893,14.99),
                                         (148,147.914818,11.24),
                                         (149,148.917180,13.82),
                                         (150,149.917271,7.38),
                                         (152,151.919728,26.75),
                                         (154,153.922205,22.75))
        self.natural_abundances['Eu'] = ((151,150.919846,47.81),
                                         (153,152.921226,52.19))
        self.natural_abundances['Gd'] = ((152,151.919788,0.20),
                                         (154,153.920862,2.18),
                                         (155,154.922619,14.80),
                                         (156,155.922120,20.47),
                                         (157,156.923957,15.65),
                                         (158,157.924101,24.84),
                                         (160,159.927051,21.86))
        self.natural_abundances['Tb'] = ((159,158.925343,100),)
        self.natural_abundances['Dy'] = ((156,155.924278,0.06),
                                         (158,157.924405,0.10),
                                         (160,159.925194,2.34),
                                         (161,160.926930,18.91),
                                         (162,161.926795,25.51),
                                         (163,162.928728,24.90),
                                         (164,163.929171,28.18))
        self.natural_abundances['Ho'] = ((165,164.930319,100),)
        self.natural_abundances['Er'] = ((162,161.928775,0.14),
                                         (164,163.929197,1.61),
                                         (166,165.930290,33.61),
                                         (167,166.932045,22.93),
                                         (168,167.932368,26.78),
                                         (170,169.935460,14.93))
        self.natural_abundances['Tm'] = ((169,168.934211,100),)
        self.natural_abundances['Yb'] = ((168,167.933894,0.13),
                                         (170,169.934759,3.04),
                                         (171,170.936322,14.28),
                                         (172,171.936378,21.83),
                                         (173,172.938207,16.13),
                                         (174,173.938858,31.83),
                                         (176,175.942568,12.76))
        self.natural_abundances['Lu'] = ((175,174.940768,97.41),
                                         (176,175.942682,2.59))
        self.natural_abundances['Hf'] = ((174,173.940040,0.16),
                                         (176,175.941402,5.26),
                                         (177,176.943220,18.60),
                                         (178,177.943698,27.28),
                                         (179,178.945815,13.62),
                                         (180,179.946549,35.08))
        self.natural_abundances['Ta'] = ((180,179.947466,0.012),
                                         (181,180.947996,99.988))
        self.natural_abundances['W']  = ((180,179.946706,0.12),
                                         (182,181.948206,26.50),
                                         (183,182.950224,14.31),
                                         (184,183.950933,30.64),
                                         (186,185.954362,28.43))
        self.natural_abundances['Re'] = ((185,184.952956,37.40),
                                         (187,186.955751,62.60))
        self.natural_abundances['Os'] = ((184,183.952491,0.02),
                                         (186,185.953838,1.59),
                                         (187,186.955748,1.96),
                                         (188,187.955836,13.24),
                                         (189,188.958145,16.15),
                                         (190,189.958445,26.26),
                                         (192,191.961479,40.78))
        self.natural_abundances['Ir'] = ((191,190.960591,37.3),
                                         (193,192.962924,62.7))
        self.natural_abundances['Pt'] = ((190,189.959930,0.014),
                                         (192,191.961035,0.782),
                                         (194,193.962664,32.967),
                                         (195,194.964774,33.832),
                                         (196,195.964935,25.242),
                                         (198,197.967876,7.163))
        self.natural_abundances['Au'] = ((197,196.966552,100),)
        self.natural_abundances['Hg'] = ((196,195.965815,0.15),
                                         (198,197.966752,9.97),
                                         (199,198.968262,16.87),
                                         (200,199.968309,23.10),
                                         (201,200.970285,13.18),
                                         (202,201.970626,29.86),
                                         (204,203.973476,6.87))
        self.natural_abundances['Tl'] = ((203,202.972329,29.524),
                                         (205,204.974412,70.476))
        self.natural_abundances['Pb'] = ((204,203.973029,1.4),
                                         (206,205.974449,24.1),
                                         (207,206.975881,22.1),
                                         (208,207.976636,52.4))
        self.natural_abundances['Bi'] = ((209,208.980383,100),)


    def parse_formula(self, elemental_formula):
        """
        Parse the elemental formula.
        
        Args:
            elemental_formula - Required  : elemental formula (Str)
        
        Returns:
            elemental_formula_parsed3  : elemental formula (Dict)
        """
        
        elemental_formula_parsed = []
        elemental_formula_regex = re.compile(r'(\(?)([a-zA-Z]+)(\d*)(\)?)(\d*)')
        elemental_formula_parts = elemental_formula_regex.split(elemental_formula)
    
        while '' in elemental_formula_parts: elemental_formula_parts.remove('')
    
        group_ends = [item for item in range(len(elemental_formula_parts)) if elemental_formula_parts[item] == ')']
        group_counts = []
        for ge in group_ends:
            try:
                gc = int(elemental_formula_parts[ge+1])
            except:
                gc = 1
            group_counts.append(gc)
    
        # parsing out concatenated elements
        ele = None
        freq = None
        group = False
        group_done = False
        while elemental_formula_parts:
            part = elemental_formula_parts.pop(0)
            if part == '(':
                group = True
                group_count = group_counts.pop(0)
                continue
            elif part == ')':
                group = False
                group_done = True
                continue
            try:
                freq = int(part)
                if group:
                    freq *= group_count
                if not group_done:
                    elemental_formula_parsed.append(freq)
            except:
                ele = part
                while ele:
                    i = len(ele)
                    while ele not in self.natural_abundances:
                        ele = ele[:-1]
                        i -= 1
                        if i < 1: return
                    elemental_formula_parsed.append(ele)
                    if group:
                        freq = group_count
                    else:
                        freq = 1
                    elemental_formula_parsed.append(freq)
                    ele = part = part[i:]

            group_done = False
    
        # removing redundant frequencies
        elemental_formula_parsed2 = []
        previous_part_type = type(0)
        for part in elemental_formula_parsed:
            if type(part) == previous_part_type:
                elemental_formula_parsed2.pop()
            elemental_formula_parsed2.append(part)
            previous_part_type = type(part)
    
        elemental_formula_parsed3 = {}
        for i in range(0,len(elemental_formula_parsed2),2):
            ele = elemental_formula_parsed2[i]
            freq = elemental_formula_parsed2[i+1]
            if ele not in elemental_formula_parsed3:
                elemental_formula_parsed3[ele] = freq
            else:
                elemental_formula_parsed3[ele] += freq
    
        return elemental_formula_parsed3


    def generate_m0mass_and_distribution(self,
                                         molecular_formula_compound,
                                         molecular_formula_ion,
                                         isotope_focus=['C13']):
        """
        Generate the isotopologue profile for the given elemental formula.
        It makes sure the following peaks are provided:
            - the most intense peaks (natural abundance based)
            - the isotopologue peaks caused by the elements specified in `focus`
        
        Args:
            molecular_formula_compound - Required : the elemental formula of the original compound as a string
            molecular_formula_ion      - Required : the elemental formula of the measured ion as a string
            isotope_focus              - Optional : the list of isotopes of which the
                                                    isotopologue peaks need to be included
                                                    default is the C13 label.
        
        Returns:
            mass_isotopologue_distribution : a list of tuples of five elements:
                                             - mass
                                             - natural abundance determined intensity
                                             - formula
                                             - formula in dictionary format
                                             - focus element tag
        """
        
        elements_compound = self.parse_formula(molecular_formula_compound)
        elements_ion = self.parse_formula(molecular_formula_ion)
    
        mass_isotopologue_distribution = [{'formula': '',
                                           'formula_dict': {},
                                           'intensity': 1.}]
    
        # first determine the most intense peaks (due to natural abundance)
        # based on the formula of the ion (which is actually measured by the machine)
        # as opposed to the compound
        # The most intense peak is the M0 peak
        for e, f in elements_ion.items():
            element_abundances = self.natural_abundances[e]
            for _ in range(f):
                mass_isotopologue_distribution_temp1 = []
                for mid in mass_isotopologue_distribution:
                    formula_dict = mid['formula_dict']
                    intensity = mid['intensity']
                    for atom_num, _, fraction in element_abundances:
                        new_formula_dict = copy.deepcopy(formula_dict)
                        atom = '%s%s' % (e, atom_num)
                        if atom in formula_dict:
                            new_formula_dict[atom] += 1
                        else:
                            new_formula_dict[atom] = 1
                        new_formula = list(new_formula_dict.items())
                        new_formula.sort(key = lambda f: f[0])
                        new_formula = str(new_formula)
                        new_intensity = intensity * fraction / 100.
                        #if not new_intensity > 0: continue
                        if not new_intensity > 1e-5: continue
                        mass_isotopologue_distribution_temp1.append({'formula': new_formula,
                                                                     'formula_dict': new_formula_dict,
                                                                     'intensity': new_intensity})
                        
                # now merge the items with the same formula
                mass_isotopologue_distribution_temp2 = {}
                for mid in mass_isotopologue_distribution_temp1:
                    formula = mid['formula']
                    intensity = mid['intensity']
                    if formula in mass_isotopologue_distribution_temp2:
                        mass_isotopologue_distribution_temp2[formula]['intensity'] += intensity
                    else:
                        mass_isotopologue_distribution_temp2[formula] = mid
                
                mass_isotopologue_distribution = list(mass_isotopologue_distribution_temp2.values())
    
        
        # add mass and empty label to the distribution information
        isotope_masses = {}
        for e, e_info in self.natural_abundances.items():
            for atom_num, isotope_mass, _ in e_info:
                atom = '%s%s' % (e, atom_num)
                isotope_masses[atom] = isotope_mass
        for mid in mass_isotopologue_distribution:
            compound_mass = 0.
            for atom, amount in mid['formula_dict'].items():
                compound_mass += isotope_masses[atom] * amount
            mid['mass'] = compound_mass
            mid['label'] = []
    
        # determine M0 species (highest intensity)
        mass_isotopologue_distribution.sort(key = lambda x: x['intensity'], reverse = True)
        M0_species = mass_isotopologue_distribution[0]
        # parse the isotopes of the M0 species
        parse_isotope_regex = re.compile(r'([^\d]+)(\d+)')
        M0_isotopes = {}
        for isotope in M0_species['formula_dict'].keys():
            e, atom_num = parse_isotope_regex.findall(isotope)[0]
            M0_isotopes[e] = atom_num
    
        # again put the isotopologue distribution in a formula based dictionary
        mass_isotopologue_distribution_temp = {}
        for mid in mass_isotopologue_distribution:
            mass_isotopologue_distribution_temp[mid['formula']] = mid
    
    
        # now add the missing focus isotopologue signals
        for isotope_f in isotope_focus:
            # parse the focus isotope (e.g.: 'C13' --> 'C' and '13')
            e, atom_num = parse_isotope_regex.findall(isotope_f)[0]
            # if this element does not appear in the M0 species, nothing needs to be done
            if e not in M0_isotopes: continue
            # if the focus isotope is identical to the corresponding M0 isotope,
            # nothing needs to be done (e.g.: C12)
            isotope_f_M0 = '%s%s' % (e, M0_isotopes[e])
            if isotope_f == isotope_f_M0: continue
    
            # first add a M-1 signal (for verification during analysis)
            label_mass_diff = isotope_masses[isotope_f] - isotope_masses[isotope_f_M0]
            compound_mass = M0_species['mass'] - label_mass_diff
            mid = {'formula': None,
                   'formula_dict': None,
                   'intensity': 0.,
                   'mass': compound_mass,
                   'label': ['%s_%s' % (isotope_f, -1)]}
            mass_isotopologue_distribution_temp[isotope_f] = mid
    
            # check how many times this element appears in the formula of the compound
            e_count = elements_compound[e]
             
            for e_i in range(e_count + 1):
                # determine the formula_dict corresponding to this amount of focus isotope
                formula_dict = copy.deepcopy(M0_species['formula_dict'])
                # get the number of M0 focus elements in the compound formula
                isotope_f_M0_count = formula_dict[isotope_f_M0]
                # temporarily remove the M0 variant of the focus isotope
                del formula_dict[isotope_f_M0]
                # and now add the correct amounts
                new_isotope_f_M0_count = isotope_f_M0_count - e_i
                new_isotope_f_count = e_i
                if new_isotope_f_M0_count > 0:
                    formula_dict[isotope_f_M0] = new_isotope_f_M0_count
                if new_isotope_f_count > 0:
                    formula_dict[isotope_f] = new_isotope_f_count
                # convert the formula dictionary to a formula
                formula = list(formula_dict.items())
                formula.sort(key = lambda f: f[0])
                formula = str(formula)
                
                # check if this formula already appears in the temporary
                # mass_isotopologue_distribution. If not it is added with intensity 0
                if formula in mass_isotopologue_distribution_temp:
                    mass_isotopologue_distribution_temp[formula]['label'].append('%s_%s' % (isotope_f, e_i))
                else:
                    compound_mass = 0.
                    for atom, amount in formula_dict.items():
                        compound_mass += isotope_masses[atom] * amount
                    mid = {'formula': formula,
                           'formula_dict': formula_dict,
                           'intensity': 0.,
                           'mass': compound_mass,
                           'label': ['%s_%s' % (isotope_f, e_i)]}
                    mass_isotopologue_distribution_temp[formula] = mid
    
        mass_isotopologue_distribution = list(mass_isotopologue_distribution_temp.values())
        mass_isotopologue_distribution.sort(key = lambda x: x['mass'])
        
        return M0_species['mass'], mass_isotopologue_distribution


    def merge_peaks(self,
                    mass_isotopologue_distribution_info,
                    mz_precision_rel,
                    mz_precision_abs,
                    isotope_focus = 'C13'):
        """
        Due to limited precision of the MS machine
        some isotopologue signals might overlap when measured
        This overlap is calculated here
       
        Args:
            mass_isotopologue_distribution_info - Required : theoretical isotopologue profile (List)
            mz_precision_rel - Required                    : relative mass error in ppm of the MS machine (Float)
            mz_precision_abs - Required                    : absolute mass error in Da of the MS machine (Float)
            isotope_focus - Optional                       : the isotope used as tracer (Str)
                                                             (default: C13)
        
        Returns:
            CM  : Fernandez1996 correction matrix (Numpy Array)
        """
        
        # first determine the groups
        previous_mass = float('-Inf')
        mass_groups = []
        tag_groups = []
        for mid in mass_isotopologue_distribution_info:
            # determine the tag
            tag_num = None
            for label in mid['label']:
                if label.startswith(isotope_focus):
                    tag_num = int(label.split('_')[1])
                    break
            # determine the mass difference
            mass = mid['mass']
            mass_diff_margin = max(mz_precision_abs,(mass * mz_precision_rel / 1000000.))
            # start new group?
            if mass - previous_mass > mass_diff_margin:
                mass_groups.append([mass])
                tag_groups.append([tag_num])
            else:
                mass_groups[-1].append(mass)
                tag_groups[-1].append(tag_num)
            # update previous mass
            previous_mass = mass
    
        # link masses to tag_nums
        tags = []
        for tag_group in tag_groups:
            while (len(tag_group) > 1) and (None in tag_group):
                tag_group.remove(None)
            tags.append(tag_group[0])
        mass2tag = {}
        for tag, masses in zip(tags, mass_groups):
            for mass in masses:
                mass2tag[mass] = tag
    
        # add the M level information to the isotopologue distribution
        for mid in mass_isotopologue_distribution_info:
            mid['M_level'] = mass2tag[mid['mass']]


    def determine_multiplication_factor(self, formula_dict):
        """
        Calculating the total number of (different) permutations
        from the given isotope composition
    
        Args:
            formula_dict - Required : the molecular isotope composition (Dict)
    
        Returns:
            num_unique_perms: number of (different) permutations
        """
        element_counts = {}
        parse_isotope_regex = re.compile(r'([^\d]+)(\d+)')
        for isotope, amount in formula_dict.items():
            if amount == 0: continue
            e, atom_num = parse_isotope_regex.findall(isotope)[0]
            if e in element_counts:
                element_counts[e].append(amount)
            else:
                element_counts[e] = [amount]
             
        multiplication_factor = 1
        for counts in element_counts.values():
            numerator = math.factorial(sum(counts))
            denominator = 1
            for c in counts:
                denominator *= math.factorial(c)
            multiplication_factor *= (numerator / denominator)
    
        return multiplication_factor


    def generate_correction_matrix(self,
                                   mass_isotopologue_distribution_info,
                                   isotope_focus = 'C13'):
        """
        generate the Fernandez1996 correction matrix
           
        Args:
            mass_isotopologue_distribution_info - Required : theoretical isotopologue profile (List)
            isotope_focus - Optional                       : the isotope used as tracer (Str)
                                                             (default: C13)
        
        Returns:
            CM  : Fernandez1996 correction matrix (Numpy Array)
        """
    
        parse_isotope_regex = re.compile(r'([^\d]+)(\d+)')
    
        # determine from the tags in the mass_isotopologue_distribution_info
        # how many times the isotope_focus can appear
        max_tag_num = 0
        for mid in mass_isotopologue_distribution_info:
            for label in mid['label']:
                if label.startswith(isotope_focus):
                    tag_num = int(label.split('_')[1])
                    if tag_num > max_tag_num:
                        max_tag_num = tag_num
    
        max_tag_num += 1
        # and now fill the correction matrix
        CM = numpy.full((max_tag_num, max_tag_num), 0.)
        for mid in mass_isotopologue_distribution_info:
            m_level = mid['M_level']
            if m_level == None or m_level < 0:
                # this signal does not make part of a signal of interest
                # and is therefore not included in the correction matrix
                continue
            else:
                column = m_level
                formula_dict = mid['formula_dict']
                if isotope_focus not in formula_dict:
                    formula_dict[isotope_focus] = 0
                tag_num = formula_dict[isotope_focus] + 1
                for row in range(tag_num):
                    f_dict = copy.deepcopy(formula_dict)
                    f_dict[isotope_focus] -= row
                    # mass increase is independent of the positions of
                    # isotopes incorporation
                    # the number of possibilities are therefore calculated 
                    mult_factor = self.determine_multiplication_factor(f_dict)
                    # calculate natural abundance intensity of matrix element
                    intensity = 1.
                    for isotope, amount in f_dict.items():
                        # get the fraction of the isotope
                        e, atom_num = parse_isotope_regex.findall(isotope)[0]
                        atom_num = int(atom_num)
                        element_abundances = self.natural_abundances[e]
                        for an, _, fraction in element_abundances:
                            if an == atom_num: break
                        for _ in range(amount):
                            intensity *= (fraction * 0.01)
                    CM[row][column] += (intensity * mult_factor)
    
        return CM
            

    def Fernandez1996_correction(self,
                                 isotopologue_profile,
                                 molecular_formula_compound,
                                 molecular_formula_ion,
                                 mz_precision_rel,
                                 mz_precision_abs,
                                 isotope_focus = 'C13'):
        """
        Perform the Fernandez1996 correction.
           
        Args:
            isotopologue_profile - Required        : isotopologue profile (List)
            elemental_formula_compound - Required  : elemental formula of the metabolite (Str)
            elemental_formula_ion - Required       : elemental formula of ion derived from
                                                     the metabolite and measured by MS (Str)
            mz_precision_rel - Required            : relative mass error in ppm of the MS machine (Float)
            mz_precision_abs - Required            : absolute mass error in Da of the MS machine (Float)
            isotope_focus - Optional               : the isotope used as tracer (Str)
                                                     (default: C13)
        
        Returns:
            corrected_isotopologue_profile  : corrected isotopologue profile (Numpy Array)
        """
    
        # determine the theoretical isotopologue distribution
        # for this ion with the signals relevant for the specific
        # isotope focus tagged
        _, mass_isotopologue_distribution_info = self.generate_m0mass_and_distribution(molecular_formula_compound,
                                                                                       molecular_formula_ion,
                                                                                       [isotope_focus])
    
        # based on the technical limitation of the mass spectrometer
        # some of the theoretical isotopologues might overlap with other
        # this overlap is calculated here
        self.merge_peaks(mass_isotopologue_distribution_info,
                         mz_precision_rel,
                         mz_precision_abs,
                         isotope_focus)
    
        # generate the correction matrix as described by the
        # Fernandez 1996 paper:
        # Correction of 13C Mass Isotopomer Distributions for Natural Stable Isotope Abundance
        CM = self.generate_correction_matrix(mass_isotopologue_distribution_info,
                                             isotope_focus)
    
        # make sure the input mass isotopologue distribution vector
        # is of same dimension M as the correction matrix (MxM)
        M = CM.shape[0]
        mid_size = len(isotopologue_profile)
        if mid_size < M:
            # add trailing zeros
            isotopologue_profile = isotopologue_profile + [0 for i in range(M - mid_size)]
        elif mid_size > M:
            # removing excess signals
            isotopologue_profile = isotopologue_profile[:M]
    
        return numpy.dot(isotopologue_profile, numpy.linalg.inv(CM))




class Deconvoluter():
    def __init__(self, m_name, m_formula, m_moieties, m_constraints,
                 isotopologue_profile, verbose = False):
        """
        Initialize the Deconvoluter class.
        
        Args:
            m_name - Required                : metabolite name (Str)
            m_formula - Required             : metabolite formula (Str)
            m_moieties - Required            : moiety info (Dict)
            m_constraints - Required         : constraints info (List)
            isotopologue_profile - Required  : isotopologue profile (List)
            verbose - Optional               : verbose flag (Bool)
        """
        
        self.name = m_name
        self.formula = m_formula
        self.moiety_info = m_moieties
        self.constraints_txt = m_constraints
        self.constraints = {}
        self.isotopologue_profile = isotopologue_profile
        self.verbose = verbose
        self.result = None
        self.res_dict = {}
        self.num_dict = {}
        self.fit_dict = {}
        # building the model
        self.build_model()
    
    
    def build_model(self):
        """
        Build the metabolite MAIMS model.
        """
        
        # get number of carbons from formula
        idc = IsotopomerDistributionCorrector()
        num_carbons = idc.parse_formula(self.formula)['C']
        
        # generate positional possibilities dictionary
        positional_possibilities = {}
        for mid, m_info in self.moiety_info.items():
            position = m_info['position']
            if position not in positional_possibilities:
                positional_possibilities[position] = [mid]
            else:
                positional_possibilities[position].append(mid)
        
        # generate labeling combinations
        labeling_combinations = list(itertools.product(*positional_possibilities.values()))
        
        # generate isotologue contributions
        isotopologue_contributions = {}
        for labeling_combination in labeling_combinations:
            isotopologue_mass = sum([self.moiety_info[lc]['num_labels'] for lc in labeling_combination])
            if isotopologue_mass not in isotopologue_contributions:
                isotopologue_contributions[isotopologue_mass] = [labeling_combination]
            else:
                isotopologue_contributions[isotopologue_mass].append(labeling_combination)

        # defining the moiety sympy variables
        moiety_vars = {}
        for m in self.moiety_info:
            moiety_vars[m] = sympy.Symbol('%s' % m)
        
        # defining the individual fit terms
        fit_terms = {}
        for c in range(num_carbons + 1):
            # building the term
            ics = isotopologue_contributions.get(c, [])
            ics = [[moiety_vars[v] for v in ic] for ic in ics]
            t = sum([functools.reduce(operator.mul, ic, 1) for ic in ics])
            fit_terms[c] = abs(t - self.isotopologue_profile[c])
        
        # parsing the constraints
        for constraint in self.constraints_txt:
            cl, cr = constraint.split('=')
            cl, cr = parse_expr(cl), parse_expr(cr)
            # getting the highest alphabetical variable (HAV)
            highest_var = sorted([str(s) for s in cl.free_symbols | cr.free_symbols])[-1]
            highest_var = moiety_vars[highest_var]
            # generating the constraint equation and solve for HAV
            c_eq = sympy.Eq(cl, cr)
            # only one solution is assumed for these constraints
            c_eq = sympy.solve(c_eq, highest_var)[0]
            # storing the constraint
            self.constraints[highest_var] = c_eq
        
        # substitute the constraints in the fit function
        substitutions_made = True
        while substitutions_made:
            substitutions_made = False
            for var, c_eq in self.constraints.items():
                for c, fit_term in fit_terms.items():
                    try:
                        # substitution not possible for terms without variables
                        fit_term_updated = fit_term.subs(var, c_eq)
                    except:
                        fit_term_updated = fit_term
                    if not fit_term == fit_term_updated:
                        substitutions_made = True
                    fit_terms[c] = fit_term_updated
        
        # potentially print the cost function
        if self.verbose:
            print('Setting the %s cost terms...' % self.name)
            for c in range(num_carbons + 1):
                print('t%s =' % c, fit_terms[c])
        
        # generating the total fit function
        fit_function_total = sum(fit_terms.values())
        
        # defining the fit function arguments
        final_moiety_vars = [v for v in fit_function_total.free_symbols if v in moiety_vars.values()]
        final_moiety_vars = sorted([str(v) for v in final_moiety_vars])
        self.contribution_variables_txt = final_moiety_vars
        final_moiety_vars = [moiety_vars[v] for v in final_moiety_vars]
        self.contribution_variables = final_moiety_vars
        self.bounds = [(0, 1) for _ in self.contribution_variables]
        
        # defining the fit function
        self.fit_function = sympy.lambdify([self.contribution_variables], fit_function_total)
    
    
    def I_ROA_rule(self, count, ROABM, min_num_iterations, min_ROABM):
        """
        Calculate the I/ROA rule value.
        
        Calculate the I/ROA (minimum number of iterations / minimum region
        of attraction) rule value
        
        Args:
            count - Required  : number of iterations performed thus far (Int)
            ROABM - Required  : current region of attraction of best minimum (Int)
            min_num_iterations - Required  : minimum required number of iterations (Int)
            min_ROABM - Required  : minimum required region of attraction of best minimum (Int)
        
        Returns:
            progress_fraction  : progress fraction (Float)
        """
        
        I_outcome = float(count) / min_num_iterations
        ROA_outcome = float(ROABM) / min_ROABM
        return min(I_outcome, ROA_outcome)
        
    
    def HCS_rule(self,
                 num_restarts,
                 HCS_max_missing_mass = 0.4,
                 HCS_confidence_level = 0.4):
        """
        Calculate the HCS rule value.
        
        Calculate the high-confidence stopping (HCS) rule value
        This rule is described in:
        How many random restarts are enough?, Travis Dick, Eric Wong, Christoph Dann
        
        Args:
            num_restarts - Required  : number of iterations performed thus far (Int)
            HCS_max_missing_mass - Required  : the c parameter of the algorithm (Float)
            HCS_confidence_level - Required  : the delta parameter of the algorithm (Float)
        
        Returns:
            value gradually reaching zero until stop is required (Float)
        """
        
        F1n = float(list(self.num_dict.values()).count(1))
        n = float(num_restarts)
        Cn = (F1n / n) + (4.560477932 * math.sqrt((math.log(3 / HCS_confidence_level)) / n))
        
        return Cn - HCS_max_missing_mass
    
    
    def uniform_initial_guess_generation(self, size = 5000):
        """
        Generate initial guesses for the parameters to optimize.
        
        Generate an set of initial guesses for each of the parameters to
        optimize within its boundary values.
        
        Args:
            size - Required  : size of the set of initial guesses (Int)
        
        Returns:
            random_initial_guesses  : set of initial guesses (List)
        """
        
        random_initial_guesses_temp = []
        for par_num in range(len(self.contribution_variables)):
            random_initial_guesses_temp.append(numpy.random.uniform(*self.bounds[par_num], size=size))
            
        random_initial_guesses = []
        for rig in zip(*random_initial_guesses_temp):
            random_initial_guesses.append(rig)
            
        return random_initial_guesses


    def deconvolute_localopt(self, initial_guess, optimization_method='SLSQP'):
        """
        Perform local optimization from starting point.
        
        Optimization method can be one of (L-BFGS-B, TNC, SLSQP)
        
        Args:
            initial_guess - Required        : starting point for the local optimization (List)
            optimization_method - Optional  : optimization method (Str)
        
        Returns:
            res  : optimization result (Obj)
        """
        res = scipy.optimize.minimize(self.fit_function,
                                      initial_guess,
                                      method = optimization_method,
                                      bounds = self.bounds)

        return res


    def deconvolute_localopt_multistart(self,
                                        optimization_method = 'SLSQP',
                                        min_num_iterations = 5000,
                                        min_ROABM = 20,
                                        HCS_max_missing_mass = 0.4,
                                        HCS_confidence_level = 0.4):
        """
        Perform the multistart deconvolution optimization.
        
        Args:
            optimization_method - Optional   : optimization method (Str)
            min_num_iterations - Optional    : minimum number of iterations (Int)
            min_ROABM - Optional             : minimum region of attraction of the best minimum (Int)
            HCS_max_missing_mass - Optional  : the c parameter of the algorithm (Float)
            HCS_confidence_level - Optional  : the delta parameter of the algorithm (Float)
        
        Returns:
            count  : the number of iterations performed (Int)
        """

        best_res = None
        best_res_red = None
        best_fit = float('Inf')
        ROABM = 0 # Region of Attraction of Best Minimum
        count = 0
        progress_fraction_fix = 0.
        starting_point_block_size = 5000

        continue_search = True

        while continue_search:
            # first generate some initial guesses as starting points
            random_initial_guesses = self.uniform_initial_guess_generation(size = starting_point_block_size)
            
            
            for rig in random_initial_guesses:
                
                if show_progress:
                    print_progress(progress_fraction_fix, 1., prefix = 'Progress:', suffix = 'Complete', barLength = 50)
    
                res = self.deconvolute_localopt(rig, optimization_method)
    
                res_red = tuple(['%.1f' % (100 * xi) for xi in res.x])
                fit = res.fun
                
                if res_red not in self.num_dict:
                    self.num_dict[res_red] = 1
                    self.res_dict[res_red] = res.x
                    self.fit_dict[res_red] = fit
                else:
                    self.num_dict[res_red] += 1
                    if fit < self.fit_dict[res_red]:
                        self.fit_dict[res_red] = fit
                        self.res_dict[res_red] = res.x
    
                if fit < best_fit:
                    best_fit = fit
                    best_res = res
                    best_res_red = res_red
                
                ROABM = self.num_dict[best_res_red]
                count += 1
                
                # the I/ROA rule part
                I_ROA_outcome = self.I_ROA_rule(count, ROABM,
                                                min_num_iterations,
                                                min_ROABM)
                
                if I_ROA_outcome >= 1.:
                    continue_search = False
                    break
                
                progress_fraction_I_ROA = I_ROA_outcome
                
                # the HCS rule code part
                HCS_outcome = self.HCS_rule(count,
                                            HCS_max_missing_mass,
                                            HCS_confidence_level)
                
                if HCS_outcome <= 0.:
                    continue_search = False
                    break
                
                progress_fraction_HCS = 1 - min(1, HCS_outcome)
                
                progress_fraction = max(progress_fraction_I_ROA,
                                        progress_fraction_HCS)
                progress_fraction = min(1, progress_fraction)
                
                if progress_fraction > progress_fraction_fix:
                    progress_fraction_fix = progress_fraction

        self.result = best_res
        return count
        

    def generate_solution(self):
        """
        Generate the general result from the deconvolution outcome
        
        The deconvolution outcome and constraints are combined
        to provide the contributions of all possible metabolite moieties.
        """
        
        self.solution = []
        for r_id, v in enumerate(self.contribution_variables):
            self.solution.append((v, self.result.x[r_id]))
        
        # add additional constraint variables to the solution
        substitutions_necessary = True
        while substitutions_necessary:
            substitutions_necessary = False
            for v, v_expr in self.constraints.items():
                v_expr_updated = v_expr.subs(self.solution)
                if len(v_expr_updated.free_symbols):
                    substitutions_necessary = True
                else:
                    if (v, v_expr_updated) not in self.solution:
                        self.solution.append((v, v_expr_updated))

        self.solution.sort(key = lambda x: str(x[0]))


    def get_result_txt(self):
        """
        Generate textual representation of the result.
        
        Returns:
            _result_txt  : result (Txt)
        """
        
        if not self.result: return ''
        
        # generating the solution
        self.generate_solution()
        
        # get longest label length
        longest_label_len = max([len(v['name']) + len(str(v['num_labels']))
                                 for v in self.moiety_info.values()])
        
        # generating the textual output
        _result_txt = ''
        for var, val in self.solution:
            v_txt = str(var)
            name = self.moiety_info[v_txt]['name']
            num_labels = self.moiety_info[v_txt]['num_labels']
            label = '%s%s' % (name, num_labels)
            _result_txt += label
            _result_txt += ' ' * (longest_label_len - len(label))
            _result_txt += ' = %5.1f' % (100* abs(val))
            _result_txt += '%\n'

        fit = 'fit = %s' % self.result.fun
        _result_txt += fit

        return _result_txt


def main():
    """
    MAIMS main method
    """
    
    print('#####################################')
    print('#      ~~~~~~   MAIMS   ~~~~~~      #')
    print('#     written by Dries Verdegem     #')
    print('#####################################')
    print('')
    
    # getting the script arguments
    arguments = parseopts()
    
    random_seed = arguments[0]
    isotopologue_fn = arguments[1]
    model_fn = arguments[2]
    natural_abundance_correction = arguments[3]
    optimization_method = arguments[4]
    min_num_iterations = arguments[5]
    min_ROABM = arguments[6]
    HCS_max_missing_mass = arguments[7]
    HCS_confidence_level = arguments[8]
    mz_precision_rel = arguments[9]
    mz_precision_abs = arguments[10]
    verbose = arguments[11]
    
    # seeding the random number generator
    numpy.random.seed(random_seed)

    # parsing the isotopologue file
    isotopologue_profile = parse_isotopologue_data(isotopologue_fn)
    
    # parsing the model file
    model_info = parse_model_data(model_fn)
    m_name = model_info[0]
    m_compound_formula = model_info[1]
    m_ion_formula = model_info[2]
    m_moieties = model_info[3]
    m_constraints = model_info[4]

    # if necessary (indicated by user), perform natural abundance correction
    if natural_abundance_correction:
        idc = IsotopomerDistributionCorrector()
        corrected_isotopologue_profile = list(idc.Fernandez1996_correction(isotopologue_profile,
                                                                           m_compound_formula,
                                                                           m_ion_formula,
                                                                           mz_precision_rel,
                                                                           mz_precision_abs,
                                                                           isotope_focus = 'C13'))
        
    else:
        corrected_isotopologue_profile = isotopologue_profile

    # clipping negative values to zero
    pos_corrected_isotopologue_profile = [max(0., cip) for cip in corrected_isotopologue_profile]

    # normalizing the isotopologue profile
    normalized_pos_corrected_isotopologue_profile = normalize_isotopologue_profile(pos_corrected_isotopologue_profile)


    if verbose:
        print('isotopologue profile')
        print('-->', isotopologue_profile)
        print('natural abundance corrected isotopologue profile')
        print('-->', corrected_isotopologue_profile)
        print('positive natural abundance corrected isotopologue profile')
        print('-->', pos_corrected_isotopologue_profile)
        print('normalized positive natural abundance corrected isotopologue profile')
        print('-->', normalized_pos_corrected_isotopologue_profile)
        print('')

    # initialize the deconvoluter
    deconv = Deconvoluter(m_name,
                          m_ion_formula,
                          m_moieties,
                          m_constraints,
                          normalized_pos_corrected_isotopologue_profile,
                          verbose)
    
    print('')
    print('Performing the %s deconvolution...' % m_name)
    print('')

    t_start = time.time()
    num_starts = deconv.deconvolute_localopt_multistart(optimization_method = optimization_method,
                                                        min_num_iterations = min_num_iterations,
                                                        min_ROABM = min_ROABM,
                                                        HCS_max_missing_mass = HCS_max_missing_mass,
                                                        HCS_confidence_level = HCS_confidence_level)
    t_end = time.time()

    if verbose:
        num_list = list(deconv.num_dict.items())
        num_list.sort(key = lambda x: x[1], reverse = True)
        fit_list = list(deconv.fit_dict.items())
        fit_list.sort(key = lambda x: x[1])
        
        print('')
        print('')
        print('top %s local minima with largest region of attraction' % num_top_results)
        print('optimized parameters | number of times found | fit value (smaller values are better)')
        print('-' * 80)
        for res, num_found in num_list[:num_top_results]:
            print('%s | (%s/%s) | %s' % (res, num_found,
                                         num_starts,
                                         deconv.fit_dict[res]))
        print('-' * 80)
        print('')

        print('top %s best local minima' % num_top_results)
        print('optimized parameters | number of times found | fit value (smaller values are better)')
        print('-' * 80)
        for res, fit_value in fit_list[:num_top_results]:
            print('%s | (%s/%s) | %s' % (res, deconv.num_dict[res],
                                         num_starts, fit_value))
        print('-' * 80)
        print('')

        # print result
        print('best optimization output')
        print(deconv.result)

    hours, remainder = divmod(int(round(t_end - t_start)), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours == 0 and minutes == 0:
        runtime = '%ss' % (seconds)
    elif hours == 0:
        runtime = '%smin %ss' % (minutes, seconds)
    else:
        runtime = '%sh %smin %ss' % (hours, minutes, seconds)

    print('')
    print('-' * 80)
    print('final result (obtained in %s and after %s local searches)' % (runtime,
                                                                         num_starts))
    print('-' * 80)
    print(deconv.get_result_txt())
    print('-' * 80)


if __name__ == '__main__':
    main()
