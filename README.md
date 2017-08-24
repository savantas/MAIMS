MAIMS
=====
* MAIMS deconvolutes the isotopologue profile of large metabolites upon specific tracer administration into individual contributions of metabolic moieties.
* Compatibility: MAIMS is Python 2.7 and Python 3.5+ compliant
* Prerequisites: Apart from a working version of Python, the following non-standard libraries need to be installed: scipy, numpy and sympy (MAIMS.py) and matplotlib (timeseries_plot.py)
* Disclaimer: MAIMS should work on all platforms, but was only tested in a Linux (Ubuntu 16.04) system.

Installation of MAIMS
---------------------
* Make sure you have all MAIMS dependencies installed: numpy, scipy, sympy and matplotlib
* Copy/paste the MAIMS source files to a dedicated folder

Running MAIMS
-------------
* For more information on how to run MAIMS, type:
```
	python /path_to_maims/MAIMS.py -h
```
* Example:
```
	python MAIMS.py -m UDP_GlcNAc.mdl -f example_UDPGlcNAc_noncorrected_isotopologue.txt -n -v
```
* For more information on how to run timeseries_plot.py, type:
```
	python /path_to_maims/timeseries_plot.py -h
```
* Example:
```
	python timeseries_plot.py -i example_timeseries_plot_input.txt -f 14 -x 0.7 -y 0.89
```

Citing MAIMS
------------
To be announced soon
