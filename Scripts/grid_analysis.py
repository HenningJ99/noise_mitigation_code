"""
This script processes the output from grid_simulation.py further.

Input:
    - Shear catalog from the simulations
    - Input catalog from the simulations
    - Config file

Output:
    - Measured shears against input shears in magnitude and time bins with uncertainties

Syntax:
    grid_analysis.py <total_gal> <gal_per_shear> <nx_tiles> <pixel_noise_times> <shear_interval> <path>

"""
import random

from astropy.table import Table
from astropy.table import QTable
from astropy.io import ascii
import numpy as np
import sys
import os
import logging
import galsim
import timeit
import configparser
import functions as fct
import math
import ray
import psutil
from scipy.optimize import curve_fit
import scipy
import matplotlib.pyplot as plt
import datetime
import ray

# ------------------------------PROCESS INPUT AND CONFIG ------------------------------------------------------------#
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("real_galaxies")


def linear_function(x, a, b):
    return a * x + b

# Parse the config file
config = configparser.ConfigParser()
config.read('config_grid.ini')

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

subfolder = sys.argv[6]
object_number = int(sys.argv[1])
average_num = int(sys.argv[2])

ray.init(num_cpus=4, log_to_driver=False)

# Read in the catalogs in chunks
catalog = ascii.read(subfolder + '/shear_catalog.dat', fast_reader={'chunk_size': 100 * 1000000})
input_catalog = ascii.read(subfolder + '/input_catalog.dat', fast_reader={'chunk_size': 100 * 1000000})

# Sort them by galaxy id
catalog = catalog.group_by('galaxy_id')
input_catalog = input_catalog.group_by('galaxy_id')

# Convert the catalogs to masked tables
catalog = Table(catalog, masked=True, copy=False)
input_catalog = Table(input_catalog, masked=True, copy=False)

""" 
Extract the intrinsic g1 from the catalog. Since the input catalog contains only one version per galaxy, we have
to insert the values for the other 3 versions as well. 
"""
intrinsic_g1 = np.ma.masked_array(input_catalog["intr_g1"][np.isin(input_catalog["galaxy_id"], catalog["galaxy_id"])])
intrinsic_g1_complete = np.insert(intrinsic_g1, np.arange(len(intrinsic_g1)) + 1, -intrinsic_g1)
intrinsic_g1_complete = np.insert(intrinsic_g1_complete, np.arange(len(intrinsic_g1_complete), step=2) + 1,
                                  intrinsic_g1)
intrinsic_g1_complete = np.insert(intrinsic_g1_complete, np.arange(len(intrinsic_g1_complete), step=3) + 1,
                                  -intrinsic_g1)


intrinsic_g2 = np.ma.masked_array(input_catalog["intr_g2"][np.isin(input_catalog["galaxy_id"], catalog["galaxy_id"])])
intrinsic_g2_complete = np.insert(intrinsic_g2, np.arange(len(intrinsic_g2)) + 1, -intrinsic_g2)
intrinsic_g2_complete = np.insert(intrinsic_g2_complete, np.arange(len(intrinsic_g2_complete), step=2) + 1,
                                  intrinsic_g2)
intrinsic_g2_complete = np.insert(intrinsic_g2_complete, np.arange(len(intrinsic_g2_complete), step=3) + 1,
                                  -intrinsic_g2)

print(len(catalog[catalog["mag_input"] < 24.5]))
# ---- SN CUT
SN_CUT = float(config['SIMULATION']["sn_cut"])
catalog["meas_g1"].mask = (catalog["S/N"] <= SN_CUT) | (catalog["meas_g1"] == 0) | (np.isnan(catalog["meas_g1"])) 
catalog["meas_g2"].mask = (catalog["S/N"] <= SN_CUT) | (catalog["meas_g2"] == 0) | (np.isnan(catalog["meas_g2"])) 

print(100 * len(catalog["meas_g1"][(catalog["meas_g1"] >= 5) | (catalog["meas_g1"] <= -5)]) / len(catalog["meas_g1"]))

# Check how many measurements end up masked out
for i in range(4):
    print(np.sum(catalog[i::4]["meas_g1"].mask))
# ---- Outlier detection (Remove measured shears with an absolute value larger than 5)
catalog["meas_g1"].mask = catalog["meas_g1"].mask | (catalog["meas_g1"] >= 5) | (catalog["meas_g1"] <= -5)
catalog["meas_g2"].mask = catalog["meas_g2"].mask | (catalog["meas_g2"] >= 5) | (catalog["meas_g2"] <= -5)
intrinsic_g1_complete.mask = catalog["meas_g1"].mask
intrinsic_g2_complete.mask = catalog["meas_g2"].mask

if simulation["bin_type"] == "GEMS":
    bin_type = "mag_input"
elif simulation["bin_type"] == "MEAS":
    bin_type = "mag_meas"

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])

shear_min = -float(sys.argv[5])
shear_max = float(sys.argv[5])

nx_tiles = int(sys.argv[3])
ring_num = nx_tiles

if int(sys.argv[4]) == 0:
    ny_tiles = 1
else:
    ny_tiles = int(sys.argv[4]) * 2

# ------------------------------------------ ANALYSIS -----------------------------------------------------------------

start = timeit.default_timer()

mag_bins = int(simulation["bins_mag"])

if mag_bins != 0:
    magnitudes = [
        float(simulation["min_mag"]) + k * (float(simulation["max_mag"]) - float(simulation["min_mag"])) / (mag_bins)
        for k in range(mag_bins + 1)]
else:
    magnitudes = [float(simulation["max_mag"])]  # If no binning at all
print(magnitudes)

binning = [-0.1 + 0.01 * k for k in range(21)]
shears = [shear_min + (shear_max - shear_min) / ((object_number / average_num) - 1) * m for m in range(20)]

index = 0

# Put the config and the command line arguments in shared memory since the analysis is parallelized
config_ref = ray.put(config)
argv_ref = ray.put(sys.argv)

# Do the analysis for both, shape and none
for run in [4, 2, 1]:
    indices = np.arange(0, len(catalog), 4)
    if run != 1:
        indices = np.sort(np.append(indices, [indices + i for i in range(1, run)]))

    data_complete = catalog[indices]
    input_g1 = intrinsic_g1_complete[indices]
    input_g2 = intrinsic_g2_complete[indices]

    data_complete_ref = ray.put(data_complete)
    input_g1_ref = ray.put(input_g1)
    input_g2_ref = ray.put(input_g2)

    columns = []

    ny_tiles = [2, 1, 1][index]
    ring_num = [2, 2, 1][index]
    every = [4, 2, 1][index]


    futures = [
        fct.one_shear_analysis.remote(m, config_ref, argv_ref, data_complete_ref, input_g1_ref, input_g2_ref, magnitudes,
                                      every, binning) for m in range(int(object_number / average_num))]
    for i in range(len(futures)):
        ready, not_ready = ray.wait(futures)

        for x in ray.get(ready)[0]:
            columns.append(x)

        futures = not_ready
        if not futures:
            break

    # Convert columns to numpy array and create the output Table
    columns = np.array(columns, dtype=float)
    columns = columns[np.lexsort((-columns[:, 9], columns[:, 0]))]

    shear_results = Table([columns[:, i] for i in range(1, 13)],
                          names=('input_g1', 'input_g2', 'meas_g1_mod', 'meas_g1_mod_err', 'meas_g1_mod_err_err',
                                 'meas_g2_mod',
                                 'meas_g2_mod_err', 'meas_g2_mod_err_err', 'n_pairs', 'mag', 'intrinsic_g1',
                                 'intrinsic_g2'))


    stop = timeit.default_timer()
    logger.info("Runtime: %.2f seconds", stop - start)

    results_name = "results_{}_{}_{}_{}.dat".format(object_number, average_num, ny_tiles, ring_num)

    ascii.write(shear_results, subfolder + "/" + results_name, overwrite=True)

    index += 1

ray.shutdown()
