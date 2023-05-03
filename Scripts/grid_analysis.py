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

rep = int(sys.argv[7])
REPS = int(simulation["reps_for_improvements"])

ray.init(num_cpus=10, log_to_driver=False)

# Read in the catalogs in chunks
catalog = ascii.read(subfolder + '/shear_catalog.dat', fast_reader={'chunk_size': 100 * 1000000})
input_catalog = ascii.read(subfolder + '/input_catalog.dat', fast_reader={'chunk_size': 100 * 1000000})

catalog["cancel_index"] = np.tile([0, 1, 2, 3], int(len(catalog["galaxy_id"]) / 4))

# Sort them by galaxy id
catalog.sort('galaxy_id')
input_catalog.sort('galaxy_id')

#print(input_catalog[0:10])
catalog["intr_g1"] = np.repeat(np.take(input_catalog["intr_g1"], np.unique(catalog["galaxy_id"]).astype(int)),
                               4) * np.tile([1, -1, 1, -1], len(np.unique(catalog["galaxy_id"])))
catalog["intr_g2"] = np.repeat(np.take(input_catalog["intr_g2"], np.unique(catalog["galaxy_id"]).astype(int)),
                               4) * np.tile([1, -1, 1, -1], len(np.unique(catalog["galaxy_id"])))

catalog = catalog[(catalog["S/N"] > float(simulation["sn_cut"])) & (catalog["meas_g1"] < 5) & (catalog["meas_g1"] > -5)]

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

catalog['binned_mag'] = np.trunc(catalog[bin_type] + 0.5)  # Adding a row with the magnitudes in bins
total_gal_ids = len(np.unique(catalog['galaxy_id']))

per_time_bin = int(total_gal_ids / int(simulation["time_bins"]))

# There might be some leftover if the object number is not divisible by the number of CPU's
rest = total_gal_ids - int(simulation["time_bins"]) * per_time_bin

config_ref = ray.put(config)
argv_ref = ray.put(sys.argv)

# Do the analysis for both, shape and none
for run in [4, 2, 1]:
    columns = []

    ny_tiles = [2, 1, 1][index]
    ring_num = [2, 2, 1][index]
    every = [4, 2, 1][index]

    indices = np.repeat(np.array([i for i in range(int(simulation["time_bins"]))]), per_time_bin)
    indices = np.append(indices, np.repeat(int(simulation["time_bins"]) - 1, rest))

    unique_indices_occ = np.unique(catalog["galaxy_id"], return_counts=True)[1]

    indices_ids = np.arange(len(unique_indices_occ))

    if rep != REPS - 1:
        indices_ids = np.random.choice(indices_ids, size=len(indices_ids), replace=True)
    else:
        indices_ids = np.random.choice(indices_ids, size=len(indices_ids), replace=False)

    cum_sum = np.cumsum(unique_indices_occ)
    cum_sum = np.insert(cum_sum, 0, 0)

    indices_ids_full = np.repeat(np.take(cum_sum, indices_ids), np.take(unique_indices_occ, indices_ids))

    tmp = np.take(unique_indices_occ, indices_ids)
    adding = [i for j in range(len(indices_ids)) for i in range(tmp[j])]


    indices_ids_full = indices_ids_full + np.array(adding)

    print(indices_ids_full[:10])
    catalog_new = catalog[indices_ids_full]

    print(np.mean(catalog_new["meas_g1"]))

    catalog_new["binned_time"] = np.repeat(indices, np.take(unique_indices_occ, indices_ids))
    del indices_ids, indices_ids_full
    #print(catalog_new[:10])
    # Stopped here on friday
    catalog_new = catalog_new[catalog_new["cancel_index"] < run]
    meas_all = catalog_new.group_by(["shear_index", "binned_mag", "binned_time"])
    meas_comp = meas_all[
        "galaxy_id", "shear_index", "binned_mag", "binned_time", "meas_g1", "meas_g2", "intr_g1", "intr_g2"].groups.aggregate(
        np.mean)
    meas_weights = meas_all["galaxy_id", "binned_time", "meas_g1"].groups.aggregate(np.size)

    # Group also by galaxy id
    meas_all_bs = catalog_new.group_by(["shear_index", "binned_mag", "binned_time", "galaxy_id"])
    meas_comp_bs = meas_all_bs[
        "galaxy_id", "shear_index", "binned_mag", "binned_time", "meas_g1", "meas_g2", "intr_g1", "intr_g2"].groups.aggregate(
        np.mean)
    meas_weights_bs = meas_all_bs["galaxy_id", "binned_time", "meas_g1"].groups.aggregate(np.size)

    meas_comp_ref = ray.put(meas_comp)
    meas_weights_ref = ray.put(meas_weights)
    meas_comp_bs_ref = ray.put(meas_comp_bs)
    meas_weights_bs_ref = ray.put(meas_weights_bs)

    futures = [
        fct.one_shear_analysis.remote(m, config_ref, argv_ref, meas_comp_ref, meas_weights_ref, meas_comp_bs_ref, meas_weights_bs_ref,
                                      magnitudes) for m in range(int(object_number / average_num))]
    for i in range(len(futures)):
        ready, not_ready = ray.wait(futures)

        for x in ray.get(ready)[0]:
            columns.append(x)

        futures = not_ready
        if not futures:
            break



    # Convert columns to numpy array and create the output Table
    columns = np.array(columns, dtype=float)
    columns = columns[np.lexsort((columns[:,14], -columns[:,13], columns[:, 0]))]

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
