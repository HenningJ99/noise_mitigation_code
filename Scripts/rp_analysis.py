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

# Define local paths
path = sys.argv[4]+"/"

subfolder = sys.argv[6]

config = configparser.ConfigParser()
config.read('config_rp.ini')

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])

mag_bins = int(simulation["bins_mag"])

if sys.argv[7] == "GEMS":
    bin_type = "mag_gems"
elif sys.argv[7] == "MAG_AUTO":
    bin_type = "mag_auto"

min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

random_seed = int(simulation['random_seed'])
pixel_scale = float(image['pixel_scale'])
exp_time = float(image['exp_time'])
gain = float(image['gain'])
read_noise = float(image['read_noise'])
sky = float(image['sky'])
zp = float(image['zp'])

lam = float(psf_config['lam_min'])
step_psf = float(psf_config['step_psf'])
tel_diam = float(psf_config['tel_diam'])

ellip_rms = float(simulation['ellip_rms'])
ellip_max = float(simulation['ellip_max'])

stamp_xsize = int(image['stamp_xsize'])
stamp_ysize = int(image['stamp_ysize'])
ssamp_grid = int(image['ssamp_grid'])

shear_bins = int(simulation['shear_bins'])

shear_min = -float(sys.argv[5])
shear_max = float(sys.argv[5])

galaxy_number = int(sys.argv[2])

complete_image_size = int(sys.argv[1])

total_scenes_per_shear = int(sys.argv[3])

magnitudes_list = [min_mag + k*(max_mag-min_mag)/(mag_bins) for k in range(mag_bins+1)]

# --------------------------------------- ANALYSIS -------------------------------------------------------------------
start1 = timeit.default_timer()

data_complete = ascii.read(subfolder + 'shear_catalog.dat')

# SN CUT
data_complete = data_complete[data_complete["S/N"] > float(simulation["sn_cut"])]

print(100 * len(data_complete["meas_g1"][(data_complete["meas_g1"] >= 5) | (data_complete["meas_g1"] <= -5)]) / len(data_complete["meas_g1"]))
# Outliers
data_complete = data_complete[(data_complete["meas_g1"] < 5) & (data_complete["meas_g1"] > -5)]





columns = []
# WRITE RESULTS TO FILE
for scene in range(total_scenes_per_shear):
    index = 0
    print(scene)
    # if simulation.getboolean('matching'):
    #     with open(path + f"output/rp_simulations/catalog_results_matched_"
    #                      f"{complete_image_size}_{galaxy_number}_{scene}.txt", "w") as file:
    #         for i in range(shear_bins):
    #             g1 = shear_min + i * (shear_max - shear_min) / (shear_bins - 1)
    #             for mag in range(mag_bins + 1):
    #
    #                 if mag == mag_bins:
    #                     lower_limit = min_mag
    #                     upper_limit = max_mag
    #                 else:
    #                     lower_limit = magnitudes_list[mag]
    #                     upper_limit = magnitudes_list[mag+1]
    #
    #                 array = data_complete[(data_complete["scene_index"] == scene) & (data_complete["shear_index"] == i) &
    #                                       (data_complete[bin_type] > lower_limit) & (data_complete[bin_type] < upper_limit)]
    #
    #
    #
    #                 if len(array) != 0:
    #                     galaxies = array.group_by("matching_index")
    #                     averages = [np.average(x["meas_g1"]) for x in galaxies.groups]
    #                     value = np.average(array["meas_g1"])
    #                     error = fct.bootstrap(averages, BOOTSTRAP_REPETITIONS)
    #                     length = len(array)
    #                 else:
    #                     value = -1
    #                     error = -1
    #                     length = 0
    #                 file.write("%.4f\t %.6f\t %.6f\t %d\t %.1f\n" %
    #                            (g1, value, error, length, magnitudes_list[mag]))

    with open(path + f"output/rp_simulations/catalog_results_{complete_image_size}_{galaxy_number}_{scene}.txt", "w") as file:
        for j in range(4):
            for i in range(shear_bins):
                for mag in range(mag_bins+1):
                    if mag == mag_bins:
                        lower_limit = min_mag
                        upper_limit = max_mag
                    else:
                        lower_limit = magnitudes_list[mag]
                        upper_limit = magnitudes_list[mag + 1]

                    meas = data_complete["meas_g1"][
                        (data_complete["scene_index"] == scene) & (data_complete["shear_index"] == i) & (
                                    data_complete["cancel_index"] == j) & (data_complete[bin_type] > lower_limit) & (data_complete[bin_type] < upper_limit)]


                    g1 = shear_min + i * (shear_max - shear_min) / (shear_bins-1)
                    #print(convolution_times[i])

                    if len(meas) != 0:
                        file.write("%.4f\t %.6f\t %.6f\t %d\t %.1f\n" %
                                       (g1, np.average(meas), np.std(meas) / np.sqrt(len(meas)),
                                        len(meas), magnitudes_list[mag]))
                        columns.append([g1, i, scene, j, np.average(meas), np.std(meas) / np.sqrt(len(meas)),
                                        len(meas), magnitudes_list[mag]])
                        #lf_results.add_row(column)
                    else:
                        file.write("%.4f\t %.6f\t %.6f\t %d\t %.1f\n" %
                                       (g1, -1, -1 , 0, magnitudes_list[mag]))
                        columns.append([g1, i, scene, j, -1, -1, 0, magnitudes_list[mag]])
                        #lf_results.add_row(column)

                    if i == mag_bins-1 and index == 3:
                        file.write("\n")

            index += 1

columns = np.array(columns, dtype=float)
lf_results = Table([columns[:, i] for i in range(8)], names=('g1', 'shear_index', 'scene_index', 'cancel_index', 'mean_g1', 'std_g1', 'weight', bin_type))
lf_results = lf_results.group_by('scene_index')

ascii.write(lf_results, subfolder + 'analysis.dat',
            overwrite=True)
