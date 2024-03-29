from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import logging
import galsim
import timeit
from astropy.table import QTable
from astropy.io import ascii
from scipy.optimize import curve_fit
import configparser
import pickle
import functions as fct
import ray
import datetime

bootstrap_fit = True


def linear_function(x, a, b):
    return a * x + b


subfolder = sys.argv[4]
path = sys.argv[5]
if not os.path.isdir(subfolder + "/plots"):
    os.mkdir(subfolder + "/plots")
start = timeit.default_timer()

object_number = int(sys.argv[1])
noise_repetitions = int(sys.argv[2])
num_shears = int(sys.argv[3])
if num_shears == 2:
    shears = np.array([-0.02, 0.02])
else:
    shears = np.array([-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)])

# Read Config File
config = configparser.ConfigParser()
config.read('config_grid.ini')

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("real_galaxies")

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']
timings = config["TIMINGS"]

rep = int(sys.argv[6])
REPS = int(simulation["reps_for_improvements"])
noise_plus_meas = float(timings["noise_plus_meas"])

if simulation["bin_type"] == "GEMS":
    bin_type = "mag_inp"
elif simulation["bin_type"] == "MEAS":
    bin_type = "mag_meas"

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])

mag_min = float(simulation["min_mag"])
mag_max = float(simulation["max_mag"])

mag_bins = int(simulation["bins_mag"])
time_bins = int(simulation["time_bins"])

data_complete = ascii.read(subfolder + "/shear_catalog.dat", fast_reader={'chunk_size': 100 * 1000000})

data_complete = Table(data_complete, masked=True, copy=False)

data_complete = data_complete.group_by("galaxy_id")  # Make sure that it is sorted

# ---- SN CUT
SN_CUT = float(simulation["sn_cut"])
data_complete["meas_g1"].mask = (data_complete["S/N"] <= SN_CUT) | (data_complete["meas_g1"] == 0)

print(100 * len(data_complete["meas_g1"][(data_complete["meas_g1"] <= -5) | (data_complete["meas_g1"] >= 5)]) / len(
    data_complete["meas_g1"]))
# ----- Outlier detection
data_complete["meas_g1"].mask = data_complete["meas_g1"].mask | (data_complete["meas_g1"] >= 5) | (
            data_complete["meas_g1"] <= -5)

# print(data_complete)
if mag_bins != 0:
    magnitudes = [
        float(simulation["min_mag"]) + k * (float(simulation["max_mag"]) - float(simulation["min_mag"])) / (mag_bins)
        for k in range(mag_bins + 1)]
else:
    magnitudes = [float(simulation["max_mag"])]  # If no binning at all

indices_base = np.arange(0, len(data_complete), 2*num_shears)

if rep != REPS-1:
    indices_base = np.random.choice(indices_base, size=len(indices_base), replace=True)

indices = np.repeat(indices_base, 2*num_shears)
indices = indices + np.repeat(np.array([i for i in range(2*num_shears)]), len(indices_base)).reshape(2*num_shears, -1).T.flatten()

data_used = data_complete[indices]
for z in range(time_bins):
    for mag in range(mag_bins + 1):
        print(mag)
        if mag == mag_bins:
            upper_limit = mag_max
            lower_limit = mag_min
        else:
            upper_limit = magnitudes[mag + 1]
            lower_limit = magnitudes[mag]

        array = data_used[(data_used[bin_type] > lower_limit) & (data_used[bin_type] < upper_limit)]

        array = array[0: int(len(array) / (z + 1)) - int(len(array) / (z + 1)) % (num_shears * 2)]
        selection_bias = fct.bootstrap_puyol_grid_galaxy_new(array["meas_g1"], BOOTSTRAP_REPETITIONS, shears[1] - shears[0],
                                                             num_shears,
                                                             int(simulation["num_cores"]))

        sel_bias_err = selection_bias[5]
        sel_bias_err_err = selection_bias[3]
        c_bias_err_err = selection_bias[4]


        if bootstrap_fit:
            # ------ BOOTSTRAP THE FIT FOR THE COMPARISON RM - LF ----------------#
            gal_number = len(array["meas_g1"]) / num_shears
            start_fit = timeit.default_timer()

            popt_all = []
            for k in range(BOOTSTRAP_REPETITIONS):
                loop_start = timeit.default_timer()
                tmp = np.arange(0, int(len(array["meas_g1"]))).reshape(-1, 2 * num_shears)
                indices = tmp[np.random.choice(np.arange(0, len(tmp)), size=len(tmp))].reshape(-1, num_shears)

                output_shears = np.mean(array["meas_g1"][indices], axis=0)

                # Do the fitting
                deviation = output_shears - shears
                popt, pcov = curve_fit(linear_function, shears, \
                                       deviation)

                popt_all.append(popt)

            error_fit_m_large = np.std(np.array(popt_all)[:, 0])
            error_fit_c_large = np.std(np.array(popt_all)[:, 1])

        output_shear = []
        output_err = []
        # ------ FITTING FOR COMPARISON ----------------#
        for o in range(num_shears):
            if len(array["meas_g1"][o::num_shears]) != 0:

                output_shear.append(np.average(array["meas_g1"][o::num_shears]))
                output_err.append(np.std(array["meas_g1"][o::num_shears])
                                  / np.sqrt(len(array["meas_g1"][o::num_shears])))
            else:
                output_shear.append(-1)
                output_err.append(-1)

        filter = np.where(np.array(output_shear) != -1)[0]
        if len(filter) >= 2:
            deviation = np.array(output_shear)[filter] - np.array(shears)[filter]
            popt, pcov = curve_fit(linear_function, np.array(shears)[filter], \
                                   deviation, sigma=np.array(output_err)[filter], absolute_sigma=True)

            error = np.sqrt(np.diag(pcov))
        else:
            popt = [-1, -1]
            error = [-1, -1]

        if rep == REPS - 1:
            mm = 1 / 25.4
            fig, ax = plt.subplots(figsize=(88 * mm, 88 * mm))
            ax.errorbar(np.array(shears)[filter], deviation, np.array(output_err)[filter], fmt="+--", markersize=5,
                        capsize=2, elinewidth=0.5)
            ax.plot(np.linspace(-0.1, 0.1, 10), linear_function(np.linspace(-0.1, 0.1, 10), *popt), alpha=0.7)
            ax.set_xlabel("$g_1^t$")
            ax.set_ylabel("$<g_1^{obs}>-g_1^t$")
            fig.savefig(subfolder + f"/plots/{z}_{mag}.png", dpi=200, bbox_inches="tight")
            plt.close()
            
        meas_1 = array.copy()["meas_g1"]
        meas_1.mask[(num_shears - 1)::num_shears] = True

        meas_2 = array.copy()["meas_g1"]
        meas_2.mask[::num_shears] = True

        average_meas_1 = np.average(meas_1)
        average_meas_2 = np.average(meas_2)

        selection_bias_old_def = (average_meas_2 - average_meas_1) / (shears[1] - shears[0]) - 1
        sel_bias_err_old_def = selection_bias[1]
        c_bias_err = selection_bias[2]
        selection_bias = selection_bias[0]

        mask = (meas_1 == 0) | (meas_2 == 0)

        # MASK OUT THE MEASUREMENTS WHERE ONLY ONE COULD BE MEASURED
        mask_solo = ~((meas_1 == 0) ^ (meas_2 == 0))

        meas_1_solo = np.ma.masked_array(meas_1, mask=mask_solo).compressed()
        meas_2_solo = np.ma.masked_array(meas_2, mask=mask_solo).compressed()

        meas_1_solo = meas_1_solo[meas_1_solo != 0]
        meas_2_solo = meas_2_solo[meas_2_solo != 0]

        R11 = array["R11"][::num_shears]
        R11_err = array["R11_err"][::num_shears]
        R11_len = array["R11_len"][::num_shears]

        alpha = array["alpha"][::num_shears]
        alpha_len = array["alpha_len"][::num_shears]
        alpha_err = array["alpha_err"][::num_shears]

        c_bias = fct.weighted_avg_and_std(alpha, weights=alpha_len)

        # Distinguish between solo measurements and complete measurements if you like
        if len(meas_1_solo) != 0 and len(meas_2_solo) != 0:
            # CALCULATE INDIVIDUAL BIASES FOR PAIRS AND SOLO MEASUREMENTS AND COMBINE THEM LATER ON
            solo_bias = (np.average(meas_2_solo) - np.average(meas_1_solo)) / (shears[1] - shears[0]) - 1
            solo_bias_weight = len(meas_1_solo) + len(meas_2_solo)

            error_1 = np.std(meas_1_solo) / np.sqrt(len(meas_1_solo))
            error_2 = np.std(meas_2_solo) / np.sqrt(len(meas_2_solo))

            solo_err = np.sqrt(error_1 ** 2 + error_2 ** 2) / (shears[1] - shears[0])

            pair_bias_stats = fct.weighted_avg_and_std_pujol_mp(R11, R11_len)
            pair_bias = pair_bias_stats[0]
            pair_bias_weight = np.sum(R11_len) * 2

            pair_err = pair_bias_stats[1]

            summed_bias = (solo_bias_weight * solo_bias + pair_bias_weight * pair_bias) / (
                        solo_bias_weight + pair_bias_weight)
            summed_bias_err = np.sqrt((solo_bias_weight * solo_err) ** 2 + (pair_bias_weight * pair_err) ** 2) / (
                    solo_bias_weight + pair_bias_weight)

            print(solo_bias_weight, pair_bias_weight)
        text_file = open(path + "/output/grid_simulations/fits.txt", "a")

        # Write in the output file depending on the error measurement method. Currently only "PUYOL" works
        if len(meas_1_solo) != 0 and len(meas_2_solo) != 0:
            text_file.write(
                "%d\t %d\t %.6f\t %.6f\t %.6f\t %.6f\t %d\t %.6f\t %.6f\t %s\t %.6f\t %.6f\t %d\t %.6f\t %.6f\t %.1f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\n" %
                (object_number, noise_repetitions, np.average(R11, weights=R11_len),
                 np.std(R11) / np.sqrt(np.sum(R11_len)),
                 c_bias[0],
                 c_bias[1], timeit.default_timer() - start,
                 selection_bias_old_def, sel_bias_err_old_def, sys.argv[3], summed_bias, summed_bias_err,
                 object_number * num_shears * (1 + noise_plus_meas * noise_repetitions) * 1 / (z + 1),
                 np.average(array["meas_g1"]), c_bias_err,
                 magnitudes[mag], selection_bias, sel_bias_err, sel_bias_err_err, c_bias_err_err, popt[0],
                 error_fit_m_large, popt[1], error_fit_c_large))
        else:

            text_file.write(
                "%d\t %d\t %.6f\t %.6f\t %.6f\t %.6f\t %d\t %.6f\t %.6f\t %s\t %.6f\t %.6f\t %d\t %.6f\t %.6f\t %.1f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\n" %
                (object_number, noise_repetitions, np.average(R11, weights=R11_len),
                 np.std(R11) / np.sqrt(np.sum(R11_len)),
                 c_bias[0],
                 c_bias[1], timeit.default_timer() - start,
                 selection_bias_old_def, sel_bias_err_old_def, sys.argv[3], -1, -1,
                 object_number * num_shears * (1 + noise_plus_meas * noise_repetitions) * 1 / (z + 1),
                 np.average(array["meas_g1"]), c_bias_err,
                 magnitudes[mag], selection_bias, sel_bias_err, sel_bias_err_err, c_bias_err_err, popt[0],
                 error_fit_m_large, popt[1], error_fit_c_large))

# Stop the timing
stop = timeit.default_timer()

logger.info("Runtime: %.2f seconds", stop - start)
