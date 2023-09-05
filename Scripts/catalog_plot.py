"""
This function reads in the shear catalog from the linear fit on random positions, calculates uncertainties,
and fits biases.

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
import configparser
from astropy.io import ascii
import timeit
import os

config = configparser.ConfigParser()
config.read('config_rp.ini')

simulation = config['SIMULATION']
timings = config['TIMINGS']

noise_plus_meas = float(timings["noise_plus_meas"])
scene_creation = float(timings["scene_creation"])
mag_bins = int(simulation["bins_mag"])

if sys.argv[5] == "GEMS":
    bin_type = "mag_gems"
elif sys.argv[5] == "MAG_AUTO":
    bin_type = "mag_auto"

min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

shear_bins = int(simulation["shear_bins"])

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])
REPS = int(simulation["reps_for_improvements"])


def linear_function(x, a, b):
    return a * x + b


def bootstrap(array, weights, n):
    """
    Takes an array and returns the standard deviation estimated via bootstrap
    """
    array = np.array(array).flatten()
    weights = np.array(weights).flatten()
    # print(weights)
    sum_weights = np.sum(weights)
    indices = np.random.choice(np.arange(len(array)), size=(n, len(array)))

    bootstrap = np.take(array, indices, axis=0).reshape(n, -1)
    weights = np.take(weights, indices, axis=0).reshape(n, -1)

    if sum_weights == 0:
        error = -1
        error_error = -1
    else:
        error = np.std(
            np.average(bootstrap[np.sum(weights, axis=1) != 0], weights=weights[np.sum(weights, axis=1) != 0], axis=1))

        # Calculate the uncertainty of the uncertainty by splitting the sample in 10 and using the standard deviation
        split_bootstrap = np.array_split(bootstrap, 10)
        split_weights = np.array_split(weights, 10)
        error_error = np.std([np.std(
            np.average(split_bootstrap[i][np.sum(split_weights[i], axis=1) != 0],
                       weights=split_weights[i][np.sum(split_weights[i], axis=1) != 0], axis=1)) for i in range(10)])

    return error, error_error / np.sqrt(10)


file_number = int(sys.argv[1])
galaxy_num = 1 # Normalise this
path = sys.argv[2] + "/"
sim_size = int(sys.argv[3])
subfolder = sys.argv[4]

data2_compl = []
magnitudes_list = [min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)]

# Read in the catalog with astropy and numpy
data_complete = ascii.read(subfolder + 'analysis.dat')
numpy_data = np.genfromtxt(subfolder + 'analysis.dat', skip_header=1, usecols=(0, 4, 5, 6, 7))
print("Finished read in")
start = timeit.default_timer()
bootstrapping = 0
errors_m = [[[[] for _ in range(mag_bins + 1)] for _ in range(file_number)] for _ in range(3)]
errors_c = [[[[] for _ in range(mag_bins + 1)] for _ in range(file_number)] for _ in range(3)]
for reps in range(REPS):
    print(f"{reps + 1}/{REPS}")
    data_compl = []

    rand_ind = np.random.randint(0, file_number, size=file_number)
    # rand_ind = np.repeat(0, file_number)

    # The permutation, which is saved should be the "normal" analysis with all runs included in consecutive order
    if reps == REPS - 1:
        rand_ind = np.array([i for i in range(file_number)])

    for run in range(file_number):

        data_compl.append(
            numpy_data[rand_ind[run] * (mag_bins + 1) * shear_bins * 4:(rand_ind[run] + 1) * (
                        mag_bins + 1) * shear_bins * 4].reshape(
                numpy_data[rand_ind[run] * (mag_bins + 1) * shear_bins * 4:(rand_ind[run] + 1) * (
                            mag_bins + 1) * shear_bins * 4].size, 1))
        data = np.hstack(data_compl)

        if run >= int(simulation["skip_first_lf"]):
            shape_noise_all = []
            both_noise_all = []

            uncertainties = []
            uncertainties_errors = []
            start_bootstrap = timeit.default_timer()
            magnitudes_list = [min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)]

            # INDIVIDUAL UNCERTAINTIES
            for j in range(4):
                for i in range(shear_bins):
                    for mag in range(mag_bins + 1):
                        bootstrap_array = np.take(data_complete["mean_g1"][
                                                      (data_complete["shear_index"] == i) & (
                                                              data_complete["cancel_index"] == j) & (
                                                                  data_complete[bin_type] == magnitudes_list[mag])],
                                                  rand_ind[:(run + 1)])

                        weights = np.take(data_complete["weight"][
                                              (data_complete["shear_index"] == i) & (
                                                      data_complete["cancel_index"] == j) & (
                                                          data_complete[bin_type] == magnitudes_list[mag])],
                                          rand_ind[:(run + 1)])

                        bootstrap_results = bootstrap(bootstrap_array, weights, BOOTSTRAP_REPETITIONS)
                        uncertainties.append(bootstrap_results[0])
                        uncertainties_errors.append(bootstrap_results[1])

            # SHAPE UNCERTAINTY
            for i in range(shear_bins):
                for j in range(mag_bins + 1):
                    try:
                        shape_noise_av = np.take(
                            np.average([data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                 & (data_complete["cancel_index"] == 0) & (
                                                                         data_complete[bin_type] ==
                                                                         magnitudes_list[j])],
                                        data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                 & (
                                                                         data_complete[
                                                                             "cancel_index"] == 1) & (
                                                                         data_complete[bin_type] ==
                                                                         magnitudes_list[j])]], weights=
                                       [data_complete["weight"][(data_complete["shear_index"] == i)
                                                                & (data_complete["cancel_index"] == 0) & (
                                                                        data_complete[bin_type] == magnitudes_list[
                                                                    j])],
                                        data_complete["weight"][(data_complete["shear_index"] == i)
                                                                & (
                                                                        data_complete["cancel_index"] == 1) & (
                                                                        data_complete[bin_type] ==
                                                                        magnitudes_list[j])]], axis=0),
                            rand_ind[:(run + 1)])

                    except ZeroDivisionError:
                        shape_noise_av = np.take(
                            np.average([data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                 & (data_complete["cancel_index"] == 0) & (
                                                                         data_complete[bin_type] ==
                                                                         magnitudes_list[j])],
                                        data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                 & (
                                                                         data_complete[
                                                                             "cancel_index"] == 1) & (
                                                                         data_complete[bin_type] ==
                                                                         magnitudes_list[j])]], axis=0),
                            rand_ind[:(run + 1)])

                    weights = np.take(np.sum(
                        [data_complete["weight"][(data_complete["shear_index"] == i)
                                                 & (data_complete["cancel_index"] == 0) & (
                                                         data_complete[bin_type] == magnitudes_list[j])],
                         data_complete["weight"][(data_complete["shear_index"] == i)
                                                 & (
                                                         data_complete["cancel_index"] == 1) & (
                                                         data_complete[bin_type] ==
                                                         magnitudes_list[j])]], axis=0), rand_ind[:(run + 1)])

                    try:
                        shape_noise_all.append(
                            np.average(shape_noise_av, weights=weights) - float(data_complete["g1"][
                                (data_complete["shear_index"] == i) & (data_complete["scene_index"] == 0) & (
                                        data_complete["cancel_index"] == 0) & (
                                        data_complete[bin_type] == magnitudes_list[j])]))
                    except ZeroDivisionError:
                        shape_noise_all.append(0)

                    bootstrap_results = bootstrap(shape_noise_av, weights, BOOTSTRAP_REPETITIONS)
                    uncertainties.append(bootstrap_results[0])
                    uncertainties_errors.append(bootstrap_results[1])



            # BOTH UNCERTAINTY
            for i in range(shear_bins):
                for j in range(mag_bins + 1):
                    try:
                        both_noise_av = np.take(np.average([data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (data_complete[
                                                                                            "cancel_index"] == 0) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])],
                                                            data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (
                                                                                             data_complete[
                                                                                                 "cancel_index"] == 1) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])],
                                                            data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (data_complete[
                                                                                            "cancel_index"] == 2) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])],
                                                            data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (
                                                                                             data_complete[
                                                                                                 "cancel_index"] == 3) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])]],
                                                           weights=
                                                           [data_complete["weight"][(data_complete["shear_index"] == i)
                                                                                    & (data_complete[
                                                                                           "cancel_index"] == 0) & (
                                                                                            data_complete[bin_type] ==
                                                                                            magnitudes_list[j])],
                                                            data_complete["weight"][(data_complete["shear_index"] == i)
                                                                                    & (
                                                                                            data_complete[
                                                                                                "cancel_index"] == 1) & (
                                                                                            data_complete[bin_type] ==
                                                                                            magnitudes_list[j])],
                                                            data_complete["weight"][(data_complete["shear_index"] == i)
                                                                                    & (data_complete[
                                                                                           "cancel_index"] == 2) & (
                                                                                            data_complete[bin_type] ==
                                                                                            magnitudes_list[j])],
                                                            data_complete["weight"][(data_complete["shear_index"] == i)
                                                                                    & (
                                                                                            data_complete[
                                                                                                "cancel_index"] == 3) & (
                                                                                            data_complete[bin_type] ==
                                                                                            magnitudes_list[j])]],
                                                           axis=0), rand_ind[:(run + 1)])
                    except ZeroDivisionError:
                        both_noise_av = np.take(np.average([data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (data_complete[
                                                                                            "cancel_index"] == 0) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])],
                                                            data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (
                                                                                             data_complete[
                                                                                                 "cancel_index"] == 1) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])],
                                                            data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (data_complete[
                                                                                            "cancel_index"] == 2) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])],
                                                            data_complete["mean_g1"][(data_complete["shear_index"] == i)
                                                                                     & (
                                                                                             data_complete[
                                                                                                 "cancel_index"] == 3) & (
                                                                                             data_complete[bin_type] ==
                                                                                             magnitudes_list[j])]],
                                                           axis=0), rand_ind[:(run + 1)])

                    weights = np.take(np.sum(
                        [data_complete["weight"][(data_complete["shear_index"] == i)
                                                 & (data_complete["cancel_index"] == 0) & (
                                                         data_complete[bin_type] == magnitudes_list[j])],
                         data_complete["weight"][(data_complete["shear_index"] == i)
                                                 & (
                                                         data_complete["cancel_index"] == 1) & (
                                                         data_complete[bin_type] ==
                                                         magnitudes_list[j])],
                         data_complete["weight"][(data_complete["shear_index"] == i)
                                                 & (data_complete["cancel_index"] == 2) & (
                                                         data_complete[bin_type] == magnitudes_list[j])],
                         data_complete["weight"][(data_complete["shear_index"] == i)
                                                 & (
                                                         data_complete["cancel_index"] == 3) & (
                                                         data_complete[bin_type] ==
                                                         magnitudes_list[j])]], axis=0), rand_ind[:(run + 1)])

                    try:
                        both_noise_all.append(
                            np.average(both_noise_av, weights=weights) - float(data_complete["g1"][
                                (data_complete["shear_index"] == i) & (data_complete["scene_index"] == 0) & (
                                        data_complete["cancel_index"] == 0) & (
                                        data_complete[bin_type] == magnitudes_list[j])]))
                    except ZeroDivisionError:
                        both_noise_all.append(0)

                    bootstrap_results = bootstrap(both_noise_av, weights, BOOTSTRAP_REPETITIONS)
                    uncertainties.append(bootstrap_results[0])
                    uncertainties_errors.append(bootstrap_results[1])

            uncertainties = np.array(uncertainties)
            uncertainties_errors = np.array(uncertainties_errors)
            bootstrapping += timeit.default_timer() - start_bootstrap
            new_data = []

            weights = data[3]
            for i in range(len(data)):
                if (i - 3) % 5 == 0:
                    try:
                        weights = data[i + 5]
                    except IndexError:
                        pass
                    finally:
                        new_data.append(np.sum(data[i]))
                else:
                    if np.sum(weights) == 0:
                        new_data.append(np.average(data[i]))
                    else:
                        new_data.append(np.average(data[i], weights=weights))

            data_run = np.array(new_data).reshape(-1, 5)
            for m in range(mag_bins + 1):
                data = data_run[m::(mag_bins + 1)]
                # print(data.reshape(-1, 100))

                shape_noise = np.array(shape_noise_all[m::(mag_bins + 1)]).flatten()
                both_noise = np.array(both_noise_all[m::(mag_bins + 1)]).flatten()

                shape_noise_err = np.array(
                    uncertainties[(mag_bins + 1) * 4 * shear_bins:(mag_bins + 1) * 5 * shear_bins][
                    m::(mag_bins + 1)])
                both_noise_err = np.array(
                    uncertainties[(mag_bins + 1) * 5 * shear_bins:(mag_bins + 1) * 6 * shear_bins][
                    m::(mag_bins + 1)])

                shape_noise_err_err = np.array(
                    uncertainties_errors[(mag_bins + 1) * 4 * shear_bins:(mag_bins + 1) * 5 * shear_bins][
                    m::(mag_bins + 1)])
                both_noise_err_err = np.array(
                    uncertainties_errors[(mag_bins + 1) * 5 * shear_bins:(mag_bins + 1) * 6 * shear_bins][
                    m::(mag_bins + 1)])

                if reps == REPS - 1:
                    mm = 1 / 25.4  # millimeter in inches
                    fig = plt.figure(figsize=(176 * mm, 88 * mm))
                    ax = fig.add_subplot(111)

                    names = ["normal", "rotated", "normal pixel", "rotated pixel"]

                # INDIVIDUAL FITS
                i = 0
                if len(np.where((data[:, 1][i * shear_bins:(i + 1) * shear_bins] != -1) & (
                        data[:, 2][i * shear_bins:(i + 1) * shear_bins] != 0))[0]) > 3:
                    filter = (data[:, 1][i * shear_bins:(i + 1) * shear_bins] != -1) & (
                            data[:, 2][i * shear_bins:(i + 1) * shear_bins] != 0)
                    deviation = data[:, 1][i * shear_bins:(i + 1) * shear_bins][filter] - \
                                data[:, 0][i * shear_bins:(i + 1) * shear_bins][filter]
                    if reps == REPS - 1:
                        ax.errorbar(data[:, 0][i * shear_bins:(i + 1) * shear_bins][filter] + i * 0.001, deviation,
                                    uncertainties[0:(mag_bins + 1) * 4 * shear_bins][m::(mag_bins + 1)][
                                    i * shear_bins:(i + 1) * shear_bins][filter], fmt="+", capsize=2, color="C0",
                                    label="no cancel")

                    try:
                        popt, pcov = curve_fit(linear_function, data[:, 0][i * shear_bins:(i + 1) * shear_bins][filter],
                                               deviation,
                                               sigma=
                                               uncertainties[0:(mag_bins + 1) * 4 * shear_bins][m::(mag_bins + 1)][
                                               i * shear_bins:(i + 1) * shear_bins][filter], absolute_sigma=True)

                        error = np.sqrt(np.diag(pcov))

                        errors_m[0][run][m].append(error[0])
                        errors_c[0][run][m].append(error[1])
                        popt_plus, pcov_plus = curve_fit(linear_function,
                                                         data[:, 0][i * shear_bins:(i + 1) * shear_bins][filter],
                                                         deviation,
                                                         sigma=
                                                         uncertainties[0:(mag_bins + 1) * 4 * shear_bins][
                                                         m::(mag_bins + 1)][i * shear_bins:(i + 1) * shear_bins][
                                                             filter] +
                                                         uncertainties_errors[0:(mag_bins + 1) * 4 * shear_bins][
                                                         m::(mag_bins + 1)][i * shear_bins:(i + 1) * shear_bins][
                                                             filter], absolute_sigma=True)

                        error_plus = np.sqrt(np.diag(pcov_plus))

                        popt_minus, pcov_minus = curve_fit(linear_function,
                                                           data[:, 0][i * shear_bins:(i + 1) * shear_bins][filter],
                                                           deviation,
                                                           sigma=
                                                           uncertainties[0:(mag_bins + 1) * 4 * shear_bins][
                                                           m::(mag_bins + 1)][i * shear_bins:(i + 1) * shear_bins][
                                                               filter] -
                                                           uncertainties_errors[0:(mag_bins + 1) * 4 * shear_bins][
                                                           m::(mag_bins + 1)][i * shear_bins:(i + 1) * shear_bins][
                                                               filter], absolute_sigma=True)

                        error_minus = np.sqrt(np.diag(pcov_minus))
                    except RuntimeError:
                        print(data[:, 0][i * shear_bins:(i + 1) * shear_bins][filter], deviation,
                              uncertainties[0:(mag_bins + 1) * 4 * shear_bins][m::(mag_bins + 1)][
                              i * shear_bins:(i + 1) * shear_bins][filter])
                        popt = (1, 1)
                        error = (1, 1)
                        error_plus = (1, 1)
                        error_minus = (-1, -1)

                    with open(path + "output/rp_simulations/fits.txt", "a") as file:
                        file.write("%s\t %d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %d\t %.1f\t %.7f\t %.7f\n" %
                                   ("none", sim_size, galaxy_num, run,
                                    popt[0], error[0], popt[1], error[1],
                                    shear_bins * galaxy_num * (run + 1) * (1 + noise_plus_meas)
                                    + shear_bins * (run + 1) * scene_creation, magnitudes_list[m],
                                    np.std(errors_m[0][run][m]), np.std(errors_c[0][run][m])))

                    r = deviation - linear_function(data[:, 0][i * shear_bins:(i + 1) * shear_bins][filter], *popt)

                    chisq = np.sum((r / data[:, 2][i * shear_bins:(i + 1) * shear_bins][filter] ** 2))
                    chisq_red = chisq / (len(r) - 2)
                    if reps == REPS - 1:
                        ax.plot(data[:, 0][0:shear_bins], linear_function(data[:, 0][0:shear_bins], *popt), color="C0")

                else:
                    with open(path + "output/rp_simulations/fits.txt", "a") as file:
                        file.write("%s\t %d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %d\t %.1f\t %.7f\t %.7f\n" %
                                   ("none", sim_size, galaxy_num, run,
                                    1, 1, 1, 1,
                                    shear_bins * galaxy_num * (run + 1) * (1 + noise_plus_meas)
                                    + shear_bins * (run + 1) * scene_creation, magnitudes_list[m], 0.0, 0.0))

                # SHAPE NOISE FITS
                if len(np.where((shape_noise_err != -1) & (shape_noise_err != 0))[0]) > 3:
                    popt, pcov = curve_fit(linear_function,
                                           data[:, 0][0:shear_bins][(shape_noise_err != 0) & (shape_noise_err != -1)],
                                           shape_noise[(shape_noise_err != -1) & (shape_noise_err != 0)],
                                           sigma=shape_noise_err[(shape_noise_err != -1) & (shape_noise_err != 0)],
                                           absolute_sigma=True)

                    error = np.sqrt(np.diag(pcov))
                    errors_m[1][run][m].append(error[0])
                    errors_c[1][run][m].append(error[1])
                    popt_plus, pcov_plus = curve_fit(linear_function,
                                                     data[:, 0][0:shear_bins][
                                                         (shape_noise_err != 0) & (shape_noise_err != -1)],
                                                     shape_noise[(shape_noise_err != -1) & (shape_noise_err != 0)],
                                                     sigma=shape_noise_err[
                                                               (shape_noise_err != -1) & (shape_noise_err != 0)] +
                                                           shape_noise_err_err[
                                                               (shape_noise_err != -1) & (shape_noise_err != 0)],
                                                     absolute_sigma=True)

                    error_plus = np.sqrt(np.diag(pcov_plus))

                    popt_minus, pcov_minus = curve_fit(linear_function, data[:, 0][0:shear_bins][
                        (shape_noise_err != 0) & (shape_noise_err != -1)],
                                                       shape_noise[(shape_noise_err != -1) & (shape_noise_err != 0)],
                                                       sigma=shape_noise_err[
                                                                 (shape_noise_err != -1) & (shape_noise_err != 0)] -
                                                             shape_noise_err_err[
                                                                 (shape_noise_err != -1) & (shape_noise_err != 0)],
                                                       absolute_sigma=True)

                    error_minus = np.sqrt(np.diag(pcov_minus))

                    if reps == REPS - 1:
                        ax.errorbar(data[:, 0][0:shear_bins][(shape_noise_err != 0) & (shape_noise_err != -1)] + 0.0004,
                                    shape_noise[(shape_noise_err != -1) & (shape_noise_err != 0)],
                                    shape_noise_err[(shape_noise_err != -1) & (shape_noise_err != 0)], fmt="+",
                                    capsize=2,
                                    label="shape", color="C1")
                        ax.plot(data[:, 0][0:shear_bins], linear_function(data[:, 0][0:shear_bins], *popt), color="C1")

                    with open(path + "output/rp_simulations/fits.txt", "a") as file:
                        file.write("%s\t %d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %d\t %.1f\t %.7f\t %.7f\n" %
                                   ("shape", sim_size, galaxy_num, run,
                                    popt[0], error[0], popt[1], error[1],
                                    shear_bins * galaxy_num * (run + 1) * (2 + 2 * noise_plus_meas)
                                    + (run + 1) * shear_bins * 2 * scene_creation, magnitudes_list[m],
                                    np.std(errors_m[1][run][m]), np.std(errors_c[1][run][m])))
                else:

                    with open(path + "output/rp_simulations/fits.txt", "a") as file:
                        file.write("%s\t %d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %d\t %.1f\t %.7f\t %.7f\n" %
                                   ("shape", sim_size, galaxy_num, run,
                                    1, 1, 1, 1,
                                    shear_bins * galaxy_num * (run + 1) * (2 + 2 * noise_plus_meas)
                                    + (run + 1) * shear_bins * 2 * scene_creation, magnitudes_list[m], 0.0, 0.0))

                # BOTH NOISE FITS
                if len(np.where((both_noise_err != -1) & (both_noise_err != 0))[0]) > 3:
                    popt, pcov = curve_fit(linear_function,
                                           data[:, 0][0:shear_bins][(both_noise_err != 0) & (both_noise_err != -1)],
                                           both_noise[(both_noise_err != -1) & (both_noise_err != 0)],
                                           sigma=both_noise_err[(both_noise_err != -1) & (both_noise_err != 0)],
                                           absolute_sigma=True)

                    error = np.sqrt(np.diag(pcov))
                    errors_m[2][run][m].append(error[0])
                    errors_c[2][run][m].append(error[1])
                    popt_plus, pcov_plus = curve_fit(linear_function,
                                                     data[:, 0][0:shear_bins][
                                                         (both_noise_err != 0) & (both_noise_err != -1)],
                                                     both_noise[(both_noise_err != -1) & (both_noise_err != 0)],
                                                     sigma=both_noise_err[(both_noise_err != -1) & (
                                                             both_noise_err != 0)] + both_noise_err_err[
                                                               (both_noise_err != -1) & (both_noise_err != 0)],
                                                     absolute_sigma=True)

                    error_plus = np.sqrt(np.diag(pcov_plus))

                    popt_minus, pcov_minus = curve_fit(linear_function,
                                                       data[:, 0][0:shear_bins][
                                                           (both_noise_err != 0) & (both_noise_err != -1)],
                                                       both_noise[(both_noise_err != -1) & (both_noise_err != 0)],
                                                       sigma=both_noise_err[(both_noise_err != -1) & (
                                                               both_noise_err != 0)] - both_noise_err_err[
                                                                 (both_noise_err != -1) & (both_noise_err != 0)],
                                                       absolute_sigma=True)

                    error_minus = np.sqrt(np.diag(pcov_minus))
                    if reps == REPS - 1:
                        ax.errorbar(data[:, 0][0:shear_bins][(both_noise_err != 0) & (both_noise_err != -1)] + 0.0008,
                                    both_noise[(both_noise_err != -1) & (both_noise_err != 0)],
                                    both_noise_err[(both_noise_err != -1) & (both_noise_err != 0)], fmt="+", capsize=2,
                                    label="both", color="C2")
                        ax.plot(data[:, 0][0:shear_bins], linear_function(data[:, 0][0:shear_bins], *popt), color="C2")

                    with open(path + "output/rp_simulations/fits.txt", "a") as file:
                        file.write("%s\t %d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %d\t %.1f\t %.7f\t %.7f\n" %
                                   ("both", sim_size, galaxy_num, run,
                                    popt[0], error[0], popt[1], error[1],
                                    shear_bins * galaxy_num * (run + 1) * (2 + 4 * noise_plus_meas)
                                    + (run + 1) * shear_bins * scene_creation * 4, magnitudes_list[m],
                                    np.std(errors_m[2][run][m]), np.std(errors_c[2][run][m])))
                else:

                    with open(path + "output/rp_simulations/fits.txt", "a") as file:
                        file.write("%s\t %d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %d\t %.1f\t %.7f\t %.7f\n" %
                                   ("both", sim_size, galaxy_num, run,
                                    1, 1, 1, 1, shear_bins * galaxy_num * (run + 1) * (2 + 4 * noise_plus_meas)
                                    + (run + 1) * shear_bins * scene_creation * 4, magnitudes_list[m], 0.0, 0.0))

                if reps == REPS - 1:
                    ax.legend()

                    ax.set_xlabel("$g_1^\mathrm{true}$")
                    ax.set_ylabel("$g_1^\mathrm{meas}-g_1^\mathrm{true}$")
                    if not os.path.isdir(path + "output/plots/fits"):
                        os.mkdir(path + "output/plots/fits")

                    if simulation.getboolean("save_fits"):
                        fig.savefig(path + "output/plots/fits/" + f"rp_fit_{run}_{m}.pdf", dpi=300,
                                    bbox_inches='tight')
                    plt.close()

print(f"total analysis time: {timeit.default_timer() - start}")
print(f"Davon bootstrapping: {bootstrapping}")
