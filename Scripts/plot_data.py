# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 16:41:35 2021

@author: Henning

This script does the fits for the linear fit method and plots them. Syntax is
plot_data.py <input_data> <1 or 2 plots (g1 and g2)> <normal or mod>
"""
WITH_ERROR = True

import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.table import Table
from astropy.io import ascii
from scipy.optimize import curve_fit
import sys
import configparser
from functions import is_outlier
import os


def linear_function(x, a, b):
    return a * x + b


def cubic_term(x, a, b, c, d):
    return a * x + b + c * x ** 2 + d * x ** 3


def cubic_term_mirrored(x, a, b, c, d):
    return a * x + b + np.sign(x) * c * x ** 2 + np.sign(x) * d * x ** 3


path = sys.argv[4] + "/"

data_complete = ascii.read(sys.argv[1])

file_name = sys.argv[1].split('/')

read_in = file_name[-1].split('_')
objects = read_in[1]
galaxies = read_in[2]
tiles = int(read_in[3])
ring_num = int(read_in[4][0])
runtime = -1  # int(read_in[5].split('.')[0])

config = configparser.ConfigParser()
config.read('config_grid.ini')

simulation = config['SIMULATION']
timings = config["TIMINGS"]

noise_plus_meas = float(timings["noise_plus_meas"])

mag_bins = int(simulation["bins_mag"])
time_bins = int(simulation["time_bins"])

subplots = int(sys.argv[2])

if sys.argv[3] == "normal":
    used_meas = ""
else:
    used_meas = "_" + sys.argv[3]

for m in range(time_bins):
    for mag in range(mag_bins + 1):
        data = data_complete[m * (mag_bins + 1) + mag:: (mag_bins + 1) * time_bins]

        mm = 1 / 25.4  # millimeter in inches

        if subplots == 2:
            fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(176 * mm, 88 * mm))
        else:
            fig, axs = plt.subplots(figsize=(88 * mm, 88 * mm))

        if (tiles >= 2) and (ring_num >= 2):
            if tiles == 1:
                pn = 0
            else:
                pn = tiles / 2
            infostr = r'$n_{tiles} = %d, n_{gal} = %d, n_{real} = %d$, %d times shape and %d times pixel noise' % (
            int(objects), int(galaxies), int(galaxies) * ring_num * tiles, ring_num, pn)
        elif (tiles == 2) and (ring_num == 1):
            infostr = r'$n_{tiles} = %d, n_{gal} = %d, n_{real} = %d$, %d times pixel noise' % (
            int(objects), int(galaxies), int(galaxies) * ring_num * tiles, ring_num)
        elif (tiles == 1) and (ring_num == 1):
            infostr = r'$n_{tiles} = %d, n_{gal} = %d, n_{real} = %d$, no cancel' % (
            int(objects), int(galaxies), int(galaxies) * ring_num * tiles)
        else:
            infostr = r'$n_{tiles} = %d, n_{gal} = %d, n_{real} = %d$, %d times shape' % (
            int(objects), int(galaxies), int(galaxies) * ring_num * tiles, ring_num)

        if subplots == 2:
            deviation = data["meas_g1" + used_meas] - data["input_g1"]
            axs[0].errorbar(data["input_g1"][~is_outlier(deviation)], deviation[~is_outlier(deviation)], \
                            yerr=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)], fmt='+', capsize=2,
                            elinewidth=0.5)

            popt, pcov = curve_fit(linear_function, data["input_g1"][~is_outlier(deviation)],
                                   deviation[~is_outlier(deviation)], \
                                   sigma=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)],
                                   absolute_sigma=True)

            error = np.sqrt(np.diag(pcov))

            if WITH_ERROR:
                popt_plus, pcov_plus = curve_fit(linear_function, data["input_g1"][~is_outlier(deviation)],
                                                 deviation[~is_outlier(deviation)], \
                                                 sigma=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)] +
                                                       data["meas_g1" + used_meas + "_err_err"][~is_outlier(deviation)],
                                                 absolute_sigma=True)

                error_plus = np.sqrt(np.diag(pcov_plus))

                popt_minus, pcov_minus = curve_fit(linear_function, data["input_g1"][~is_outlier(deviation)],
                                                   deviation[~is_outlier(deviation)], \
                                                   sigma=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)] -
                                                         data["meas_g1" + used_meas + "_err_err"][
                                                             ~is_outlier(deviation)],
                                                   absolute_sigma=True)

                error_minus = np.sqrt(np.diag(pcov_minus))

            r = deviation[~is_outlier(deviation)] - linear_function(data["input_g1"][~is_outlier(deviation)], *popt)

            chisq = np.sum((r / data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)]) ** 2)
            chisq_red = chisq / (len(r) - 2)

            axs[0].plot(data["input_g1"], linear_function(data["input_g1"], *popt))
            axs[0].set_xlabel("$g_1^t$")
            axs[0].set_ylabel("$<g_1^{obs}>-g_1^t$")
            # axs[0].grid(True)

            textstr = '\n'.join((r'$\mu = (%.4f \pm %.4f)$' % (popt[0], error[0]),
                                 r'$c = (%.5f \pm %.5f)$' % (popt[1], error[1]),
                                 r'$\chi_{red}^2 = (%.2f)$' % (chisq_red)))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[0].text(0.3, 0.95, textstr, transform=axs[0].transAxes, fontsize=8,
                        verticalalignment='top', bbox=props)

            deviation = data["meas_g2" + used_meas] - data["input_g2"]
            axs[1].errorbar(data["input_g2"][~is_outlier(deviation)], deviation[~is_outlier(deviation)], \
                            yerr=data["meas_g2" + used_meas + "_err"][~is_outlier(deviation)], fmt='+', capsize=2,
                            elinewidth=0.5)

            popt1, pcov1 = curve_fit(linear_function, data["input_g2"][~is_outlier(deviation)],
                                     deviation[~is_outlier(deviation)], \
                                     sigma=data["meas_g2" + used_meas + "_err"][~is_outlier(deviation)],
                                     absolute_sigma=True)

            error1 = np.sqrt(np.diag(pcov1))

            if WITH_ERROR:
                popt1_plus, pcov1_plus = curve_fit(linear_function, data["input_g2"][~is_outlier(deviation)],
                                                   deviation[~is_outlier(deviation)], \
                                                   sigma=data["meas_g2" + used_meas + "_err"][~is_outlier(deviation)] +
                                                         data["meas_g2" + used_meas + "_err_err"][
                                                             ~is_outlier(deviation)],
                                                   absolute_sigma=True)

                error1_plus = np.sqrt(np.diag(pcov1_plus))

                popt1_minus, pcov1_minus = curve_fit(linear_function, data["input_g2"][~is_outlier(deviation)],
                                                     deviation[~is_outlier(deviation)], \
                                                     sigma=data["meas_g2" + used_meas + "_err"][
                                                               ~is_outlier(deviation)] -
                                                           data["meas_g2" + used_meas + "_err_err"][
                                                               ~is_outlier(deviation)],
                                                     absolute_sigma=True)

                error1_minus = np.sqrt(np.diag(pcov1_minus))

            r = deviation[~is_outlier(deviation)] - linear_function(data["input_g2"][~is_outlier(deviation)], *popt)

            chisq = np.sum((r / data["meas_g2" + used_meas + "_err"][~is_outlier(deviation)]) ** 2)
            chisq_red = chisq / (len(r) - len(popt1))

            axs[1].plot(data["input_g2"], linear_function(data["input_g2"], *popt1))
            axs[1].set_xlabel("$g_2^t$")
            axs[1].set_ylabel("$<g_2^{obs}>-g_2^t$")
            # axs[1].grid(True)
            axs[0].set_ylim(-0.005, 0.005)
            axs[1].set_ylim(-0.005, 0.005)
            textstr = '\n'.join((r'$\mu = (%.4f \pm %.4f)$' % (popt1[0], error1[0]),
                                 r'$c = (%.5f \pm %.5f)$' % (popt1[1], error1[1]),
                                 r'$\chi_{red}^2 = (%.2f)$' % (chisq_red)))
            axs[1].text(0.3, 0.95, textstr, transform=axs[1].transAxes, fontsize=8,
                        verticalalignment='top', bbox=props)


        else:

            deviation = data["meas_g1" + used_meas] - data["input_g1"]

            axs.errorbar(data["input_g1"][~is_outlier(deviation)], deviation[~is_outlier(deviation)], \
                         yerr=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)], fmt='+', capsize=2,
                         elinewidth=0.5)

            popt, pcov = curve_fit(linear_function, data["input_g1"][~is_outlier(deviation)],
                                   deviation[~is_outlier(deviation)], \
                                   sigma=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)],
                                   absolute_sigma=True)

            error = np.sqrt(np.diag(pcov))

            if WITH_ERROR:
                popt_plus, pcov_plus = curve_fit(linear_function, data["input_g1"][~is_outlier(deviation)],
                                                 deviation[~is_outlier(deviation)], \
                                                 sigma=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)] +
                                                       data["meas_g1" + used_meas + "_err_err"][~is_outlier(deviation)],
                                                 absolute_sigma=True)

                error_plus = np.sqrt(np.diag(pcov_plus))

                popt_minus, pcov_minus = curve_fit(linear_function, data["input_g1"][~is_outlier(deviation)],
                                                   deviation[~is_outlier(deviation)], \
                                                   sigma=data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)] -
                                                         data["meas_g1" + used_meas + "_err_err"][
                                                             ~is_outlier(deviation)],
                                                   absolute_sigma=True)

                error_minus = np.sqrt(np.diag(pcov_minus))

            r = deviation[~is_outlier(deviation)] - linear_function(data["input_g1"][~is_outlier(deviation)], *popt)

            chisq = np.sum((r / data["meas_g1" + used_meas + "_err"][~is_outlier(deviation)]) ** 2)
            chisq_red = chisq / (len(r) - len(popt))
            axs.plot(data["input_g1"], linear_function(data["input_g1"], *popt))
            axs.set_xlabel("$g_1^t$")
            axs.set_ylabel("$<g_1^{obs}>-g_1^t$")
            # axs.grid(True)
            if tiles == 1 and ring_num == 1:
                axs.set_title("No noise cancellations")
                label = "None"
            elif tiles == 1 and ring_num == 2:
                axs.set_title("Shape noise cancellation")
                label = "Shape"
            elif tiles == 2 and ring_num == 2:
                axs.set_title("Both noise cancellations")
                label = "Both"

            textstr = '\n'.join(
                (r'$\mu\,[10^{-2}]= %.2f \pm %.2f$' % (1e2 * popt[0], 1e2 * error[0]),
                 r'$c\,[10^{-4}] = %.2f \pm %.2f$' % (1e4 * popt[1], 1e4 * error[1]),
                 r'$\chi_{red}^2\,\,\,\,\,\,\,\,\,\,\,\,\, = %.2f$' % (chisq_red)))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            axs.text(0.4, 0.95, textstr, transform=axs.transAxes, fontsize=8,
                     verticalalignment='top', bbox=props)

        if not os.path.isdir(path + "output/plots/fits"):
            os.mkdir(path + "output/plots/fits")

        if simulation.getboolean("save_fits"):
            fig.savefig(path + f'output/plots/fits/grid_fit_{label}_{m}_{mag}.pdf', dpi=300, bbox_inches='tight')

        plt.close()
        with open(path + "output/grid_simulations/fits.txt", "a") as text_file:
            if subplots == 2:
                if WITH_ERROR:
                    text_file.write(
                        "%d \t %d \t %d \t %d \t %.5f \t %.5f \t %.6f \t %.6f\t %.5f \t %.5f \t %.6f \t %.6f\t %d\t %.6f\t %.6f\t %.6f\t %.6f\n" % \
                        (int(objects), int(galaxies), int(tiles), int(ring_num), popt[0], error[0], popt[1], error[1],
                         popt1[0], error1[0], popt1[1], error1[1], int(runtime), (error_plus[0] - error_minus[0]) / 2,
                         (error_plus[1] - error_minus[1]) / 2
                         , (error1_plus[0] - error1_minus[0]) / 2, (error1_plus[1] - error1_minus[1]) / 2))
                else:
                    text_file.write(
                        "%d \t %d \t %d \t %d \t %.5f \t %.5f \t %.6f \t %.6f\t %.5f \t %.5f \t %.6f \t %.6f\t %d\n" % \
                        (int(objects), int(galaxies), int(tiles), int(ring_num), popt[0], error[0], popt[1], error[1],
                         popt1[0], error1[0], popt1[1], error1[1], int(runtime)))
            else:
                if WITH_ERROR:
                    text_file.write(
                        "%d \t %d \t %d \t %d \t %.6f \t %.6f \t %.6f \t %.6f\t %d\t %s\t %.1f\t %d\t %.6f\t %.6f\n" % \
                        (int(objects), int(galaxies), int(tiles), int(ring_num), popt[0], error[0], popt[1], error[1],
                         int(runtime), \
                         used_meas, np.sum(data["n_pairs"]),
                         int(objects) * int(ring_num) * (1 + int(tiles) * noise_plus_meas) * 1 / (m + 1),
                         (error_plus[0] - error_minus[0]) / 2, (error_plus[1] - error_minus[1]) / 2))
                else:
                    text_file.write(
                        "%d \t %d \t %d \t %d \t %.6f \t %.6f \t %.6f \t %.6f\t %d\t %s\t %.1f\t %d\n" % \
                        (int(objects), int(galaxies), int(tiles), int(ring_num), popt[0], error[0], popt[1], error[1],
                         int(runtime), \
                         used_meas, np.sum(data["n_pairs"]),
                         int(objects) * int(ring_num) * (1 + int(tiles) * noise_plus_meas) * 1 / (m + 1)))
