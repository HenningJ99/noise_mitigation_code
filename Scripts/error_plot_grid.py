# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:10:37 2021

@author: Henning
"""

import numpy as np
import matplotlib . pyplot as plt
from scipy.optimize import curve_fit
import sys
import configparser
from astropy.io import ascii
from astropy.table import Table

WITH_ERROR = False
# Define paths

path = sys.argv[1]+"/"
fit_folder = sys.argv[3]
shear_interval = sys.argv[4]

if shear_interval == "0.1":
    num_shears = 11
else:
    num_shears = 2

config = configparser.ConfigParser()
config.read('config_grid.ini')


simulation = config['SIMULATION']

mag_bins = int(simulation["bins_mag"])
time_bins = int(simulation["time_bins"])

REPS = int(simulation["reps_for_improvements"])

# Read in the relevant parameters from the temporary fit file
if sys.argv[2] == "M":
    data_complete = np.genfromtxt(path+"output/grid_simulations/tmp.txt", skip_footer=REPS * (mag_bins+1) * time_bins,
                                  usecols=(0,4,5, 10, 9, 11, 2, 3)) # 4,5, for m-bias | 6,7 for c-

    data_puyol_compl = np.genfromtxt(path+"output/grid_simulations/tmp.txt",
                                     skip_header=REPS * 3 * (mag_bins+1) * time_bins,
                                     usecols=(0, 7, 8, 12, 11, 18)) # 7,8 for m-bias | 13,14 for c-bias
elif sys.argv[2] == "C":
    data_complete = np.genfromtxt(path + "output/grid_simulations/tmp.txt", skip_footer=REPS * (mag_bins+1) * time_bins,
                                  usecols=(0, 6, 7, 10, 9, 12, 2, 3))  # 4,5, for m-bias | 6,7 for c-
    data_puyol_compl = np.genfromtxt(path + "output/grid_simulations/tmp.txt",
                                     skip_header=REPS * 3 * (mag_bins+1) * time_bins,
                                     usecols=(0, 13, 14, 12, 11, 19))  # 7,8 for m-bias | 13,14 for c-bias
test1 = np.genfromtxt(path+"output/grid_simulations/tmp.txt", skip_footer=REPS * (mag_bins+1) * time_bins,
                      usecols=(0,1,2,3))

num_meas = test1[:,0] * test1[:,2] * test1[:,3]

test2 = np.genfromtxt(path+"output/grid_simulations/tmp.txt", skip_header=REPS * 3 * (mag_bins+1) * time_bins,
                      usecols=(0, 1))

num_meas_pujol = test2[:,0] * 11





def linear_function(x, a):
    return a*x


# print(data_puyol_compl[:,3])

def sqrt_function(x, a):
    return a*x**(-0.5)


magnitude_list = [float(simulation["min_mag"]) + k*(float(simulation["max_mag"])-float(simulation["min_mag"]))/
                  (mag_bins) for k in range(mag_bins+1)]

improvements = [[[] for _ in range(mag_bins+1)] for _ in range(4)]
improvements_gal = [[[] for _ in range(mag_bins+1)] for _ in range(4)]
fit_parameters = [[[] for _ in range(mag_bins+1)] for _ in range(4)]
columns = []
for rep in range(REPS):
    for mag in range(mag_bins+1):

        no_cancel = data_complete[3 * rep * (mag_bins + 1) * time_bins: (mag_bins + 1) * time_bins * (3 * rep + 1)][
                    mag::(mag_bins + 1)]

        shape_noise = data_complete[
                      (mag_bins + 1) * time_bins * (3 * rep + 1): (mag_bins + 1) * time_bins * (3 * rep + 2)][
                      mag::(mag_bins + 1)]
        both = data_complete[(mag_bins + 1) * time_bins * (3 * rep + 2): (mag_bins + 1) * time_bins * (3 * rep + 3)][
               mag::(mag_bins + 1)]
        data_puyol = data_puyol_compl[rep * (mag_bins + 1) * time_bins:(rep + 1) * (mag_bins + 1) * time_bins][
                     mag::(mag_bins + 1)]


        runtime_data = data_complete[:, 3]
        runtime_data_gal = data_complete[:, 0]

        runtime_norm = np.max([np.max(runtime_data), np.max(data_puyol[:, 3])])

        runtime_norm_gal = np.max([np.max(runtime_data_gal) * 4, np.max(data_puyol[:, 0]) * num_shears])

        if rep == REPS-1:
            mm = 1/25.4  # millimeter in inches

            fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(88*mm, 88*mm))

        text_file = open(path+"output/grid_simulations/error_scaling.txt", "a")
        i = 0
        names = ("no cancel", "shape", "both", "RM")
        colors = ("blue", "red", "black", "orange", "purple")
        for data in (no_cancel, shape_noise, both, data_puyol):
            # data = data[data[:, 0].argsort()]
            factor_runtime = 0.0
            if i != 3:
                runtime_data = (data[:, 3] + num_meas[(mag_bins + 1) * time_bins * (3 * rep + i):(mag_bins +1) *
                                time_bins * (3 * rep + i + 1)][mag::(mag_bins+1)] * factor_runtime * data[:,3] /
                                data[:,3][0])
                runtime_data_gal = data[:, 0] * data[:, 6] * data[:, 7] * (time_bins - np.arange(time_bins)) / time_bins
            else:
                runtime_data = (data[:, 3] + num_meas_pujol[rep * (mag_bins+1) * time_bins:(rep+1) * (mag_bins + 1) *
                                                                                          time_bins][mag::(mag_bins+1)]
                                * factor_runtime * data[:,3] / data[:,3][0])
                runtime_data_gal = data[:, 0] * num_shears / np.arange(1, time_bins+1)

            if WITH_ERROR:
                popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data / runtime_norm), data[:, 2],
                                       sigma=data[:, 5], absolute_sigma=True)
                popt_gal, pcov_gal = curve_fit(linear_function, 1 / np.sqrt(runtime_data_gal / runtime_norm_gal),
                                               data[:, 2], sigma=data[:, 5], absolute_sigma=True)

            else:
                popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data / runtime_norm), data[:, 2])
                popt_gal, pcov_gal = curve_fit(linear_function, 1 / np.sqrt(runtime_data_gal / runtime_norm_gal),
                                               data[:, 2])

            error = np.sqrt(np.diag(pcov))
            error_gal = np.sqrt(np.diag(pcov_gal))

            if i == 0:
                a_no_cancel = popt[0]
                a_no_cancel_err = error[0]
                fit_parameters[i][mag].append(a_no_cancel)
                improvement = 0
                improvements[i][mag].append(improvement)
                improvement_err = 0
                factor = 0
                factor_err = 0

                a_no_cancel_gal = popt_gal[0]
                # print(a_no_cancel)
                a_no_cancel_err_gal = error_gal[0]
                improvement_gal = 0
                improvements_gal[i][mag].append(improvement_gal)
                improvement_err_gal = 0
                factor_gal = 0
                factor_err_gal = 0
            else:
                factor = a_no_cancel / (popt[0])
                fit_parameters[i][mag].append(popt[0])
                factor_err = np.sqrt((a_no_cancel_err / (popt[0]))**2 + (a_no_cancel*(error[0]) / (popt[0])**2)**2)
                improvement = factor**2
                improvements[i][mag].append(improvement)
                improvement_err = 2 * factor * factor_err

                factor_gal = a_no_cancel_gal / popt_gal[0]
                factor_err_area = np.sqrt(
                    (a_no_cancel_err_gal / (popt_gal[0])) ** 2 + (
                                a_no_cancel_gal * (error_gal[0]) / (popt_gal[0]) ** 2) ** 2)
                improvement_gal = factor_gal ** 2
                improvements_gal[i][mag].append(improvement_gal)
                improvement_err_gal = 2 * factor_gal * factor_err_gal

            if rep == REPS-1:
                if WITH_ERROR:
                    axs[0].errorbar(runtime_data / runtime_norm, data[:, 2],yerr=data[:, 5], fmt= '+',
                                    label=names[i], markersize=5, color=colors[i], capsize=2, elinewidth=0.5)
                    axs[1].errorbar(1 / np.sqrt(runtime_data / runtime_norm), data[:, 2],yerr=data[:, 5],
                                    fmt= '+', label=names[i], markersize=5, color=colors[i], capsize=2, elinewidth=0.5)
                else:
                    axs[0].plot(runtime_data / runtime_norm, data[:, 2], '+', label=names[i], markersize=5,
                                color=colors[i])
                    axs[1].plot(1 / np.sqrt(runtime_data / runtime_norm), data[:, 2], '+', label=names[i], markersize=5,
                                color=colors[i])


                axs[1].plot(1 / np.sqrt(runtime_data / runtime_norm),
                            linear_function(1 / np.sqrt(runtime_data / runtime_norm), *popt), color=colors[i],
                            alpha=0.5)
                axs[1].fill_between(1 / np.sqrt(runtime_data / runtime_norm),
                                    linear_function(1 / np.sqrt(runtime_data / runtime_norm),
                                                    np.min(fit_parameters[i][mag])),
                                    linear_function(1 / np.sqrt(runtime_data / runtime_norm),
                                                    np.max(fit_parameters[i][mag])), color=colors[i], alpha=.25)

                axs[0].plot(runtime_data / runtime_norm, sqrt_function(runtime_data / runtime_norm, *popt),
                            color=colors[i], alpha=0.5)
                axs[0].fill_between(runtime_data / runtime_norm,
                                    sqrt_function(runtime_data / runtime_norm, np.min(fit_parameters[i][mag])),
                                    sqrt_function(runtime_data / runtime_norm, np.max(fit_parameters[i][mag])),
                                    color=colors[i], alpha=.25)

                goal = sqrt_function(100, a_no_cancel_gal)
                # print((popt[0] / goal) ** 2)
                gal_equivalent = (popt_gal[0] / goal) ** 2

                columns.append([i, mag, popt[0], error[0], factor, factor_err, improvement, np.std(improvements[i][mag]),
                                improvement_gal, np.std(improvements_gal[i][mag]), gal_equivalent])

                axs[0].set_yscale('log')
                axs[0].set_xscale('log')

                axs[0].set_xlabel(r'$t_{run}$ normalized')
                if sys.argv[2] == "M":
                    axs[0].set_ylabel(r'$\sigma_\mu$')
                    axs[1].set_ylabel(r'$\sigma_\mu$')
                elif sys.argv[2] == "C":
                    axs[0].set_ylabel(r'$\sigma_c$')
                    axs[1].set_ylabel(r'$\sigma_c$')

                axs[0].legend(prop={'size': 6})
                axs[1].legend(prop={'size': 6})

                axs[1].set_xlabel(r'$1/\sqrt{t_{run}}$')

                if sys.argv[2] == "M":
                    fig.savefig(path+f'output/plots/grid_{magnitude_list[mag]}.pdf', dpi=300, bbox_inches='tight')
                elif sys.argv[2] == "C":
                    fig.savefig(path + f'output/plots/grid_{magnitude_list[mag]}_c_bias.pdf', dpi=300,
                                bbox_inches='tight')
            i += 1

columns = np.array(columns, dtype=float)
fit_results = Table([columns[:, i] for i in range(11)],
                    names=('cancellation_index', 'mag_index', 'a', 'a_err', 'factor', 'factor_err',
                                                               'improvement', 'improvement_err', 'improvement_gal',
                                                               'improvement_gal_err', 'gal_equivalent'))

ascii.write(fit_results, fit_folder + f"/error_scaling_{sys.argv[2]}.txt",
            overwrite=True)