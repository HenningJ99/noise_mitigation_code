# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:10:37 2021

This function reads in the temporary file with the fits and fits the uncertainty behaviour with time.

It outputs the runtime improvements.

@author: Henning
"""
WITH_ERROR = False

import numpy as np
import matplotlib . pyplot as plt
from scipy.optimize import curve_fit
import configparser
import sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter)

run_lf = int(sys.argv[1])
run_rm = int(sys.argv[2])
path = sys.argv[3]+"/"
print(run_lf, run_rm)

config = configparser.ConfigParser()
config.read('config_rp.ini')

simulation = config['SIMULATION']

REPS = int(simulation["reps_for_improvements"])

# How many runs to skip for fit methods
SKIP_FIRST_LF = int(simulation["skip_first_lf"])
PLOT_EVERY_LF = int(simulation["plot_every_lf"])
# How many runs to skip for RM
SKIP_FIRST_RM = int(simulation["skip_first_rm"])


mag_bins = int(simulation["bins_mag"])
shear_bins = int(simulation["shear_bins"])

pujol_every = int(simulation["puj_analyse_every"])
min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

improvements = [[[] for _ in range(mag_bins+1)] for _ in range(4)]
fit_parameters = [[[] for _ in range(mag_bins+1)] for _ in range(4)]
improvements_area = [[[] for _ in range(mag_bins+1)] for _ in range(4)]
for reps in range(REPS):
    # Read in the columns with relevant information from the temporary file
    names_list = np.genfromtxt(path+"output/rp_simulations/tmp_rm.txt", skip_footer=int(REPS * (mag_bins+1) * run_rm / pujol_every + 3 * (run_lf - SKIP_FIRST_LF) * (mag_bins+1) * (REPS - (reps+1))),
                               dtype="unicode", skip_header=int(3 * (mag_bins+1) * (run_lf - SKIP_FIRST_LF) * reps))[:,0]

    if sys.argv[4] == "C":
        data_complete = np.genfromtxt(path+"output/rp_simulations/tmp_rm.txt", skip_footer=int(REPS * (mag_bins+1) * run_rm / pujol_every + 3 * (run_lf - SKIP_FIRST_LF) * (mag_bins+1) * (REPS - (reps+1))), usecols=(0,6,7,8,9,1,2,3,11)
                                      , skip_header=int(3 * (mag_bins+1) * (run_lf - SKIP_FIRST_LF) * reps))

        if sys.argv[5] == "0.02":
            data_puyol_compl = np.genfromtxt(path+"output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (run_lf - SKIP_FIRST_LF) * (mag_bins + 1) * REPS + (mag_bins + 1) * SKIP_FIRST_RM + (mag_bins + 1) * run_rm / pujol_every * reps),
                                             usecols=(0, 18, 19, 11, 24,1,2,3, 29), skip_footer=int((mag_bins+1) * (REPS - (reps+1)) * run_rm / pujol_every))
            # data_puyol_compl[:, 3] = data_puyol_compl[:, 3] * 2 / 11

        elif sys.argv[5] == "0.1":
            data_puyol_compl = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (run_lf - SKIP_FIRST_LF) * (mag_bins + 1) * REPS + (mag_bins + 1) * SKIP_FIRST_RM + (mag_bins + 1) * run_rm / pujol_every * reps),
                                             usecols=(0, 18, 19, 11, 24, 1, 2, 3, 27), skip_footer=int((mag_bins+1) * (REPS - (reps+1)) * run_rm / pujol_every))
    elif sys.argv[4] == "M":
        data_complete = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_footer=int(REPS * (mag_bins+1) * run_rm / pujol_every + 3 * (run_lf - SKIP_FIRST_LF) * (mag_bins+1) * (REPS - (reps+1))),
                                      usecols=(0, 4, 5, 8, 9, 1, 2, 3, 10), skip_header=int(3 * (mag_bins+1) * (run_lf - SKIP_FIRST_LF) * reps))
        if sys.argv[5] == "0.02":
            data_puyol_compl = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (run_lf - SKIP_FIRST_LF) * (mag_bins + 1) * REPS + (mag_bins + 1) * SKIP_FIRST_RM + (mag_bins + 1) * run_rm / pujol_every * reps),
                                         usecols=(0, 16, 17, 11, 24, 1, 2, 3, 28), skip_footer=int((mag_bins+1) * (REPS - (reps+1)) * run_rm / pujol_every))
            # data_puyol_compl[:, 3] = data_puyol_compl[:, 3] * 2 / 11

        elif sys.argv[5] == "0.1":
            data_puyol_compl = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (run_lf - SKIP_FIRST_LF) * (mag_bins + 1) * REPS + (mag_bins + 1) * SKIP_FIRST_RM + (mag_bins + 1) * run_rm / pujol_every * reps),
                                             usecols=(0, 16, 17, 11, 24, 1, 2, 3, 26), skip_footer=int((mag_bins+1) * (REPS - (reps+1)) * run_rm / pujol_every))

    def linear_function(x, a):
        return a*x

    #print(reps)
    def sqrt_function(x, a):
        return a*x**(-0.5)

    magnitude_list = [min_mag + k*(max_mag-min_mag)/(mag_bins) for k in range(mag_bins+1)]
    for mag in range((mag_bins+1)):

        data_puyol = data_puyol_compl[mag::(mag_bins+1)][(data_puyol_compl[mag::(mag_bins+1)][:, 1] != -1) & (data_puyol_compl[mag::(mag_bins+1)][:, 2] != 0)]

        no_cancel = data_complete[np.where((names_list == "none") & (data_complete[:,4] == magnitude_list[mag]))[0]]
        shape_noise = data_complete[np.where((names_list == "shape") & (data_complete[:,4] == magnitude_list[mag]))[0]]
        both = data_complete[np.where((names_list == "both") & (data_complete[:,4] == magnitude_list[mag]))[0]]

        runtime_data = data_complete[:, 3]
        # Find out the longest runtime from all possible methods
        try:
            num_shears = 2 if sys.argv[5] == "0.02" else 11
            runtime_norm = np.max([np.max(runtime_data), np.max(data_puyol[:, 3])])
            runtime_norm_area = np.max([np.max((no_cancel[:, 7]+1) * shear_bins * 4 * no_cancel[:, 5]**2 / (100 * 3600**2)),
                                   np.max(data_puyol[:, 7] * num_shears * data_puyol[:, 5]**2 / (100 * 3600**2))])
        except ValueError:
            runtime_norm = np.max(runtime_data)
            runtime_norm_area = np.max((no_cancel[:, 7]+1) * shear_bins * 4 * no_cancel[:, 5]**2 / (100 * 3600**2))


        if reps == REPS-1:
            mm = 1 / 25.4  # millimeter in inches
            fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(88 * mm, 88 * mm))

        text_file = open(path+"output/rp_simulations/error_scaling.txt", "a")
        i = 0
        cancel_index = [1, 2, 4]
        names = ("no cancel", "shape (global)", "both (global)", "RM")
        colors = ("blue", "red", "black", "orange", "purple")
        for data in (no_cancel, shape_noise, both, data_puyol):
            # Ignore this. Only for testing
            factor_runtime = 0

            if len(data) == 0: #If no data is available at all (for example faintest bin)
                if reps == REPS -1:
                    text_file.write("%s\t %.7f\t %.7f\t %.6f\t %.6f\t %.6f\t %.6f\n" % (
                    names[i], 0, 0, 0, 0, 1, 0))
                i += 1
                continue

            if i != 3:
                runtime_data = data[:, 3] + factor_runtime * (data[:, 7]+1) * 20 * cancel_index[i]
                runtime_data_area = (data[:, 7] + 1) * data[:, 5]**2 / (100 * 3600**2) * cancel_index[i] * shear_bins
            else:
                if sys.argv[5] == "0.1":
                    runtime_data = data[:, 3] + factor_runtime * data_puyol[:, 7] * 11
                    runtime_data_area = data[:, 7] * data[:, 5]**2 / (100 * 3600**2) * 11

                elif sys.argv[5] == "0.02":
                    runtime_data = data[:, 3] + factor_runtime * data_puyol[:, 7] * 2
                    runtime_data_area = data[:, 7] * data[:, 5] ** 2 / (100 * 3600 ** 2) * 2

            filter = ~np.isnan(data[:,2]) & ~np.isinf(data[:,2])

            if len(data[:, 2][filter]) == 0: #If no data is available at all (for example faintest bin)
                if reps == REPS - 1:
                    text_file.write("%s\t %.7f\t %.7f\t %.6f\t %.6f\t %.6f\t %.6f\n" % (
                    names[i], 0, 0, 0, 0, 1, 0))
                i += 1
                continue

            
            if WITH_ERROR:
                popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data[filter] / runtime_norm), data[:, 2][filter], sigma=data[:, 8][filter], absolute_sigma=True)
                popt_area, pcov_area = curve_fit(linear_function, 1 / np.sqrt(runtime_data_area[filter] / runtime_norm_area),
                                       data[:, 2][filter], sigma=data[:, 8][filter], absolute_sigma=True)
            else:
                popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data[filter] / runtime_norm), data[:, 2][filter])
                popt_area, pcov_area = curve_fit(linear_function, 1 / np.sqrt(runtime_data_area[filter] / runtime_norm_area),
                                       data[:, 2][filter])
            error = np.sqrt(np.diag(pcov))
            error_area = np.sqrt(np.diag(popt_area))

            if i == 0:
                a_no_cancel = popt[0]
                fit_parameters[i][mag].append(a_no_cancel)
                # print(a_no_cancel)
                a_no_cancel_err = error[0]
                improvement = 0
                improvements[i][mag].append(improvement)
                improvement_err = 0
                factor = 0
                factor_err = 0

                a_no_cancel_area = popt_area[0]
                # print(a_no_cancel)
                a_no_cancel_err_area = error_area[0]
                improvement_area = 0
                improvements_area[i][mag].append(improvement_area)
                improvement_err_area = 0
                factor_area = 0
                factor_err_area = 0
            else:
                factor = a_no_cancel / popt[0]
                fit_parameters[i][mag].append(popt[0])
                factor_err = np.sqrt((a_no_cancel_err / (popt[0]))**2 + (a_no_cancel*(error[0]) / (popt[0])**2)**2)
                improvement = factor**2
                improvements[i][mag].append(improvement)
                improvement_err = 2 * factor * factor_err

                factor_area = a_no_cancel_area/ popt_area[0]
                factor_err_area = np.sqrt(
                    (a_no_cancel_err_area / (popt_area[0])) ** 2 + (a_no_cancel_area * (error_area[0]) / (popt_area[0]) ** 2) ** 2)
                improvement_area = factor_area ** 2
                improvements_area[i][mag].append(improvement_area)
                improvement_err_area = 2 * factor_area * factor_err_area

            if i != 3:
                skip = PLOT_EVERY_LF
            else:
                skip = 1

            if reps == REPS-1:


                if not WITH_ERROR:
                    axs[0].plot(runtime_data[::skip] / runtime_norm, data[:, 2][::skip], '+',
                                label=names[i], markersize=5, color=colors[i])
                    axs[1].plot(1 / np.sqrt(runtime_data[::skip] / runtime_norm), data[:, 2][::skip], '+',
                                label=names[i], markersize=5, color=colors[i])
                else:
                    axs[0].errorbar(runtime_data[::skip] / runtime_norm, data[:, 2][::skip], yerr=data[:, 8][::skip],
                                    fmt= '+', label=names[i], markersize=5, color=colors[i], capsize=2, elinewidth=0.5)
                    axs[1].errorbar(1 / np.sqrt(runtime_data[::skip] / runtime_norm), data[:, 2][::skip],
                                    yerr=data[:, 8][::skip],fmt= '+', label=names[i], markersize=5,
                                color=colors[i], capsize=2, elinewidth=0.5)


                axs[1].fill_between(1 / np.sqrt(runtime_data / runtime_norm),
                                    linear_function(1 / np.sqrt(runtime_data / runtime_norm),
                                                    np.min(fit_parameters[i][mag])),
                                    linear_function(1 / np.sqrt(runtime_data / runtime_norm),
                                                    np.max(fit_parameters[i][mag])), color=colors[i], alpha=.25)

                axs[1].plot(1 / np.sqrt(runtime_data / runtime_norm),
                            linear_function(1 / np.sqrt(runtime_data / runtime_norm), *popt),
                            color=colors[i], alpha=0.5)

                axs[0].plot(runtime_data / runtime_norm, sqrt_function(runtime_data / runtime_norm, *popt),
                            color=colors[i], alpha=0.5)

                axs[0].fill_between(runtime_data / runtime_norm, sqrt_function(runtime_data / runtime_norm,
                                                                               np.min(fit_parameters[i][mag])),
                                    sqrt_function(runtime_data / runtime_norm, np.max(fit_parameters[i][mag])),
                                    color=colors[i], alpha=.25)

                goal = sqrt_function(100, a_no_cancel_area)
                # print((popt[0] / goal) ** 2)
                area_equivalent = (popt_area[0] / goal) ** 2

                text_file.write("%s\t %.7f\t %.7f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.6f\t %.2f\n" %
                                (names[i], popt[0], error[0], factor, factor_err, improvement,
                                 np.std(improvements[i][mag]), improvement_area, np.std(improvements_area[i][mag]),
                                 area_equivalent))
            i += 1



        if reps == REPS-1:
            axs[0].set_yscale('log')

            axs[0].set_xscale('log')

            for axis in [axs[0].xaxis, axs[0].yaxis]:
                axis.set_minor_formatter(NullFormatter())

            axs[0].set_xlabel(r'$t_{run}$ normalized')


            axs[0].legend(prop={'size': 6})

            axs[1].legend(prop={'size': 6})
            axs[1].set_xlabel(r'$1/\sqrt{t_{run}}$')

            if sys.argv[4] == "M":
                axs[0].set_ylabel(r'$\sigma_{\mu}$')
                axs[1].set_ylabel(r'$\sigma_{\mu}$')
                #axs[0].set_title("Test")

                fig.savefig(path+f'output/plots/rp_{magnitude_list[mag]}.pdf', dpi=300, bbox_inches='tight')

            elif sys.argv[4] == "C":
                axs[0].set_ylabel(r'$\sigma_{c}$')
                axs[1].set_ylabel(r'$\sigma_{c}$')
                # axs[0].set_title("Test")

                fig.savefig(path + f'output/plots/rp_{magnitude_list[mag]}_c_bias.pdf', dpi=300,
                            bbox_inches='tight')

