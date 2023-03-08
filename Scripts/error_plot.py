# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:10:37 2021

This function reads in the temporary file with the fits and fits the uncertainty behaviour with time.

It outputs the runtime improvements.

@author: Henning
"""
WITH_ERROR = True

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

# How many runs to skip for fit methods
SKIP_FIRST_LF = int(simulation["skip_first_lf"])
PLOT_EVERY_LF = int(simulation["plot_every_lf"])
# How many runs to skip for RM
SKIP_FIRST_RM = int(simulation["skip_first_rm"])

if (SKIP_FIRST_RM <= 1) or (SKIP_FIRST_LF <=1):
    raise ValueError("At least first 5 points should be skipped since bootstrapping likely yields NaNs before")



mag_bins = int(simulation["bins_mag"])

pujol_every = int(simulation["puj_analyse_every"])
min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

# Read in the columns with relevant information from the temporary file
names_list = np.genfromtxt(path+"output/rp_simulations/tmp_rm.txt", skip_footer=int((mag_bins+1) * run_rm / pujol_every), dtype="unicode", skip_header=int(3 * (mag_bins+1) * SKIP_FIRST_LF))[:,0]

if sys.argv[4] == "C":
    data_complete = np.genfromtxt(path+"output/rp_simulations/tmp_rm.txt", skip_footer=int((mag_bins+1) * run_rm / pujol_every), usecols=(0,6,7,8,9,1,2,3,11)
                                  , skip_header=int(3 * (mag_bins+1) * SKIP_FIRST_LF))

    if sys.argv[5] == "0.02":
        data_puyol_compl = np.genfromtxt(path+"output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (mag_bins+1) * run_lf + (mag_bins+1) * SKIP_FIRST_RM), usecols=(0, 18, 19, 11, 24,1,2,3, 29))
        # data_puyol_compl[:, 3] = data_puyol_compl[:, 3] * 2 / 11

    elif sys.argv[5] == "0.1":
        data_puyol_compl = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (mag_bins+1) * run_lf + (mag_bins+1) * SKIP_FIRST_RM),
                                         usecols=(0, 18, 19, 11, 24, 1, 2, 3, 27))
elif sys.argv[4] == "M":
    data_complete = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_footer=int((mag_bins+1) * run_rm / pujol_every),
                                  usecols=(0, 4, 5, 8, 9, 1, 2, 3, 10), skip_header=int(3 * (mag_bins+1) * SKIP_FIRST_LF))
    if sys.argv[5] == "0.02":
        data_puyol_compl = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (mag_bins+1) * run_lf + (mag_bins+1) * SKIP_FIRST_RM),
                                     usecols=(0, 16, 17, 11, 24, 1, 2, 3, 28))
        # data_puyol_compl[:, 3] = data_puyol_compl[:, 3] * 2 / 11

    elif sys.argv[5] == "0.1":
        data_puyol_compl = np.genfromtxt(path + "output/rp_simulations/tmp_rm.txt", skip_header=int(3 * (mag_bins+1) * run_lf + (mag_bins+1) * SKIP_FIRST_RM),
                                         usecols=(0, 16, 17, 11, 24, 1, 2, 3, 26))

def linear_function(x, a):
    return a*x


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
        runtime_norm = np.max([np.max(runtime_data), np.max(data_puyol[:, 3])])
    except ValueError:
        runtime_norm = np.max(runtime_data)

    mm = 1/25.4  # millimeter in inches

    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(88*mm, 88*mm))

    text_file = open(path+"output/rp_simulations/error_scaling.txt", "a")
    i = 0
    test1 = [1, 2, 4]
    names = ("no cancel", "shape (global)", "both (global)", "RM")
    colors = ("blue", "red", "black", "orange", "purple")
    for data in (no_cancel, shape_noise, both, data_puyol):
        # Ignore this. Only for testing
        factor_runtime = 0

        if len(data) == 0: #If no data is available at all (for example faintest bin)
            text_file.write("%s\t %.7f\t %.7f\t %.6f\t %.6f\t %.6f\t %.6f\n" % (
            names[i], 0, 0, 0, 0, 1, 0))
            i += 1
            continue

        if i != 3:
            runtime_data = data[:, 3] + factor_runtime * (data[:, 7]+1) * 20 * test1[i]
        else:
            runtime_data = data[:, 3] + factor_runtime * data_puyol[:, 7] * 11
        filter = ~np.isnan(data[:,2]) & ~np.isinf(data[:,2])

        if len(data[:, 2][filter]) == 0: #If no data is available at all (for example faintest bin)
            text_file.write("%s\t %.7f\t %.7f\t %.6f\t %.6f\t %.6f\t %.6f\n" % (
            names[i], 0, 0, 0, 0, 1, 0))
            i += 1
            continue

        # print(runtime_data)
        if WITH_ERROR:
            popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data[filter] / runtime_norm), data[:, 2][filter], sigma=data[:, 8][filter], absolute_sigma=True)
        else:
            popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data[filter] / runtime_norm), data[:, 2][filter])
        error = np.sqrt(np.diag(pcov))

        if i == 0:
            a_no_cancel = popt[0] / runtime_norm
            # print(a_no_cancel)
            a_no_cancel_err = error[0] / runtime_norm
            improvement = 0
            improvement_err = 0
            factor = 0
            factor_err = 0
        else:
            factor = a_no_cancel / (popt[0] / runtime_norm)
            factor_err = np.sqrt((a_no_cancel_err / (popt[0] / runtime_norm))**2 + (a_no_cancel*(error[0] / runtime_norm) / (popt[0] / runtime_norm)**2)**2)
            improvement = factor**2
            improvement_err = 2 * factor * factor_err

        if i != 3:
            skip = PLOT_EVERY_LF
        else:
            skip = 1

        if not WITH_ERROR:
            axs[0].plot(runtime_data[::skip] / runtime_norm, data[:, 2][::skip], '+', label=names[i], markersize=5, color=colors[i])
            axs[1].plot(1 / np.sqrt(runtime_data[::skip] / runtime_norm), data[:, 2][::skip], '+', label=names[i], markersize=5, color=colors[i])
        else:
            axs[0].errorbar(runtime_data[::skip] / runtime_norm, data[:, 2][::skip], yerr=data[:, 8][::skip], fmt= '+', label=names[i], markersize=5, color=colors[i], capsize=2, elinewidth=0.5)
            axs[1].errorbar(1 / np.sqrt(runtime_data[::skip] / runtime_norm), data[:, 2][::skip], yerr=data[:, 8][::skip],fmt= '+', label=names[i], markersize=5,
                        color=colors[i], capsize=2, elinewidth=0.5)



        axs[1].plot(1 / np.sqrt(runtime_data / runtime_norm),
                    linear_function(1 / np.sqrt(runtime_data / runtime_norm), *popt), color=colors[i], alpha=0.5)
        axs[0].plot(runtime_data / runtime_norm, sqrt_function(runtime_data / runtime_norm, *popt), color=colors[i], alpha=0.5)

        text_file.write("%s\t %.7f\t %.7f\t %.6f\t %.6f\t %.6f\t %.6f\n" % (names[i], popt[0], error[0], factor, factor_err, improvement, improvement_err))
        i += 1




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

