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
from matplotlib.ticker import ScalarFormatter, FuncFormatter, NullFormatter, LogFormatter, FormatStrFormatter, LogLocator

WITH_ERROR = True
# Define paths

path = sys.argv[1]+"/"

config = configparser.ConfigParser()
config.read('config_grid.ini')


simulation = config['SIMULATION']

mag_bins = int(simulation["bins_mag"])
time_bins = int(simulation["time_bins"])

if sys.argv[2] == "M":
    data_complete = np.genfromtxt(path+"output/plots/error_with_real/tmp.txt", skip_footer=(mag_bins+1) * time_bins, usecols=(0,4,5, 11, 10, 12)) # 4,5, for m-bias | 6,7 for c-
    data_puyol_compl = np.genfromtxt(path+"output/plots/error_with_real/tmp.txt", skip_header=3 * (mag_bins+1) * time_bins, usecols=(0, 7, 8, 12, 11, 18)) # 7,8 for m-bias | 13,14 for c-bias
elif sys.argv[2] == "C":
    data_complete = np.genfromtxt(path + "output/plots/error_with_real/tmp.txt", skip_footer=(mag_bins+1) * time_bins,
                                  usecols=(0, 6, 7, 11, 10, 13))  # 4,5, for m-bias | 6,7 for c-
    data_puyol_compl = np.genfromtxt(path + "output/plots/error_with_real/tmp.txt", skip_header=3 * (mag_bins+1) * time_bins,
                                     usecols=(0, 13, 14, 12, 11, 19))  # 7,8 for m-bias | 13,14 for c-bias
test1 = np.genfromtxt(path+"output/plots/error_with_real/tmp.txt", skip_footer=(mag_bins+1) * time_bins, usecols=(0,1,2,3))
num_meas = test1[:,0] * test1[:,2] * test1[:,3]

test2 = np.genfromtxt(path+"output/plots/error_with_real/tmp.txt", skip_header=3 * (mag_bins+1) * time_bins, usecols=(0, 1))
num_meas_pujol = test2[:,0] * 11





def linear_function(x, a):
    return a*x


# print(data_puyol_compl[:,3])

def sqrt_function(x, a):
    return a*x**(-0.5)

magnitude_list = [float(simulation["min_mag"]) + k*(float(simulation["max_mag"])-float(simulation["min_mag"]))/(mag_bins) for k in range(mag_bins+1)]
for mag in range(mag_bins+1):
    # if mag == mag_bins:
    #     data_puyol = data_puyol_compl[mag+1::7][(data_puyol_compl[mag+1::7][:, 1] != -1) & (data_puyol_compl[mag+1::7][:, 2] != 0)]
    # else:
    #     data_puyol = data_puyol_compl[mag::6][(data_puyol_compl[mag::6][:, 1] != -1) & (data_puyol_compl[mag::6][:, 2] != 0)]
    # print(data_puyol)
    # data = data_complete[mag::6]
    #print(data)
    #two_in_one = data_complete[-7:]
    #print(data[30])
    # if mag == 4:
      #   mag +=1
    no_cancel = data_complete[0:((mag_bins+1) * time_bins)][mag::(mag_bins+1)]
    # print(no_cancel)
    # pixel = data_complete[60:120][mag::6]
    shape_noise = data_complete[((mag_bins+1) * time_bins):(2 * (mag_bins+1) * time_bins)][mag::(mag_bins+1)]
    both = data_complete[(2 * (mag_bins+1) * time_bins):(3 * (mag_bins+1) * time_bins)][mag::(mag_bins+1)]
    data_puyol = data_puyol_compl[mag::(mag_bins+1)]

    # #pixel_noise = data[2::4]
    # both = data[32:60:3]
    # pujol = data_puyol[20:400]
    # pujol_11 = data_puyol[410:480]

    # no_cancel = data_complete[np.where((names_list == "none") & (data_complete[:,4] == magnitude_list[mag]))[0]]
    #
    # shape_noise = data_complete[np.where((names_list == "shape") & (data_complete[:,4] == magnitude_list[mag]))[0]]
    # both = data_complete[np.where((names_list == "both") & (data_complete[:,4] == magnitude_list[mag]))[0]]
    #print(data_puyol)
    # both_no_pixel = data[42:53]
    # rest = data[53:57]
    # rest1 = data[57:61]
    # rest2 = data[61:64]
    # both_4_pixel = data[65:76]
    # both_no_sub = data[76:86]

    runtime_measure = "THEO"

    if runtime_measure == "THEO":
        runtime_data = data_complete[:, 3]
    else:
        runtime_data = data_complete[:, 3]


    runtime_norm = np.max([np.max(runtime_data), np.max(data_puyol[:, 3])])
    # runtime_norm = np.max(runtime_data)
    mm = 1/25.4  # millimeter in inches

    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(88*mm, 88*mm))

    #axs[0].grid(True)
    #axs[1].grid(True)

    text_file = open(path+"output/grid_simulations/error_scaling.txt", "a")
    i = 0
    names = ("no cancel", "shape", "both", "RM")
    colors = ("blue", "red", "black", "orange", "purple")
    for data in (no_cancel, shape_noise, both, data_puyol):
        # data = data[data[:, 0].argsort()]
        factor_runtime = 0.0
        if i != 3:
            runtime_data = data[:, 3] + num_meas[i*((mag_bins+1)*time_bins):(i+1)*((mag_bins+1)*time_bins)][mag::(mag_bins+1)] * factor_runtime * data[:,3] / data[:,3][0]
        else:
            runtime_data = data[:, 3] + num_meas_pujol[mag::(mag_bins+1)] * factor_runtime * data[:,3] / data[:,3][0]

        if WITH_ERROR:
            popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data / runtime_norm), data[:, 2], sigma=data[:, 5], absolute_sigma=True)
            error = np.sqrt(np.diag(pcov))
        else:
            popt, pcov = curve_fit(linear_function, 1 / np.sqrt(runtime_data / runtime_norm), data[:, 2])
            error = np.sqrt(np.diag(pcov))

        if i == 0:
            a_no_cancel = popt[0] / runtime_norm
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
        # if i == 3:
        #     print("test")
        #     data = data_puyol[::6]
        #     runtime_data = data[: ,3]
        if WITH_ERROR:
            axs[0].errorbar(runtime_data / runtime_norm, data[:, 2],yerr=data[:, 5], fmt= '+', label=names[i], markersize=5, color=colors[i], capsize=2, elinewidth=0.5)
            axs[1].errorbar(1 / np.sqrt(runtime_data / runtime_norm), data[:, 2],yerr=data[:, 5], fmt= '+', label=names[i], markersize=5, color=colors[i], capsize=2, elinewidth=0.5)
        else:
            axs[0].plot(runtime_data / runtime_norm, data[:, 2], '+', label=names[i], markersize=5, color=colors[i])
            axs[1].plot(1 / np.sqrt(runtime_data / runtime_norm), data[:, 2], '+', label=names[i], markersize=5,
                        color=colors[i])


        axs[1].plot(1 / np.sqrt(runtime_data / runtime_norm),
                    linear_function(1 / np.sqrt(runtime_data / runtime_norm), *popt), color=colors[i], alpha=0.5)
        axs[0].plot(runtime_data / runtime_norm, sqrt_function(runtime_data / runtime_norm, *popt), color=colors[i], alpha=0.5)

        text_file.write("%s\t %.7f\t %.7f\t %.6f\t %.6f\t %.6f\t %.6f\n" % (names[i], popt[0], error[0], factor, factor_err, improvement, improvement_err))
        i += 1



    # axs[0].plot(rest2[:,8]/runtime_norm,rest2[:,5],'+')
    # axs[1].plot(1/np.sqrt(rest2[:,8]/runtime_norm),rest2[:,5],'+')
    # axs[0].set_xlabel(r'$n_{real}$')
    axs[0].set_yscale('log')
    # axs[0].yaxis.set_major_formatter(ScalarFormatter())
    axs[0].set_xscale('log')
    # axs[0].set_xlim((1e-2, 1e0))
    # axs[0].set_ylim((1e-4, 1e-1))
    # axs[1].set_yscale('log')
    # axs[0].get_xaxis().set_major_formatter(ScalarFormatter())
    # f = ScalarFormatter()
    # f.set_scientific(False)
    # axs[0].xaxis.set_minor_formatter(LogFormatter())
    # axs[0].xaxis.set_major_locator(LogLocator(subs=(1, 2,)))

    # axs[0].xaxis.set_minor_formatter(ScalarFormatter())
    #axs[0].tick_params(axis='both', which='minor', labelsize=6)
    # for i, label in enumerate(axs[0].get_xticklabels(which="both")):
    #     print(i, label)
    #     if i > 0 and i < len(axs[0].get_xticklabels(which="both")) - 1:
    #         label.set_visible(False)

    axs[0].set_xlabel(r'$t_{run}$ normalized')
    if sys.argv[2] == "M":
        axs[0].set_ylabel(r'$\sigma_\mu$')
        axs[1].set_ylabel(r'$\sigma_\mu$')
    elif sys.argv[2] == "C":
        axs[0].set_ylabel(r'$\sigma_c$')
        axs[1].set_ylabel(r'$\sigma_c$')

    axs[0].legend(prop={'size': 6})
    #axs[0].text(0.4, 0.8*np.max(no_cancel[:, 12]), f"runtime norm = {int(runtime_norm)} s")

    # axs[1].set_yscale('log')
    # axs[1].set_xscale('log')
    axs[1].legend(prop={'size': 6})
    # axs[1].set_xlabel(r'$1/\sqrt{n_{real}}$')
    axs[1].set_xlabel(r'$1/\sqrt{t_{run}}$')

    #axs[0].set_title("Test")

    if sys.argv[2] == "M":
        fig.savefig(path+f'output/plots/error_with_real_run10_{magnitude_list[mag]}.pdf', dpi=300, bbox_inches='tight')
    elif sys.argv[2] == "C":
        fig.savefig(path + f'output/plots/error_with_real_run10_{magnitude_list[mag]}_c_bias.pdf', dpi=300,
                    bbox_inches='tight')
