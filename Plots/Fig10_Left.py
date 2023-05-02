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


mag_bins = 4
time_bins = 10


data_complete = np.genfromtxt("data/Fig10_Left.txt", skip_footer=(mag_bins+1) * time_bins, usecols=(0,4,5, 11, 10, 12)) # 4,5, for m-bias | 6,7 for c-
data_puyol_compl = np.genfromtxt("data/Fig10_Left.txt", skip_header=3 * (mag_bins+1) * time_bins, usecols=(0, 7, 8, 12, 11, 18)) # 7,8 for m-bias | 13,14 for c-bias

test1 = np.genfromtxt("data/Fig10_Left.txt", skip_footer=(mag_bins+1) * time_bins, usecols=(0,1,2,3))
num_meas = test1[:,0] * test1[:,2] * test1[:,3]

test2 = np.genfromtxt("data/Fig10_Left.txt", skip_header=3 * (mag_bins+1) * time_bins, usecols=(0, 1))
num_meas_pujol = test2[:,0] * 11


def linear_function(x, a):
    return a*x


def sqrt_function(x, a):
    return a*x**(-0.5)
magnitude_list = [20.5 + k*1 for k in range(mag_bins+1)]

no_cancel = data_complete[0:((mag_bins+1) * time_bins)][4::(mag_bins+1)]
shape_noise = data_complete[((mag_bins+1) * time_bins):(2 * (mag_bins+1) * time_bins)][4::(mag_bins+1)]
both = data_complete[(2 * (mag_bins+1) * time_bins):(3 * (mag_bins+1) * time_bins)][4::(mag_bins+1)]
data_puyol = data_puyol_compl[4::(mag_bins+1)]



runtime_measure = "THEO"

if runtime_measure == "THEO":
    runtime_data = data_complete[:, 3]
else:
    runtime_data = data_complete[:, 3]


runtime_norm = np.max([np.max(runtime_data), np.max(data_puyol[:, 3])])

mm = 1/25.4  # millimeter in inches

fig, axs = plt.subplots(constrained_layout=True, figsize=(93*mm, 88*mm))

i = 0
names = ("no cancel", "shape", "both", "RM")
colors = ("blue", "red", "black", "orange", "purple")
fmts = ("s", "^", "v", "o")
for data in (no_cancel, shape_noise, both, data_puyol):
    # data = data[data[:, 0].argsort()]
    factor_runtime = 0.0
    if i != 3:
        runtime_data = data[:, 3] + num_meas[i*((mag_bins+1)*time_bins):(i+1)*((mag_bins+1)*time_bins)][4::(mag_bins+1)] * factor_runtime * data[:,3] / data[:,3][0]
    else:
        runtime_data = data[:, 3] + num_meas_pujol[4::(mag_bins+1)] * factor_runtime * data[:,3] / data[:,3][0]


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


    axs.plot(runtime_data / runtime_norm, data[:, 2], fmts[i], label=names[i], markersize=4, color=colors[i])

    axs.plot(runtime_data / runtime_norm, sqrt_function(runtime_data / runtime_norm, *popt), color=colors[i], alpha=0.5)

    i += 1



axs.set_title(f"[$-${0.02}, {0.02}]")
axs.set_yscale('log')
#axs.set_yticklabels([])


axs.set_xscale('log')
axs.set_ylim(0.0003, 0.05)

axs.set_xlabel(r'$t_{run}$ normalized')
axs.set_ylabel(r'$\sigma_\mu$')

fig.savefig("Fig10_Left.pdf", dpi=300, bbox_inches='tight')

