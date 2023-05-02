# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:10:37 2021

@author: Henning
"""

import numpy as np
import matplotlib . pyplot as plt
from scipy.optimize import curve_fit
import configparser
import sys
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, NullFormatter)
import matplotlib.ticker as mticker

run_lf = 140
run_rm = 2000

# How many runs to skip for fit methods
SKIP_FIRST_LF = 59
PLOT_EVERY_LF = 10
# How many runs to skip for RM
SKIP_FIRST_RM = 5

mag_bins = 6

pujol_every = 100
min_mag = 20.5
max_mag = 26.5

names_list = np.genfromtxt("data/Fig11_Right.txt", skip_footer=int((mag_bins+1) * run_rm / pujol_every + 3 * (mag_bins+1) * run_lf), dtype="unicode", skip_header=int(3 * (mag_bins+1) * SKIP_FIRST_LF))[:,0]
names_list_2 = np.genfromtxt("data/Fig11_Right.txt", skip_footer=int((mag_bins+1) * run_rm / pujol_every), dtype="unicode", skip_header=int(3 * (mag_bins+1) * SKIP_FIRST_LF + 3 * (mag_bins+1) * run_lf))[:,0]


data_complete = np.genfromtxt("data/Fig11_Right.txt", skip_footer=int((mag_bins+1) * run_rm / pujol_every + 3 * (mag_bins+1) * run_lf),
                              usecols=(0, 4, 5, 8, 9, 1, 2, 3, 10), skip_header=int(3 * (mag_bins+1) * SKIP_FIRST_LF))

data_complete_2 = np.genfromtxt("data/Fig11_Right.txt",
                              skip_footer=int((mag_bins+1) * run_rm / pujol_every),
                              usecols=(0, 4, 5, 8, 9, 1, 2, 3, 10),
                              skip_header=int(3 * (mag_bins+1) * SKIP_FIRST_LF + 3 * (mag_bins+1) * run_lf))

data_puyol_compl = np.genfromtxt("data/Fig11_Right.txt", skip_header=int(6 * (mag_bins+1) * run_lf + (mag_bins+1) * SKIP_FIRST_RM),
                                         usecols=(0, 16, 17, 11, 24, 1, 2, 3, 26))

def linear_function(x, a):
    return a*x


def sqrt_function(x, a):
    return a*x**(-0.5)

magnitude_list = [min_mag + k*(max_mag-min_mag)/(mag_bins) for k in range(mag_bins+1)]

data_puyol = data_puyol_compl[6::(mag_bins+1)][(data_puyol_compl[6::(mag_bins+1)][:, 1] != -1) & (data_puyol_compl[6::(mag_bins+1)][:, 2] != 0)]

no_cancel = data_complete[np.where((names_list == "none") & (data_complete[:,4] == magnitude_list[6]))[0]]
print(len(no_cancel))
# print(data_puyol)
shape_noise = data_complete[np.where((names_list == "shape") & (data_complete[:,4] == magnitude_list[6]))[0]]
both = data_complete[np.where((names_list == "both") & (data_complete[:,4] == magnitude_list[6]))[0]]
shape_noise_2 = data_complete_2[np.where((names_list_2 == "shape") & (data_complete_2[:, 4] == magnitude_list[6]))[0]]
both_2 = data_complete_2[np.where((names_list_2 == "both") & (data_complete_2[:, 4] == magnitude_list[6]))[0]]

runtime_measure = "THEO"

if runtime_measure == "THEO":
    runtime_data = data_complete[:, 3]
else:
    runtime_data = data_complete[:, 3]

try:
    runtime_norm = np.max([np.max(runtime_data), np.max(data_puyol[:, 3])])
except ValueError:
    runtime_norm = np.max(runtime_data)

mm = 1/25.4  # millimeter in inches

fig, axs = plt.subplots(constrained_layout=True, figsize=(83*mm, 88*mm))


i = 0
test1 = [1, 2, 4, 2, 4]
names = ("no cancel", "shape (global)", "both (global)", "shape (local)", "both (local)", "RM")
colors = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
fmts = ("s", "o", "v", "^", "x", "p")
for data in (no_cancel, shape_noise, both, shape_noise_2, both_2, data_puyol):
    # data = data[data[:, 0].argsort()]
    factor_runtime = 0

    if len(data) == 0: #If no data is available at all (for example faintest bin)
        i += 1
        continue

    if i != 5:
        runtime_data = data[:, 3] + factor_runtime * (data[:, 7]+1) * 20 * test1[i]
    else:
        runtime_data = data[:, 3] + factor_runtime * data_puyol[:, 7] * 11
    filter = ~np.isnan(data[:,2]) & ~np.isinf(data[:,2])

    if len(data[:, 2][filter]) == 0: #If no data is available at all (for example faintest bin)
        i += 1
        continue

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

    if i != 5:
        skip = PLOT_EVERY_LF
    else:
        skip = 1

    axs.plot(runtime_data[::skip] / runtime_norm, data[:, 2][::skip], fmts[i], label=names[i], markersize=4, color=colors[i])

    axs.plot(runtime_data / runtime_norm, sqrt_function(runtime_data / runtime_norm, *popt), color=colors[i], alpha=0.5)

    i += 1

axs.set_yscale('log')

axs.set_xscale('log')

for axis in [axs.xaxis, axs.yaxis]:
    axis.set_minor_formatter(NullFormatter())

axs.set_xlabel(r'$t_{run}$ normalized')

axs.set_title(f"[$-${0.1}, {0.1}]")
axs.legend()
axs.set_ylim(0.0003, 0.03)
axs.set_yticklabels([])

fig.savefig("Fig11_Right.pdf", dpi=300, bbox_inches='tight')



