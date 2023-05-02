#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:35:24 2022

This script produces plots of the bias as binned in magnitude
@author: hjansen
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from scipy.optimize import curve_fit
import configparser
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition


def linear_function(x, a, b):
    return a * x + b


# Reads in the file name and the type of the simulation
file_name = "data/FigB2_B4_Right.txt"

mag_bins = 5
min_mag = 20.5
max_mag = 25.5

data_2 = np.genfromtxt(file_name, skip_footer=mag_bins + 1)
data_p = np.genfromtxt(file_name, skip_header=(mag_bins + 1) * 3)



mm = 1 / 25.4  # millimeter in inches

# Creates magnitude array from binning and min/max
magnitudes = np.array([min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)])

fig, axs = plt.subplots(figsize=(83 * mm, 88 * mm), sharey="row")


axs.set_title("Grid")

# None
axs.errorbar(magnitudes + 0.3, data_2[:, 6][0:(mag_bins + 1)], yerr=data_2[:, 7][0:(mag_bins + 1)], fmt='s',
             capsize=2,
             label="none", markersize=3)

# Shape
axs.errorbar(magnitudes + 0.4, data_2[:, 6][(mag_bins + 1):(mag_bins + 1) * 2],
             yerr=data_2[:, 7][(mag_bins + 1):(mag_bins + 1) * 2], fmt='^', capsize=2,
             label="shape", markersize=3)

# Both
axs.errorbar(magnitudes + 0.5, data_2[:, 6][(mag_bins + 1) * 2:(mag_bins + 1) * 3],
             yerr=data_2[:, 7][(mag_bins + 1) * 2:(mag_bins + 1) * 3], fmt='v', capsize=2,
             label="both", markersize=3)

# Response method
axs.errorbar(magnitudes + 0.6, data_p[:, 22], yerr=data_p[:, 23], fmt="x", capsize=2, label="RM", markersize=3)
axs.set_xlabel("$m_{\mathrm{GEMS}}$")
#axs.set_ylabel("$c$-bias")
axs.set_ylim(-0.0005, 0.002)
# axs.grid(True)
axs.set_yticklabels([])

axs.set_xlim(20.5, magnitudes[-1])


axs.legend()

fig.tight_layout()
fig.savefig("FigB4_Right.pdf", dpi=200)
plt.close(fig)
