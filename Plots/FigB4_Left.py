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
file_name = "data/FigB2_B4_Left.txt"

mag_bins = 6
min_mag = 20.5
max_mag = 26.5

data = np.genfromtxt(file_name, skip_footer=(mag_bins + 1))
data_pu = np.genfromtxt(file_name, skip_header=(mag_bins + 1) * 6)



mm = 1 / 25.4  # millimeter in inches

# Creates magnitude array from binning and min/max
magnitudes = np.array([min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)])

fig, axs = plt.subplots(figsize=(93 * mm, 88 * mm), sharey="row")


axs.set_title("Random positions")

# None
axs.errorbar(magnitudes + 0.3, (data[:, 6][0:(mag_bins + 1) * 3:3] + data[:, 6][(mag_bins + 1) * 3::3]) / 2,
             yerr=np.sqrt(
                 data[:, 7][0:(mag_bins + 1) * 3:3] ** 2 + data[:, 7][(mag_bins + 1) * 3::3] ** 2) / 2,
             fmt='s', capsize=2, label="none", markersize=3)

# Shape local
axs.errorbar(magnitudes + 0.4, data[:, 6][1:(mag_bins + 1) * 3:3], yerr=data[:, 7][1:(mag_bins + 1) * 3:3],
             fmt='^', capsize=2, label="shape local", markersize=3)

# Both local
axs.errorbar(magnitudes + 0.5, data[:, 6][2:(mag_bins + 1) * 3:3], yerr=data[:, 7][2:(mag_bins + 1) * 3:3],
             fmt='v', capsize=2, label="both local", markersize=3)

# Shape local
axs.errorbar(magnitudes + 0.6, data[:, 6][(mag_bins + 1) * 3 + 1::3],
             yerr=data[:, 7][(mag_bins + 1) * 3 + 1::3], fmt='x', capsize=2, label="shape global", markersize=3)

# Shape global
axs.errorbar(magnitudes + 0.7, data[:, 6][(mag_bins + 1) * 3 + 2::3],
             yerr=data[:, 7][(mag_bins + 1) * 3 + 2::3], fmt='p', capsize=2, label="both global", markersize=3)

# Response method
axs.errorbar(magnitudes + 0.8, data_pu[:, 18], yerr=data_pu[:, 19], fmt="+", capsize=2, label="RM", markersize=3)
axs.set_xlabel("$m_{\mathrm{AUTO}}$")
axs.set_ylabel("$c$-bias")
axs.set_xlim(20.5, magnitudes[-1])
axs.set_ylim(-0.0005, 0.002)
axs.set_xticks([21,22,23,24,25,26])

# axs.grid(True)
axs.set_xlim(20.5, magnitudes[-1])
axs.set_xticks([21, 22, 23, 24, 25, 26])

axs.legend(ncol=2, prop={'size':6})

# axs[0].set_ylim(-0.25, 0.1)
# axs[1].set_ylim(-0.25, 0.1)

fig.tight_layout()
fig.savefig("FigB4_Left.pdf", dpi=200)
plt.close(fig)
