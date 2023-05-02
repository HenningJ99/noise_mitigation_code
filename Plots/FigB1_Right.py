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
file_name = "data/FigB1_B3_Right.txt"

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
axs.errorbar(magnitudes + 0.25, data_2[:, 4][0:(mag_bins + 1)], yerr=data_2[:, 5][0:(mag_bins + 1)],
             fmt='s', capsize=2,
             label="none", markersize=3)

# Shape
axs.errorbar(magnitudes + 0.35, data_2[:, 4][(mag_bins + 1):(mag_bins + 1) * 2],
             yerr=data_2[:, 5][(mag_bins + 1):(mag_bins + 1) * 2], fmt='^', capsize=2,
             label="shape", markersize=3)

# Both
axs.errorbar(magnitudes + 0.45, data_2[:, 4][(mag_bins + 1) * 2:(mag_bins + 1) * 3],
             yerr=data_2[:, 5][(mag_bins + 1) * 2:(mag_bins + 1) * 3], fmt='v', capsize=2,
             label="both", markersize=3)

# Response method
axs.errorbar(magnitudes + 0.55, data_p[:, 20], yerr=data_p[:, 21], fmt="x", capsize=2, label="RM", markersize=3)
axs.set_ylim(-0.35, 0.1)
# Do the zoom ins
for bin in range(mag_bins):
    axins = inset_axes(axs, width=0.4, height=0.7,
                       axes_kwargs={"xlim": (magnitudes[bin] + 0.15, magnitudes[bin] + 0.65),
                                    "xticklabels": (), "yticklabels": (), "xticks": (), "yticks": ()})

    # None
    axins.errorbar(magnitudes[bin] + 0.25, data_2[:, 4][0:(mag_bins + 1)][bin],
                   yerr=data_2[:, 5][0:(mag_bins + 1)][bin], fmt='s', capsize=2,
                   label="none", markersize=3)

    # Shape
    axins.errorbar(magnitudes[bin] + 0.35, data_2[:, 4][(mag_bins + 1):(mag_bins + 1) * 2][bin],
                   yerr=data_2[:, 5][(mag_bins + 1):(mag_bins + 1) * 2][bin], fmt='^', capsize=2,
                   label="shape", markersize=3)

    # Both
    axins.errorbar(magnitudes[bin] + 0.45, data_2[:, 4][(mag_bins + 1) * 2:(mag_bins + 1) * 3][bin],
                   yerr=data_2[:, 5][(mag_bins + 1) * 2:(mag_bins + 1) * 3][bin], fmt='v', capsize=2,
                   label="both", markersize=3)

    # Response method
    axins.errorbar(magnitudes[bin] + 0.55, data_p[:, 20][bin], yerr=data_p[:, 21][bin], fmt="x", capsize=2,
                   label="RM", markersize=3)
    ylim = axs.get_ylim()
    y_delta = ylim[1] - ylim[0]
    ip = InsetPosition(axs,
                       [0.02 + bin / mag_bins, 1 - np.abs(axins.get_ylim()[0] - ylim[1]) / y_delta - 0.3, 1 / (mag_bins+1),
                        0.25])
    axins.set_axes_locator(ip)
    axins.spines[["left", "right", "bottom", "top"]].set_alpha(0.5)
    rect, lines = axs.indicate_inset_zoom(axins)
    lines[0].set(visible=True)
    lines[1].set(visible=True)
    lines[2].set(visible=True)
    lines[3].set(visible=True)
axs.set_xlabel("$m_{\mathrm{GEMS}}$")
#axs.set_ylabel("$\mu$-bias")
axs.set_xlim(20.5, magnitudes[-1])

#axs.set_yticks([])
axs.set_yticklabels([])
# axs.grid(True)
#axs.set_ylabel("$\mu$-bias")
# axs.grid(True)
axs.set_xlim(20.5, magnitudes[-1])


axs.legend()

fig.tight_layout()
fig.savefig("FigB1_Right.pdf", dpi=200)
plt.close(fig)
