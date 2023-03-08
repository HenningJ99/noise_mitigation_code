#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:35:24 2022

@author: hjansen
"""

import numpy as np
import matplotlib . pyplot as plt
import math
import sys
from scipy.optimize import curve_fit
import configparser
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition



def linear_function(x,a,b):
    return (a*x+b)

file_name = sys.argv[1]
type = sys.argv[2]

if type == "RP":
    config = configparser.ConfigParser()
    config.read('config_rp.ini')

    simulation = config['SIMULATION']

    mag_bins = int(simulation["bins_mag"])
    min_mag = float(simulation["min_mag"])
    max_mag = float(simulation["max_mag"])

    data = np.genfromtxt(file_name, skip_footer=(mag_bins+1))
    data_pu = np.genfromtxt(file_name, skip_header=(mag_bins+1)*6)

elif type == "GR":
    config = configparser.ConfigParser()
    config.read('config_grid.ini')

    simulation = config['SIMULATION']

    mag_bins = int(simulation["bins_mag"])
    min_mag = float(simulation["min_mag"])
    max_mag = float(simulation["max_mag"])

    data_2 = np.genfromtxt(file_name, skip_footer=mag_bins + 1)
    data_p = np.genfromtxt(file_name, skip_header=(mag_bins+1) * 3)





# print(data_pu[:,23])



# print(data)
mm = 1/25.4  # millimeter in inches


magnitudes = np.array([min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)])


for biases in ["M", "C"]:
    fig, axs = plt.subplots(figsize=(88*mm, 88*mm), sharey="row")

    if biases == "M":
        if type == "RP":
            axs.set_title("Random positions")
            axs.errorbar(magnitudes+0.25, (data[:,4][0:(mag_bins+1)* 3:3] + data[:, 4][(mag_bins+1)*3::3]) / 2, yerr=np.sqrt(data[:,5][0:(mag_bins+1)*3:3]**2 + data[:,5][(mag_bins+1)*3::3]**2) / 2, fmt='+', capsize=2, label="none")
            axs.errorbar(magnitudes+0.35, data[:,4][1:(mag_bins+1)*3:3], yerr=data[:,5][1:(mag_bins+1)*3:3], fmt='+', capsize=2, label="shape local")
            axs.errorbar(magnitudes+0.45, data[:,4][2:(mag_bins+1)*3:3], yerr=data[:,5][2:(mag_bins+1)*3:3], fmt='+', capsize=2, label="both local")
            axs.errorbar(magnitudes + 0.55, data[:, 4][(mag_bins+1)*3+1::3], yerr=data[:, 5][(mag_bins+1)*3+1::3], fmt='+', capsize=2, label="shape global")
            axs.errorbar(magnitudes + 0.65, data[:, 4][(mag_bins+1)*3+2::3], yerr=data[:, 5][(mag_bins+1)*3+2::3], fmt='+', capsize=2, label="both global")
            axs.errorbar(magnitudes+0.75, data_pu[:,16], yerr=data_pu[:,17], fmt="+", capsize=2, label="RM")

            axs.set_ylim(-0.5, 0.2)
            for bin in range(mag_bins):

                axins = inset_axes(axs, width=0.4, height=0.7, axes_kwargs={"xlim":(magnitudes[bin]+0.15, magnitudes[bin]+0.85), "xticklabels":(), "yticklabels":(), "xticks":(), "yticks":()})

                axins.errorbar(magnitudes[bin] + 0.25,
                               ((data[:, 4][0:(mag_bins + 1) * 3:3] + data[:, 4][(mag_bins + 1) * 3::3]))[bin] / 2, yerr=np.sqrt(
                        data[:, 5][0:(mag_bins + 1) * 3:3] ** 2 + data[:, 5][(mag_bins + 1) * 3::3] ** 2)[bin] / 2, fmt='+',
                                capsize=2, label="none", alpha=0.8)
                axins.errorbar(magnitudes[bin] + 0.35, data[:, 4][1:(mag_bins + 1) * 3:3][bin],
                                yerr=data[:, 5][1:(mag_bins + 1) * 3:3][bin], fmt='+', capsize=2, label="shape local", alpha=0.8)
                axins.errorbar(magnitudes[bin] + 0.45, data[:, 4][2:(mag_bins + 1) * 3:3][bin],
                                yerr=data[:, 5][2:(mag_bins + 1) * 3:3][bin], fmt='+', capsize=2, label="both local", alpha=0.8)
                axins.errorbar(magnitudes[bin] + 0.55, data[:, 4][(mag_bins + 1) * 3 + 1::3][bin],
                                yerr=data[:, 5][(mag_bins + 1) * 3 + 1::3][bin], fmt='+', capsize=2, label="shape global", alpha=0.8)
                axins.errorbar(magnitudes[bin] + 0.65, data[:, 4][(mag_bins + 1) * 3 + 2::3][bin],
                                yerr=data[:, 5][(mag_bins + 1) * 3 + 2::3][bin], fmt='+', capsize=2, label="both global", alpha=0.8)
                axins.errorbar(magnitudes[bin] + 0.75, data_pu[:, 16][bin], yerr=data_pu[:, 17][bin], fmt="+", capsize=2, label="RM", alpha=0.8)
                ylim = axs.get_ylim()
                y_delta = ylim[1]-ylim[0]
                ip = InsetPosition(axs, [0.02 + 0.165 * bin, 1 - np.abs(axins.get_ylim()[0] - ylim[1]) / y_delta - 0.3, 0.13, 0.25])
                axins.set_axes_locator(ip)
                axins.spines[["left", "right", "bottom", "top"]].set_alpha(0.5)

                rect, lines = axs.indicate_inset_zoom(axins)
                lines[0].set(visible=True)
                lines[1].set(visible=True)
                lines[2].set(visible=True)
                lines[3].set(visible=True)
            # axs.errorbar(magnitudes + 0.3, data[:, 4][0:(mag_bins + 1) * 3:3],
            #                 yerr=3*data[:, 5][0:(mag_bins + 1) * 3:3], fmt='+', capsize=2, label="none", alpha=0.7, color="C0")
            # axs.errorbar(magnitudes + 0.4, data[:, 4][1:(mag_bins + 1) * 3:3],
            #                 yerr=3*data[:, 5][1:(mag_bins + 1) * 3:3], fmt='+', capsize=2, label="shape (stat)", alpha=0.7, color="C1")
            # axs.errorbar(magnitudes + 0.5, data[:, 4][2:(mag_bins + 1) * 3:3],
            #                 yerr=3*data[:, 5][2:(mag_bins + 1) * 3:3], fmt='+', capsize=2, label="both (stat)", alpha=0.7, color="C2")
            # axs.errorbar(magnitudes + 0.6, data[:, 4][(mag_bins + 1) * 3 + 1::3],
            #                 yerr=3*data[:, 5][(mag_bins + 1) * 3 + 1::3], fmt='+', capsize=2, label="shape (dyn)", alpha=0.7, color="C3")
            # axs.errorbar(magnitudes + 0.7, data[:, 4][(mag_bins + 1) * 3 + 2::3],
            #                 yerr=3*data[:, 5][(mag_bins + 1) * 3 + 2::3], fmt='+', capsize=2, label="both (dyn)", alpha=0.7, color="C4")
            # axs.errorbar(magnitudes + 0.8, data_pu[:, 4], yerr=3*data_pu[:, 5], fmt="+", capsize=2, label="RM", alpha=0.7, color="C5")
            axs.set_xlabel("$m_{\mathrm{AUTO}}$")
            axs.set_ylabel("$\mu$-bias")
            # axs.grid(True)
            axs.set_xlim(20.5, magnitudes[-2])
        elif type == "GR":
            axs.set_title("Grid")
            axs.errorbar(magnitudes + 0.25, data_2[:, 4][0:(mag_bins+1)], yerr=data_2[:, 5][0:(mag_bins+1)], fmt='+', capsize=2,
                            label="none")
            axs.errorbar(magnitudes + 0.35, data_2[:, 4][(mag_bins+1):(mag_bins+1)*2], yerr=data_2[:, 5][(mag_bins+1):(mag_bins+1)*2], fmt='+', capsize=2,
                            label="shape")
            axs.errorbar(magnitudes + 0.45, data_2[:, 4][(mag_bins+1)*2:(mag_bins+1)*3], yerr=data_2[:, 5][(mag_bins+1)*2:(mag_bins+1)*3], fmt='+', capsize=2,
                            label="both")
            axs.errorbar(magnitudes + 0.55, data_p[:, 20], yerr=data_p[:, 21], fmt="+", capsize=2, label="RM")
            for bin in range(mag_bins):

                axins = inset_axes(axs, width=0.4, height=0.7, axes_kwargs={"xlim":(magnitudes[bin]+0.15, magnitudes[bin]+0.65), "xticklabels":(), "yticklabels":(), "xticks":(), "yticks":()})

                axins.errorbar(magnitudes[bin] + 0.25, data_2[:, 4][0:(mag_bins + 1)][bin],
                                yerr=data_2[:, 5][0:(mag_bins + 1)][bin], fmt='+', capsize=2,
                                label="none")
                axins.errorbar(magnitudes[bin] + 0.35, data_2[:, 4][(mag_bins + 1):(mag_bins + 1) * 2][bin],
                                yerr=data_2[:, 5][(mag_bins + 1):(mag_bins + 1) * 2][bin], fmt='+', capsize=2,
                                label="shape")
                axins.errorbar(magnitudes[bin] + 0.45, data_2[:, 4][(mag_bins + 1) * 2:(mag_bins + 1) * 3][bin],
                                yerr=data_2[:, 5][(mag_bins + 1) * 2:(mag_bins + 1) * 3][bin], fmt='+', capsize=2,
                                label="both")
                axins.errorbar(magnitudes[bin] + 0.55, data_p[:, 20][bin], yerr=data_p[:, 21][bin], fmt="+", capsize=2, label="RM")
                ylim = axs.get_ylim()
                y_delta = ylim[1]-ylim[0]
                ip = InsetPosition(axs, [0.02 + 0.2 * bin, 1 - np.abs(axins.get_ylim()[0] - ylim[1]) / y_delta - 0.3, 0.15, 0.25])
                axins.set_axes_locator(ip)
                axins.spines[["left", "right", "bottom", "top"]].set_alpha(0.5)
                rect, lines = axs.indicate_inset_zoom(axins)
                lines[0].set(visible=True)
                lines[1].set(visible=True)
                lines[2].set(visible=True)
                lines[3].set(visible=True)
            axs.set_xlabel("$m_{\mathrm{GEMS}}$")
            axs.set_ylabel("$\mu$-bias")
            axs.set_xlim(20.5, magnitudes[-2])
            # axs.grid(True)


    elif biases == "C":
        if type == "RP":
            axs.set_title("Random positions")
            axs.errorbar(magnitudes+0.3, (data[:,6][0:(mag_bins+1)*3:3]+data[:,6][(mag_bins+1)*3::3])/2, yerr=np.sqrt(data[:,7][0:(mag_bins+1)*3:3]**2+data[:,7][(mag_bins+1)*3::3]**2)/2, fmt='+', capsize=2, label="none")
            axs.errorbar(magnitudes+0.4, data[:,6][1:(mag_bins+1)*3:3], yerr=data[:,7][1:(mag_bins+1)*3:3], fmt='+', capsize=2, label="shape local")
            axs.errorbar(magnitudes+0.5, data[:,6][2:(mag_bins+1)*3:3], yerr=data[:,7][2:(mag_bins+1)*3:3], fmt='+', capsize=2, label="both local")
            axs.errorbar(magnitudes + 0.6, data[:, 6][(mag_bins+1)*3+1::3], yerr=data[:, 7][(mag_bins+1)*3+1::3], fmt='+', capsize=2, label="shape global")
            axs.errorbar(magnitudes + 0.7, data[:, 6][(mag_bins+1)*3+2::3], yerr=data[:, 7][(mag_bins+1)*3+2::3], fmt='+', capsize=2, label="both global")
            axs.errorbar(magnitudes +0.8, data_pu[:,18], yerr=data_pu[:,19], fmt="+", capsize=2, label="RM")
            axs.set_xlabel("$m_{\mathrm{AUTO}}$")
            axs.set_ylabel("$c$-bias")
            axs.set_xlim(20.5, magnitudes[-2])

        elif type == "GR":
            axs.set_title("Grid")
            axs.errorbar(magnitudes + 0.3, data_2[:, 6][0:(mag_bins+1)], yerr=data_2[:, 7][0:(mag_bins+1)], fmt='+', capsize=2,
                            label="none")
            axs.errorbar(magnitudes + 0.4, data_2[:, 6][(mag_bins+1):(mag_bins+1)*2], yerr=data_2[:, 7][(mag_bins+1):(mag_bins+1)*2], fmt='+', capsize=2,
                           label="shape")
            axs.errorbar(magnitudes + 0.5, data_2[:, 6][(mag_bins+1)*2:(mag_bins+1)*3], yerr=data_2[:, 7][(mag_bins+1)*2:(mag_bins+1)*3], fmt='+', capsize=2,
                            label="both")
            axs.errorbar(magnitudes + 0.6, data_p[:, 22], yerr=data_p[:, 23], fmt="+", capsize=2, label="RM")
            axs.set_xlabel("$m_{\mathrm{GEMS}}$")
            axs.set_ylabel("$c$-bias")
            axs.set_ylim(-0.0010, 0.0025)
            # axs.grid(True)
    #

            axs.set_xlim(20.5,magnitudes[-2])

    axs.legend(prop={'size': 6})

    # axs[0].set_ylim(-0.25, 0.1)
    # axs[1].set_ylim(-0.25, 0.1)

    fig.tight_layout()
    fig.savefig(f"{file_name.split('.')[0]}_{biases}.pdf", dpi=200)
    plt.close(fig)