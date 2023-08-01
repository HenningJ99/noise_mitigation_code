import numpy as np
import matplotlib.pyplot as plt
import math
from astropy.table import Table
from astropy.io import ascii
from scipy.optimize import curve_fit
import sys
import configparser
import os


def cubic_function(x, a, b, c):
    return a * np.sign(x) * x**2 + b * x + c


data_complete = ascii.read("data/Fig7.dat")

objects = 3200000
galaxies = 160000
tiles = 2
ring_num = 2
runtime = -1  # int(read_in[5].split('.')[0])

noise_plus_meas = 0.25

mag_bins = 4
time_bins = 10

used_meas = "_" + "mod"


data = data_complete[4:: (mag_bins + 1)]

mm = 1 / 25.4  # millimeter in inches


fig, axs = plt.subplots(figsize=(88 * mm, 88 * mm))


deviation = data["meas_g1" + used_meas] - data["input_g1"]

axs.errorbar(data["input_g1"], deviation, \
             yerr=data["meas_g1" + used_meas + "_err"], fmt='+', capsize=2,
             elinewidth=0.5)

popt, pcov = curve_fit(cubic_function, data["input_g1"],
                       deviation, \
                       sigma=data["meas_g1" + used_meas + "_err"],
                       absolute_sigma=True)

error = np.sqrt(np.diag(pcov))

r = deviation - cubic_function(data["input_g1"], *popt)

chisq = np.sum((r / data["meas_g1" + used_meas + "_err"]) ** 2)
chisq_red = chisq / (len(r) - len(popt))
axs.plot(data["input_g1"], cubic_function(data["input_g1"], *popt))
axs.set_xlabel("$g_1^\mathrm{t}$")
axs.set_ylabel("$<g_1^\mathrm{obs}>-g_1^\mathrm{t}$")


textstr = '\n'.join((r'$\alpha \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, = %.2f \pm %.2f$' % (popt[0], error[0]),
                             r'$\mu\,[10^{-3}]= %.2f \pm %.2f$' % (1e3 * popt[1], 1e3 *error[1]),
                             r'$c\,[10^{-4}] = %.2f \pm %.2f$' % (1e4 * popt[2], 1e4 * error[2]),
                             r'$\chi_\mathrm{red}^2\,\,\,\,\,\,\,\,\,\,\,\,\, = %.2f$' % (chisq_red)))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs.text(0.05, 0.95, textstr, transform=axs.transAxes, fontsize=8,
         verticalalignment='top', bbox=props)


fig.savefig("Fig7_Bot.pdf", dpi=300, bbox_inches='tight')

plt.close()

