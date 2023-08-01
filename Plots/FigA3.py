import numpy as np
import matplotlib.pyplot as plt
import sys
import configparser
from matplotlib.ticker import ScalarFormatter

filename = "data/FigA3.txt"

shear_interval = 0.02

mag_bins = 6

data = np.genfromtxt(filename, delimiter="\t")

mag_max = 26.5
mag_min = 20.5

if mag_bins != 0:
    interval = (mag_max - mag_min) / (mag_bins)
    magnitudes = [mag_min + (k + 0.5) * interval for k in range(mag_bins)]
else:
    magnitudes = [mag_max]  # If no binning at all

mm = 1 / 25.4  # millimeter in inches


fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(176*mm, 88*mm), sharey="row")
# magnitudes = magnitudes[:-1]
linestyles = ["solid", "dotted"]
for i in range(4):
    no_cancel = data[4 * i * (mag_bins + 1):4 * (i + 1) * (mag_bins + 1):4]
    shape_noise = data[4 * i * (mag_bins + 1) + 1:4 * (i + 1) * (mag_bins + 1):4]
    both_noise = data[4 * i * (mag_bins + 1) + 2:4 * (i + 1) * (mag_bins + 1):4]
    response = data[4 * i * (mag_bins + 1) + 3:4 * (i + 1) * (mag_bins + 1):4]

    axs[i % 2].set_title(f"[$-${shear_interval}, {shear_interval}]")

    axs[i % 2].plot(magnitudes, shape_noise[:, 5][:-1], '+-' if i < 2 else "x-", color="C0" if i < 2 else "C1", label="shape local" if i < 2 else "shape global")

    axs[i % 2].plot(magnitudes, both_noise[:, 5][:-1], '^-' if i < 2 else "v-", color="C2" if i < 2 else "C3", label="both local" if i < 2 else "both global")

    if i < 2:

        axs[i % 2].plot(magnitudes, response[:, 5][:-1], 'p-', color="C4", label="RM")

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    axs[0].legend()
    #axs[1].legend(prop={'size':8})

    axs[0].set_xlabel(r'$m_\mathrm{AUTO}$')
    axs[1].set_xlabel(r'$m_\mathrm{GEMS}$')


    axs[0].set_ylabel('Runtime improvement')
    fig.savefig("FigA3.pdf", dpi=300, bbox_inches='tight')
