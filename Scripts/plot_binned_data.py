import numpy as np
import matplotlib.pyplot as plt
import sys
import configparser
from matplotlib.ticker import ScalarFormatter
from astropy.io import ascii

filename = sys.argv[1]
type = sys.argv[4]
config_file = sys.argv[2]

shear_interval = float(sys.argv[3])
# Read Config File
config = configparser.ConfigParser()
config.read(config_file)

simulation = config['SIMULATION']
mag_bins = int(simulation["bins_mag"])
label = simulation["bin_type"]

data = ascii.read(filename)

mag_max = float(simulation["max_mag"])
mag_min = float(simulation["min_mag"])

if mag_bins != 0:
    interval = (mag_max - mag_min) / (mag_bins)
    magnitudes = [mag_min + (k + 0.5) * interval for k in range(mag_bins)]
else:
    magnitudes = [mag_max]  # If no binning at all

mm = 1 / 25.4  # millimeter in inches

if type == "RP":
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(176*mm, 88*mm), sharey="row")
    # magnitudes = magnitudes[:-1]
    linestyles = ["solid", "dotted"]
    for i in range(4):
        no_cancel = data[4 * i * (mag_bins + 1):4 * (i + 1) * (mag_bins + 1):4]
        shape_noise = data[4 * i * (mag_bins + 1) + 1:4 * (i + 1) * (mag_bins + 1):4]
        both_noise = data[4 * i * (mag_bins + 1) + 2:4 * (i + 1) * (mag_bins + 1):4]
        response = data[4 * i * (mag_bins + 1) + 3:4 * (i + 1) * (mag_bins + 1):4]

        axs[i % 2].set_title(f"[$-${shear_interval}, {shear_interval}]")
        # axs[i % 2].errorbar(magnitudes, shape_noise[:, 5][:-1], yerr=shape_noise[:, 6][:-1], fmt='+' if i < 2 else "x", capsize=2, elinewidth=0.5,
        #              color="blue" if i < 2 else "navy", label="shape local" if i < 2 else "shape global", markersize=4)

        axs[i % 2].plot(magnitudes, shape_noise["col7"][:-1], '+-' if i < 2 else "x-", color="blue" if i < 2 else "navy", label="shape local" if i < 2 else "shape global")

        # axs[i % 2].errorbar(magnitudes, both_noise[:, 5][:-1], yerr=both_noise[:, 6][:-1], fmt='^' if i < 2 else "v", capsize=2, elinewidth=0.5,
        #              color="orange" if i < 2 else "orangered", label="both local" if i < 2 else "both global", markersize=4)

        axs[i % 2].plot(magnitudes, both_noise["col7"][:-1], '^-' if i < 2 else "v-", color="orange" if i < 2 else "orangered", label="both local" if i < 2 else "both global")

        if i < 2:
            # axs[i % 2].errorbar(magnitudes, response[:, 5][:-1], yerr=response[:, 6][:-1], fmt='p', capsize=2, elinewidth=0.5,
            #              color="green", label="RM", markersize=4)
            axs[i % 2].plot(magnitudes, response["col7"][:-1], 'p-', color="green", label="RM")

        axs[0].set_yscale('log')
        axs[1].set_yscale('log')

        axs[0].legend(prop={'size': 6})
        axs[1].legend(prop={'size': 6})

        axs[0].set_xlabel(r'$m_\mathrm{AUTO}$')
        axs[1].set_xlabel(r'$m_\mathrm{GEMS}$')


        axs[0].set_ylabel('Runtime improvement')
        fig.savefig(filename.split('.')[0] + '.pdf', dpi=300, bbox_inches='tight')
elif type == "GRID":
    fig, axs = plt.subplots(constrained_layout=True, figsize=(88 * mm, 88 * mm))

    no_cancel = data[0::4]
    shape_noise = data[1::4]
    both_noise = data[2::4]
    response = data[3::4]

    axs.set_title(f"[$-${shear_interval}, {shear_interval}]")
    # axs.errorbar(magnitudes, shape_noise[:, 5][:-1], yerr=shape_noise[:, 6][:-1], fmt='s', capsize=2,
    #                     elinewidth=0.5,
    #                     color="blue", label="shape", markersize=4)

    axs.plot(magnitudes, shape_noise["col6"][:-1], 's-', label="shape", color="blue")

    # axs.errorbar(magnitudes, both_noise[:, 5][:-1], yerr=both_noise[:, 6][:-1], fmt='^', capsize=2,
    #                     elinewidth=0.5,
    #                     color="orange", label="both", markersize=4)

    axs.plot(magnitudes, both_noise["col6"][:-1], "^-", label="both", color="orange")


    # axs.errorbar(magnitudes, response[:, 5][:-1], yerr=response[:, 6][:-1], fmt='v', capsize=2,
    #                     elinewidth=0.5,
    #                     color="green", label="RM", markersize=4)
    axs.plot(magnitudes, response["col6"][:-1], "v-", label="RM", color="green")

    axs.set_yscale('log')

    axs.legend(prop={'size': 6})
    axs.set_xlabel(r'$m_\mathrm{GEMS}$')


    axs.set_ylabel('Runtime improvement')
    fig.savefig(filename.split('.')[0] + '.pdf', dpi=300, bbox_inches='tight')
