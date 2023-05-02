import numpy as np
import matplotlib.pyplot as plt
import sys
import configparser
from matplotlib.ticker import ScalarFormatter

filename = "data/FigA1_Right.txt"

shear_interval = 0.1

mag_bins = 5

data = np.genfromtxt(filename, delimiter="\t")

mag_max = 25.5
mag_min = 20.5

if mag_bins != 0:
    interval = (mag_max - mag_min) / (mag_bins)
    magnitudes = [mag_min + (k + 0.5) * interval for k in range(mag_bins)]
else:
    magnitudes = [mag_max]  # If no binning at all

mm = 1 / 25.4  # millimeter in inches

fig, axs = plt.subplots(constrained_layout=True, figsize=(83 * mm, 88 * mm))

no_cancel = data[0::4]
shape_noise = data[1::4]
both_noise = data[2::4]
response = data[3::4]

axs.set_title(f"[$-${shear_interval}, {shear_interval}]")


axs.plot(magnitudes, shape_noise[:, 5][:-1], 's-', label="shape", color="blue")

axs.plot(magnitudes, both_noise[:, 5][:-1], "^-", label="both", color="orange")

axs.plot(magnitudes, response[:, 5][:-1], "v-", label="RM", color="green")

axs.set_yscale('log')

axs.set_xlabel(r'$m_\mathrm{GEMS}$')
axs.set_ylim(0.1, 100)
axs.set_yticklabels([])
axs.legend()

fig.savefig('FigA1_Right.pdf', dpi=300, bbox_inches='tight')
