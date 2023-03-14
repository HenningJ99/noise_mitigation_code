import pickle
import sys
import functions as fct
import numpy as np
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('config_rp.ini')

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])

mag_bins = int(simulation["bins_mag"])
min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

path1 = sys.argv[1]
path2 = sys.argv[2]
scene = int(sys.argv[3]) - 1

arrays1 = pickle.load(open(path1, "rb"))
arrays2 = pickle.load(open(path2, "rb"))

if simulation.getboolean("summarize_pujol"):
    division = 2
else:
    division = 1

BOOTSTRAP_REPETITIONS = 1000

meas0_averages = arrays1[0] - arrays2[0]
meas1_averages = arrays1[1] - arrays2[1]
meas0_weights = arrays1[2] + arrays2[2]
meas1_weights = arrays1[3] + arrays2[2]

biases = []
errors = []

magnitudes_list = [min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)]

for mag in range(mag_bins + 1):
    shear_diff = float(simulation["same_but_shear_diff"])

    bias_data = fct.bootstrap_puyol(meas1_averages[mag][:int((scene + 1) / division)],
                                    meas0_averages[mag][:int((scene + 1) / division)], BOOTSTRAP_REPETITIONS,
                                    shear_diff, meas1_weights[mag][:int((scene + 1) / division)],
                                    meas0_weights[mag][:int((scene + 1) / division)])

    bias = (np.average(meas1_averages[mag][:int((scene + 1) / division)],
                       weights=meas1_weights[mag][:int((scene + 1) / division)]) -
            np.average(meas0_averages[mag][:int((scene + 1) / division)],
                       weights=meas0_weights[mag][:int((scene + 1) / division)])) / shear_diff

    err = bias_data[1]

    err_err = np.std([fct.bootstrap_puyol(meas1_averages[mag][:int((scene + 1) / division)],
                                          meas0_averages[mag][:int((scene + 1) / division)],
                                          BOOTSTRAP_REPETITIONS,
                                          shear_diff,
                                          meas1_weights[mag][:int((scene + 1) / division)],
                                          meas0_weights[mag][:int((scene + 1) / division)])[1] for _ in
                      range(10)])

    biases.append(bias)
    errors.append(err)


mm = 1 / 25.4
fig, axs = plt.subplots(figsize=(88*mm, 88*mm))

axs.errorbar(biases, np.array(magnitudes_list) + 0.5 , xerr=err, fmt="+")
axs.vlines(0, min_mag, max_mag + 1, linestyles="dashed", alpha=0.5, color="C1")
axs.set_xlabel("$\Delta \mu$")
axs.set_ylabel("$MAG_\mathrm{AUTO}$")

fig.savefig("/mnt/EXTERN/Mastersemester/Masterarbeit/Noise_Mitigation_Code/Simulations/output/plots/variable_shear_diff.pdf", dpi=300, bbox_inches="tight")



