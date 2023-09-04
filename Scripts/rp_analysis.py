from astropy.table import Table
from astropy.io import ascii
import numpy as np
import sys
import timeit
import configparser

# Define local paths
path = sys.argv[3] + "/"

subfolder = sys.argv[5]

config = configparser.ConfigParser()
config.read('config_rp.ini')

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])

mag_bins = int(simulation["bins_mag"])

if sys.argv[6] == "GEMS":
    bin_type = "mag_gems"
elif sys.argv[6] == "MAG_AUTO":
    bin_type = "mag_auto"

min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

random_seed = int(simulation['random_seed'])
pixel_scale = float(image['pixel_scale'])
exp_time = float(image['exp_time'])
gain = float(image['gain'])
read_noise = float(image['read_noise'])
sky = float(image['sky'])
zp = float(image['zp'])

lam = float(psf_config['lam_min'])
step_psf = float(psf_config['step_psf'])
tel_diam = float(psf_config['tel_diam'])

ellip_rms = float(simulation['ellip_rms'])
ellip_max = float(simulation['ellip_max'])

stamp_xsize = int(image['stamp_xsize'])
stamp_ysize = int(image['stamp_ysize'])
ssamp_grid = int(image['ssamp_grid'])

shear_bins = int(simulation['shear_bins'])

shear_min = -float(sys.argv[4])
shear_max = float(sys.argv[4])

# galaxy_number = int(sys.argv[2])

complete_image_size = int(sys.argv[1])

total_scenes_per_shear = int(sys.argv[2])

magnitudes_list = np.array([min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)])
mag_auto_bin = magnitudes_list
# --------------------------------------- ANALYSIS -------------------------------------------------------------------
start1 = timeit.default_timer()

data_complete = ascii.read(subfolder + 'shear_catalog.dat', fast_reader={'chunk_size': 100 * 1000000})

# SN CUT
data_complete = data_complete[data_complete["S/N"] > float(simulation["sn_cut"])]

print(
    f"Outlier with shears larger 5 in percent: {100 * len(data_complete['meas_g1'][(data_complete['meas_g1'] >= 5) | (data_complete['meas_g1'] <= -5)]) / len(data_complete['meas_g1']):.5f}")
# Outliers
data_complete = data_complete[(data_complete["meas_g1"] < 5) & (data_complete["meas_g1"] > -5)]

# Magnitude cut
data_complete = data_complete[(data_complete[bin_type] > min_mag) & (data_complete[bin_type] < max_mag)]


# Define function for the standard error to use by aggregate
def std_error(array):
    return np.std(array) / np.sqrt(array.size)


# Use astropy to group arrays and calculate mean ellipticity for each scene, cancel index, shear index and magnitude
data_complete['binned_mag'] = np.trunc(data_complete[bin_type] + 0.5)  # Adding a row with the magnitudes in bins
meas_comp = data_complete.group_by(['scene_index', 'cancel_index', 'shear_index', 'binned_mag'])
meas_means = meas_comp['scene_index', 'cancel_index', 'shear_index', 'binned_mag', 'meas_g1'].groups.aggregate(np.mean)
meas_std = meas_comp['scene_index', 'cancel_index', 'shear_index', 'binned_mag', 'meas_g1'].groups.aggregate(std_error)
lengths = meas_comp['scene_index', 'cancel_index', 'shear_index', 'binned_mag', 'meas_g1'].groups.aggregate(np.size)

full_impr = data_complete.group_by(['scene_index', 'cancel_index', 'shear_index'])
meas_means_full = full_impr['meas_g1'].groups.aggregate(np.mean)
meas_std_full = full_impr['meas_g1'].groups.aggregate(std_error)
lengths_full = full_impr['meas_g1'].groups.aggregate(np.size)

# WRITE RESULTS TO FILE
columns = []
for scene in range(total_scenes_per_shear):
    index = 0
    print(f"{scene + 1} / {total_scenes_per_shear}")
    for j in range(4):
        for i in range(shear_bins):
            for mag in range(mag_bins + 1):
                if mag == mag_bins:
                    lower_limit = min_mag
                    upper_limit = max_mag
                else:
                    lower_limit = magnitudes_list[mag]
                    upper_limit = magnitudes_list[mag + 1]
                start = timeit.default_timer()

                g1 = shear_min + i * (shear_max - shear_min) / (shear_bins - 1)

                if mag != mag_bins:
                    value = meas_means["meas_g1"][
                        (meas_means["scene_index"] == scene) & (meas_means["cancel_index"] == j) & (
                                    meas_means["shear_index"] == i) & (
                                    meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data
                    error = meas_std["meas_g1"][
                        (meas_means["scene_index"] == scene) & (meas_means["cancel_index"] == j) & (
                                meas_means["shear_index"] == i) & (
                                meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data
                    weight = lengths["meas_g1"][
                        (meas_means["scene_index"] == scene) & (meas_means["cancel_index"] == j) & (
                                meas_means["shear_index"] == i) & (
                                meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data

                    if len(value) == 0:
                        value = -1
                        error = -1
                        weight = 0
                    else:
                        value = value[0]
                        error = error[0]
                        weight = weight[0]

                    columns.append([g1, i, scene, j, value, error,
                                    weight, magnitudes_list[mag]])

                else:
                    ind = scene * shear_bins * 4 + j * shear_bins + i

                    columns.append([g1, i, scene, j, meas_means_full[ind], meas_std_full[ind],
                                    lengths_full[ind], magnitudes_list[mag]])

        index += 1

columns = np.array(columns, dtype=float)
lf_results = Table([columns[:, i] for i in range(8)],
                   names=('g1', 'shear_index', 'scene_index', 'cancel_index', 'mean_g1', 'std_g1', 'weight', bin_type))
lf_results = lf_results.group_by('scene_index')

ascii.write(lf_results, subfolder + 'analysis.dat',
            overwrite=True)
