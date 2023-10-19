from astropy.table import Table
from astropy.io import ascii
import numpy as np
import sys
import os
import timeit
import configparser
import functions as fct
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import pickle

# ---------------------------- INITIALIZATION --------------------------------------------------------------------------

bootstrap_fit = True # Takes ages

path = sys.argv[4] + "/"
subfolder = sys.argv[5]
if not os.path.isdir(subfolder + "/plots"):
    os.mkdir(subfolder + "/plots")
complete_image_size = int(sys.argv[1])
num_shears = int(sys.argv[3])
total_scenes_per_shear = int(sys.argv[2])
input_catalog = ascii.read(subfolder + 'input_catalog.dat')
galaxy_number = len(input_catalog) / total_scenes_per_shear
print(f"Average galaxy number: {galaxy_number}")
# Time the program
start = timeit.default_timer()


def linear_function(x, a, b):
    return a * x + b


def bootstrap(array, weights, n):
    indices = np.random.choice(np.arange(len(array)), size=(n, len(array)))
    bootstrap = np.take(array, indices, axis=0).reshape(n, -1)
    weights = np.take(weights, indices, axis=0).reshape(n, -1)

    filter = (np.sum(weights, axis=1) != 0)

    bootstrap = bootstrap[filter]
    weights = weights[filter]

    return np.std(np.average(bootstrap, axis=1, weights=weights))


config = configparser.ConfigParser()
config.read('config_rp.ini')

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])
REPS = int(simulation["reps_for_improvements"])

num_cores = int(simulation["num_cores"])
mag_bins = int(simulation["bins_mag"])

timings = config['TIMINGS']

noise_plus_meas = float(timings["noise_plus_meas"])
scene_creation = float(timings["scene_creation"])

analyse_every = int(simulation["puj_analyse_every"])

if sys.argv[6] == "GEMS":
    bin_type = "mag_gems"
elif sys.argv[6] == "MAG_AUTO":
    bin_type = "mag_auto"

min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

# For the PSF import the vega spectrum
data = np.genfromtxt(path + "input/vega_spectrumdat.sec")

# Define the shear arrays
if num_shears == 2:
    shears = [-0.02, 0.02]
else:
    if simulation.getboolean("same_but_shear"):
        shears = [-0.025 + 0.05 / (num_shears - 1) * k for k in range(num_shears)]
    else:
        shears = [-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)]
# ------------------------------- ANALYSIS ----------------------------------------------------------------------------
start1 = timeit.default_timer()

data_complete = ascii.read(subfolder + "shear_catalog.dat", fast_reader={'chunk_size': 100 * 1000000})
#input_catalogs = ascii.read(subfolder + "input_catalog.dat")

# S/N Cut
data_complete = data_complete[data_complete["S/N"] > float(simulation["sn_cut"])]

print(f"Outlier with shears larger than 5 in percent {100 * len(data_complete['meas_g1'][(data_complete['meas_g1'] >= 5) | (data_complete['meas_g1'] <= -5)]) / len(data_complete['meas_g1']):.5f}")
# Outliers
data_complete = data_complete[(data_complete["meas_g1"] < 5) & (data_complete["meas_g1"] > -5)]

# Magnitude cut
data_complete = data_complete[(data_complete[bin_type] > min_mag) & (data_complete[bin_type] < max_mag)]

magnitudes_list = [min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)]

# ------------------------------ INPUT SHEAR ------------------------------------------------------------------------#
# input_shears = [[[] for _ in range(mag_bins + 1)] for _ in range(num_shears)]
# input_weights = [[[] for _ in range(mag_bins + 1)] for _ in range(num_shears)]
# for j in range(num_shears):
#     for i in range(total_scenes_per_shear):
#         for mag in range(len(magnitudes_list)):
#             if mag == len(magnitudes_list) - 1:
#                 upper_limit = max_mag
#                 lower_limit = min_mag
#             else:
#                 upper_limit = magnitudes_list[mag + 1]
#                 lower_limit = magnitudes_list[mag]
#
#             matched_sources = input_catalogs[input_catalogs["scene_index"] == i][
#                 np.array(data_complete[(data_complete["scene_index"] == i) &
#                                        (data_complete["shear_index"] == j) &
#                                        (data_complete[bin_type] > lower_limit) & (data_complete[bin_type] < upper_limit)][
#                              "matching_index_optimized"], dtype=int)]
#             if len(matched_sources) != 0:
#                 input_g1 = matched_sources["e"] * np.cos(2 * matched_sources["beta"])
#                 input_shears[j][mag].append(np.average(input_g1))
#                 input_weights[j][mag].append(len(input_g1))
#             else:
#                 input_shears[j][mag].append(-1)
#                 input_weights[j][mag].append(0)
#
# input_shears = np.array(input_shears)
# input_weights = np.array(input_weights)

masks = []
# For uncertainty behaviour analyse after every run
scene = total_scenes_per_shear - 1

meas1_averages_small = [[[] for _ in range(len(magnitudes_list))] for _ in range(10)]
meas0_averages_small = [[[] for _ in range(len(magnitudes_list))] for _ in range(10)]
meas1_weights_small = [[[] for _ in range(len(magnitudes_list))] for _ in range(10)]
meas0_weights_small = [[[] for _ in range(len(magnitudes_list))] for _ in range(10)]

meas1_averages = [[] for _ in range(len(magnitudes_list))]
meas0_averages = [[] for _ in range(len(magnitudes_list))]
meas1_weights = [[] for _ in range(len(magnitudes_list))]
meas0_weights = [[] for _ in range(len(magnitudes_list))]

meas_averages = [[] for _ in range(len(magnitudes_list))]
meas_averages_small = [[[] for _ in range(len(magnitudes_list))] for i in range(10)]
meas_averages_small_ = [[[] for _ in range(len(magnitudes_list))] for i in range(10)]
meas_weights = [[] for _ in range(len(magnitudes_list))]
meas_weights_small = [[[] for _ in range(len(magnitudes_list))] for i in range(10)]
meas_weights_small_ = [[[] for _ in range(len(magnitudes_list))] for i in range(10)]
start_sorting = timeit.default_timer()

def std_error(array):
    return np.std(array) / np.sqrt(array.size)

data_complete['binned_mag'] = np.trunc(data_complete[bin_type] + 0.5)  # Adding a row with the magnitudes in bins
meas_comp = data_complete.group_by(["scene_index", "shear_index", "binned_mag"])
meas_means = meas_comp["scene_index", "shear_index", "binned_mag", "meas_g1"].groups.aggregate(np.mean)
meas_stderr = meas_comp["scene_index", "shear_index", "binned_mag", "meas_g1"].groups.aggregate(std_error)
meas_lengths = meas_comp["scene_index", "shear_index", "binned_mag", "meas_g1"].groups.aggregate(np.size)

full_meas = data_complete.group_by(["scene_index", "shear_index"])
meas_means_full = full_meas["meas_g1"].groups.aggregate(np.mean)
meas_stderr_full = full_meas["meas_g1"].groups.aggregate(std_error)
meas_lengths_full = full_meas["meas_g1"].groups.aggregate(np.size)

for mag in range(len(magnitudes_list)):
    if mag == len(magnitudes_list) - 1:
        upper_limit = max_mag
        lower_limit = min_mag
    else:
        upper_limit = magnitudes_list[mag + 1]
        lower_limit = magnitudes_list[mag]


    print(len(data_complete["meas_g1"][
                  (data_complete["scene_index"] <= scene) & (data_complete[bin_type] > lower_limit) & (
                              data_complete[bin_type] < upper_limit)]))
    if len(data_complete["meas_g1"][
               (data_complete["scene_index"] <= scene) & (data_complete[bin_type] > lower_limit) & (
                       data_complete[bin_type] < upper_limit)]) != 0:
        meas_averages_ = []
        meas_weights_ = []

        if mag != len(magnitudes_list) -1:
            for i in range(num_shears):
                for o in range(scene+1):
                    value = meas_means["meas_g1"][(meas_means["scene_index"] == o) & (meas_means["shear_index"] == i) & (meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data
                    weight = meas_lengths["meas_g1"][
                        (meas_means["scene_index"] == o) & (meas_means["shear_index"] == i) & (
                                    meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data
                    if len(value) == 0:
                        value = -1
                        weight = 0
                    else:
                        value = value[0]
                        weight = weight[0]

                    meas_averages_.append(value)
                    meas_weights_.append(weight)
        else:
            meas_averages_ = meas_means_full
            meas_weights_ = meas_lengths_full

        meas0_averages_ = np.delete(meas_averages_, np.s_[num_shears-1::num_shears])
        meas0_weights_ = np.delete(meas_weights_, np.s_[num_shears-1::num_shears])

        meas1_averages_ = np.delete(meas_averages_, np.s_[::num_shears])
        meas1_weights_ = np.delete(meas_weights_, np.s_[::num_shears])

        if num_shears == 11:
            for interval in range(10):
                meas1_averages_small_ = np.array(
                    meas1_averages_[interval:len(meas1_averages_):(num_shears - 1)])
                meas0_averages_small_ = np.array(
                    meas0_averages_[interval:len(meas0_averages_):(num_shears - 1)])

                meas1_weights_small_ = np.array(meas1_weights_[interval :len(meas1_weights_):(num_shears - 1)])
                meas0_weights_small_ = np.array(
                    meas0_weights_[interval :len(meas0_weights_):(num_shears - 1)])

                if simulation.getboolean("summarize_pujol"):
                    # Summarize always the two versions belonging to each other
                    meas1_averages_small[interval][mag].append(np.array([(meas1_averages_small_[2 * i] * meas1_weights_small_[2 * i] +
                                                                meas1_averages_small_[2 * i + 1] * meas1_weights_small_[
                                                                    2 * i + 1]) / (
                                                                       meas1_weights_small_[2 * i] + meas1_weights_small_[
                                                                   2 * i + 1]) if (meas1_weights_small_[2 * i] +
                                                                                   meas1_weights_small_[2 * i + 1]) != 0 else -1
                                                               for i in range(int(len(meas1_averages_small_) / 2))]))

                    meas1_weights_small[interval][mag].append(np.sum(np.array(meas1_weights_small_).reshape(-1, 2), axis=1))

                    meas0_averages_small[interval][mag].append(np.array([(meas0_averages_small_[2 * i] * meas0_weights_small_[2 * i] +
                                                                meas0_averages_small_[2 * i + 1] * meas0_weights_small_[
                                                                    2 * i + 1]) / (
                                                                       meas0_weights_small_[2 * i] + meas0_weights_small_[
                                                                   2 * i + 1]) if (meas0_weights_small_[2 * i] +
                                                                                   meas0_weights_small_[2 * i + 1]) != 0 else -1
                                                               for i in range(int(len(meas0_averages_small_) / 2))]))

                    meas0_weights_small[interval][mag].append(np.sum(np.array(meas0_weights_small_).reshape(-1, 2), axis=1))

                else:
                    meas1_averages_small[interval][mag].append(meas1_averages_small_)
                    meas1_weights_small[interval][mag].append(meas1_weights_small_)
                    meas0_averages_small[interval][mag].append(meas0_averages_small_)
                    meas0_weights_small[interval][mag].append(meas0_weights_small_)

        # For 11 shears build averages of the 10 response estimates first to have uncorrelated error estimates
        meas0_averages_ = [np.average(meas0_averages_[i * (num_shears - 1):(i + 1) * (num_shears - 1)],
                                      weights=meas0_weights_[i * (num_shears - 1):(i + 1) * (num_shears - 1)])
                           if np.sum(meas0_weights_[i * (num_shears - 1):(i + 1) * (num_shears - 1)]) != 0 else -1 for i
                           in
                           range(scene + 1)]

        meas1_averages_ = [np.average(meas1_averages_[i * (num_shears - 1):(i + 1) * (num_shears - 1)],
                                      weights=meas1_weights_[i * (num_shears - 1):(i + 1) * (num_shears - 1)])
                           if np.sum(meas1_weights_[i * (num_shears - 1):(i + 1) * (num_shears - 1)]) != 0 else -1 for i
                           in
                           range(scene + 1)]
        # print(meas0_averages, meas1_averages)

        if num_shears == 11:
            for interval in range(10):
                meas_averages_small_[interval] = [
                    np.average([meas_averages_[interval + i * (num_shears)], meas_averages_[interval+1 + i * (num_shears)]], weights=
                    [meas_weights_[interval + i * (num_shears)], meas_weights_[interval + 1 + i * (num_shears)]]) if np.sum(
                        [meas_weights_[interval + i * (num_shears)], meas_weights_[interval + 1 + i * (num_shears)]]) != 0 else -1
                    for i in range(scene + 1)]

                meas_weights_small_[interval] = [np.sum([meas_weights_[interval + i * (num_shears)], meas_weights_[interval + 1 + i * (num_shears)]])
                                       for i
                                       in
                                       range(scene + 1)]
        # print(meas0_averages)
        meas_averages_ = [np.average(meas_averages_[i * (num_shears):(i + 1) * (num_shears)],
                                     weights=meas_weights_[i * (num_shears):(i + 1) * (num_shears)])
                          if np.sum(meas_weights_[i * (num_shears):(i + 1) * (num_shears)]) != 0 else -1 for i in
                          range(scene + 1)]

        meas0_weights_ = [np.sum(meas0_weights_[i * (num_shears - 1):(i + 1) * (num_shears - 1)]) for i in
                          range(scene + 1)]
        # print(meas0_weights)
        meas1_weights_ = [np.sum(meas1_weights_[i * (num_shears - 1):(i + 1) * (num_shears - 1)]) for i in
                          range(scene + 1)]


        meas_weights_ = [np.sum(meas_weights_[i * (num_shears):(i + 1) * (num_shears)]) for i in range(scene + 1)]

        if simulation.getboolean("summarize_pujol"):

            division = 2
            # Summarize always the two versions belonging to each other
            meas_averages[mag].append(np.array([(meas_averages_[2 * i] * meas_weights_[2 * i] +
                                                 meas_averages_[2 * i + 1] * meas_weights_[2 * i + 1]) / (
                                                        meas_weights_[2 * i] + meas_weights_[
                                                    2 * i + 1]) if (meas_weights_[2 * i] +
                                                                    meas_weights_[2 * i + 1]) != 0 else -1
                                                for i in range(int(len(meas_averages_) / 2))]))

            meas_weights[mag].append(np.sum(np.array(meas_weights_).reshape(-1, 2), axis=1))

            if num_shears == 11:
                for interval in range(10):
                    meas_averages_small[interval][mag].append(np.array([(meas_averages_small_[interval][2 * i] * meas_weights_small_[interval][2 * i] +
                                                               meas_averages_small_[interval][2 * i + 1] * meas_weights_small_[interval][
                                                                   2 * i + 1]) / (
                                                                      meas_weights_small_[interval][2 * i] + meas_weights_small_[interval][
                                                                  2 * i + 1]) if (meas_weights_small_[interval][2 * i] +
                                                                                  meas_weights_small_[interval][2 * i + 1]) != 0 else -1
                                                              for i in range(int(len(meas_averages_small_[interval]) / 2))]))

                    meas_weights_small[interval][mag].append(np.sum(np.array(meas_weights_small_[interval]).reshape(-1, 2), axis=1))

            meas1_averages[mag].append(np.array([(meas1_averages_[2 * i] * meas1_weights_[2 * i] +
                                                  meas1_averages_[2 * i + 1] * meas1_weights_[2 * i + 1]) / (
                                                         meas1_weights_[2 * i] + meas1_weights_[
                                                     2 * i + 1]) if (meas1_weights_[2 * i] +
                                                                     meas1_weights_[2 * i + 1]) != 0 else -1
                                                 for i in range(int(len(meas1_averages_) / 2))]))

            meas1_weights[mag].append(np.sum(np.array(meas1_weights_).reshape(-1, 2), axis=1))

            meas0_averages[mag].append(np.array([(meas0_averages_[2 * i] * meas0_weights_[2 * i] +
                                                  meas0_averages_[2 * i + 1] * meas0_weights_[2 * i + 1]) / (
                                                         meas0_weights_[2 * i] + meas0_weights_[
                                                     2 * i + 1]) if (meas0_weights_[2 * i] +
                                                                     meas0_weights_[2 * i + 1]) != 0 else -1
                                                 for i in range(int(len(meas0_averages_) / 2))]))

            meas0_weights[mag].append(np.sum(np.array(meas0_weights_).reshape(-1, 2), axis=1))

        else:
            division = 1

            meas_averages[mag].append(meas_averages_)
            meas_weights[mag].append(meas_weights_)
            if num_shears == 11:
                for interval in range(10):
                    meas_averages_small[interval][mag].append(meas_averages_small_[interval])
                    meas_weights_small[interval][mag].append(meas_weights_small_[interval])
            meas1_averages[mag].append(meas1_averages_)
            meas1_weights[mag].append(meas1_weights_)
            meas0_averages[mag].append(meas0_averages_)
            meas0_weights[mag].append(meas0_weights_)

        # print(len(meas0_weights), len(meas1_weights))

if num_shears == 11:
    for interval in range(10):
        meas1_averages_small[interval] = np.squeeze(np.array(meas1_averages_small[interval]))
        meas0_averages_small[interval] = np.squeeze(np.array(meas0_averages_small[interval]))
        meas1_weights_small[interval] = np.squeeze(np.array(meas1_weights_small[interval]))
        meas0_weights_small[interval] = np.squeeze(np.array(meas0_weights_small[interval]))
        meas_averages_small[interval] = np.squeeze(np.array(meas_averages_small[interval]))
        meas_weights_small[interval] = np.squeeze(np.array(meas_weights_small[interval]))

meas1_averages = np.squeeze(np.array(meas1_averages))
meas0_averages = np.squeeze(np.array(meas0_averages))
meas1_weights = np.squeeze(np.array(meas1_weights))
meas0_weights = np.squeeze(np.array(meas0_weights))

meas_averages = np.squeeze(np.array(meas_averages))
meas_weights = np.squeeze(np.array(meas_weights))


sorting_time = timeit.default_timer() - start_sorting

start_rest = timeit.default_timer()
output_shears = [[] for _ in range(len(magnitudes_list))]
output_errors = [[] for _ in range(len(magnitudes_list))]
output_weights = [[] for _ in range(len(magnitudes_list))]
columns = []
time_fit = 0

meas1_averages_ = meas1_averages.copy()
meas0_averages_ = meas0_averages.copy()
meas1_weights_ = meas1_weights.copy()
meas0_weights_ = meas0_weights.copy()

meas_averages_ = meas_averages.copy()
meas_weights_ = meas_weights.copy()

if num_shears == 11:
    meas1_averages_small_ = meas1_averages_small.copy()
    meas0_averages_small_ = meas0_averages_small.copy()
    meas1_weights_small_ = meas1_weights_small.copy()
    meas0_weights_small_ = meas0_weights_small.copy()

    meas_averages_small_ = meas_averages_small.copy()
    meas_weights_small_ = meas_weights_small.copy()


pickle.dump([meas0_averages, meas1_averages, meas0_weights, meas1_weights], open(subfolder + "meas_arrays.p", "wb"))
for reps in range(REPS):
    rand_int = np.random.randint(0, int(total_scenes_per_shear / division), size= int(total_scenes_per_shear / division))
    if reps == REPS - 1:
        rand_int = [i for i in range(int(total_scenes_per_shear / division))]


    meas1_averages = np.take(meas1_averages_, rand_int, axis=1)
    meas0_averages = np.take(meas0_averages_, rand_int, axis=1)
    meas1_weights = np.take(meas1_weights_, rand_int, axis=1)
    meas0_weights = np.take(meas1_weights_, rand_int, axis=1)

    meas_averages = np.take(meas_averages_, rand_int, axis=1)
    meas_weights = np.take(meas_weights_, rand_int, axis=1)

    if num_shears == 11:
        meas1_averages_small = np.take(meas1_averages_small_, rand_int, axis=2)
        meas0_averages_small = np.take(meas0_averages_small_, rand_int, axis=2)
        meas1_weights_small = np.take(meas1_weights_small_, rand_int, axis=2)
        meas0_weights_small = np.take(meas1_weights_small_, rand_int, axis=2)

        meas_averages_small = np.take(meas_averages_small_, rand_int, axis=2)
        meas_weights_small = np.take(meas_weights_small_, rand_int, axis=2)


    for scene in range(analyse_every-1, total_scenes_per_shear, analyse_every):
        counter = 0

        for mag in range(len(magnitudes_list)):

            if mag == len(magnitudes_list) - 1:
                upper_limit = max_mag
                lower_limit = min_mag
            else:
                upper_limit = magnitudes_list[mag + 1]
                lower_limit = magnitudes_list[mag]

            if bootstrap_fit:
                start_fit = timeit.default_timer()
                #------ BOOTSTRAP THE FIT FOR THE COMPARISON RM - LF ----------------#
                for div in range(analyse_every):

                    if mag != mag_bins:
                        output_shear = []
                        output_error = []
                        output_weight = []
                        for o in range(num_shears):
                            value = meas_means["meas_g1"][
                                (meas_means["scene_index"] == scene + div - analyse_every + 1) & (meas_means["shear_index"] == o) & (
                                            meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data
                            error = meas_stderr["meas_g1"][
                                (meas_means["scene_index"] == scene + div - analyse_every + 1) & (meas_means["shear_index"] == o) & (
                                        meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data
                            weight = meas_lengths["meas_g1"][
                                (meas_means["scene_index"] == scene + div - analyse_every + 1) & (meas_means["shear_index"] == o) & (
                                        meas_means["binned_mag"] == magnitudes_list[mag] + 0.5)].data

                            if len(value) == 0:
                                value = -1
                                error = -1
                                weight = 0
                            else:
                                value = value[0]
                                error = error[0]
                                weight = weight[0]

                            output_shear.append(value)
                            output_error.append(error)
                            output_weight.append(weight)

                        output_shears[mag].append(output_shear)
                        output_errors[mag].append(output_error)
                        output_weights[mag].append(output_weight)
                    else:
                        output_shears[mag].append(meas_means_full[(scene + div - analyse_every + 1) * num_shears : (scene + div - analyse_every + 2) * num_shears])
                        output_errors[mag].append(meas_stderr_full[(scene + div - analyse_every + 1) * num_shears : (scene + div - analyse_every + 2) * num_shears])
                        output_weights[mag].append(meas_lengths_full[(scene + div - analyse_every + 1) * num_shears : (scene + div - analyse_every + 2) * num_shears])

                indices = np.random.choice(np.arange(0, scene+1, division), size=(BOOTSTRAP_REPETITIONS, int((scene + 1) / division)))

                if division != 1:
                    indices = np.append(indices, indices+1, axis=1)

                bootstrap_array = np.take(output_shears[mag], indices, axis=0).reshape(BOOTSTRAP_REPETITIONS, scene + 1, -1)
                weights_array = np.take(output_weights[mag], indices, axis=0).reshape(BOOTSTRAP_REPETITIONS, scene + 1, -1)
                errors_array = np.take(output_errors[mag], indices, axis=0).reshape(BOOTSTRAP_REPETITIONS, scene + 1, -1)

                filter = weights_array == 0

                bootstrap_array = np.ma.array(bootstrap_array, mask=filter)
                weights_array = np.ma.array(weights_array, mask=filter)
                errors_array = np.ma.array(errors_array, mask=filter)

                for_fitting = np.ma.average(bootstrap_array, weights=weights_array, axis=1)
                uncertainties = np.sqrt(np.sum(np.multiply(np.square(errors_array), np.square(weights_array)), axis=1)) / (np.sum(weights_array, axis=1))

                popt_all = []
                popt_s_all = []

                # Do the fitting
                for k in range(BOOTSTRAP_REPETITIONS):
                    filter = np.where(for_fitting[k] != -1)[0]
                    if len(filter) >= 2:
                        deviation = np.array(for_fitting[k])[filter] - np.array(shears)[filter]
                        popt, pcov = curve_fit(linear_function, np.array(shears)[filter], \
                                               deviation, sigma=np.array(uncertainties[k])[filter], absolute_sigma=True)

                        popt_all.append(popt)

                    if num_shears == 11:
                        filter = np.where(np.array(for_fitting[k][4:7:2]) != -1)[0]
                        if len(filter) >= 2:
                            deviation_s = np.array(for_fitting[k][4:7:2])[filter] - np.array(shears[4:7:2])[filter]
                            popt_s, pcov_s = curve_fit(linear_function, np.array(shears)[4:7:2][filter], \
                                                       deviation_s, sigma=np.array(uncertainties[k][4:7:2])[filter], absolute_sigma=True)

                            popt_s_all.append(popt_s)

                time_fit += timeit.default_timer() - start_fit
                error_fit_m_large = np.std(np.array(popt_all)[:,0])
                error_fit_c_large = np.std(np.array(popt_all)[:,1])
                if num_shears == 11:
                    error_fit_m_small = np.std(np.array(popt_s_all)[:,0])
                    error_fit_c_small = np.std(np.array(popt_s_all)[:,1])


            output_shear = []
            output_err = []

            # ------ FITTING FOR COMPARISON ----------------#
            for o in range(num_shears):
                if len(data_complete["meas_g1"][(data_complete["scene_index"] <= scene) &
                                                (data_complete["shear_index"] == o) & (
                                                        data_complete[bin_type] > lower_limit) & (
                                                        data_complete[bin_type] < upper_limit)]) != 0:

                    output_shear.append(np.average(data_complete["meas_g1"][(data_complete["scene_index"] <= scene) &
                                                                            (data_complete["shear_index"] == o) & (
                                                                                    data_complete[
                                                                                        bin_type] > lower_limit) & (
                                                                                    data_complete[
                                                                                        bin_type] < upper_limit)]))
                    output_err.append(np.std(data_complete["meas_g1"][(data_complete["scene_index"] <= scene) & (
                            data_complete["shear_index"] == o) & (
                                                                              data_complete[
                                                                                  bin_type] > lower_limit) & (
                                                                              data_complete[
                                                                                  bin_type] < upper_limit)])
                                      / np.sqrt(len(data_complete["meas_g1"][(data_complete["scene_index"] <= scene) & (
                            data_complete["shear_index"] == o) & (
                                                                                     data_complete[
                                                                                         bin_type] > lower_limit) & (
                                                                                     data_complete[
                                                                                         bin_type] < upper_limit)])))
                else:
                    output_shear.append(-1)
                    output_err.append(-1)

            filter = np.where(np.array(output_shear) != -1)[0]
            if len(filter) >= 2:
                deviation = np.array(output_shear)[filter] - np.array(shears)[filter]
                popt, pcov = curve_fit(linear_function, np.array(shears)[filter], \
                                       deviation, sigma=np.array(output_err)[filter], absolute_sigma=True)

                error = np.sqrt(np.diag(pcov))

                if reps == REPS - 1:
                    # Plot the fit
                    mm = 1 / 25.4
                    fig, ax = plt.subplots(figsize=(88 * mm, 88 * mm))

                    ax.errorbar(np.array(shears)[filter], deviation, np.array(output_err)[filter], fmt="+--",
                                markersize=5,
                                capsize=2, elinewidth=0.5)
                    ax.plot(np.linspace(-0.1, 0.1, 10), linear_function(np.linspace(-0.1, 0.1, 10), *popt), alpha=0.7)


                    ax.hlines(0.0, -0.1, 0.1, linestyle="dashed", alpha=0.8)

                    ax.set_xlabel("$g_1^t$")
                    ax.set_ylabel("$<g_1^{obs}>-g_1^t$")
                    fig.savefig(subfolder + f"/plots/{scene}_{mag}.png", dpi=200, bbox_inches="tight")
                    plt.close()
            else:
                popt = [-1, -1]
                error = [-1, -1]

            if num_shears == 11:
                filter = np.where(np.array(output_shear[4:7:2]) != -1)[0]
                if len(filter) >= 2:
                    deviation_s = np.array(output_shear[4:7:2])[filter] - np.array(shears[4:7:2])[filter]
                    popt_s, pcov_s = curve_fit(linear_function, np.array(shears)[4:7:2][filter], \
                                               deviation_s, sigma=np.array(output_err[4:7:2])[filter], absolute_sigma=True)

                    error_s = np.sqrt(np.diag(pcov_s))
                else:
                    popt_s = [-1, -1]
                    error_s = [-1, -1]

            if not bootstrap_fit:
                error_fit_m_large = error[0]
                error_fit_c_large = error[1]
                if num_shears == 11:
                    error_fit_m_small = error_s[0]
                    error_fit_c_small = error_s[1]

            c_bias = np.average(output_shear)
            c_bias_err = np.sqrt(np.sum(np.square(output_err))) / len(output_shear)


            if num_shears == 11:
                individual_biases = []
                individual_biases_err = []
                individual_biases_err_err = []
                for interval in range(10):
                    # -------- M-BIAS SMALL INTERVAL -----------------#
                    if np.sum(meas1_weights_small[interval][mag][:int((scene + 1) / division)]) != 0 and np.sum(
                            meas0_weights_small[interval][mag][:int((scene + 1) / division)]) != 0:
                        bias_data = fct.bootstrap_puyol(meas1_averages_small[interval][mag][:int((scene + 1) / division)],
                                                        meas0_averages_small[interval][mag][:int((scene + 1) / division)],
                                                        BOOTSTRAP_REPETITIONS, 0.02,
                                                        meas1_weights_small[interval][mag][:int((scene + 1) / division)],
                                                        meas0_weights_small[interval][mag][:int((scene + 1) / division)])

                        bias_small = (np.average(meas1_averages_small[interval][mag][:int((scene + 1) / division)],
                                                 weights=meas1_weights_small[interval][mag][:int((scene + 1) / division)]) - np.average(
                            meas0_averages_small[interval][mag][:int((scene + 1) / division)], weights=meas0_weights_small[interval][mag][:int(
                                (scene + 1) / division)])) / 0.02 - 1  # (meas1_stats[0]-meas0_stats[0])/0.04 - 1

                        bias_small_err = bias_data[1]
                        bias_small_err_err = np.std([fct.bootstrap_puyol(meas1_averages_small[interval][mag][:int((scene + 1) / division)],
                                                                         meas0_averages_small[interval][mag][:int((scene + 1) / division)],
                                                                         int(BOOTSTRAP_REPETITIONS/10), 0.02,
                                                                         meas1_weights_small[interval][mag][:int((scene + 1) / division)],
                                                                         meas0_weights_small[interval][mag][:int((scene + 1) / division)])[1]
                                                     for _ in range(10)]) / np.sqrt(10)

                    else:
                        bias_small = -1
                        bias_small_err = -1
                        bias_small_err_err = -1

                    individual_biases.append(bias_small)
                    individual_biases_err.append(bias_small_err)
                    individual_biases_err_err.append(bias_small_err_err)

            # -------------------- C-BIAS TREATMENTS -----------------------------#
            if np.sum(meas_weights[mag][:int((scene + 1) / division)]) != 0:
                c_bias = np.average(meas_averages[mag][:int((scene + 1) / division)],
                                    weights=meas_weights[mag][:int((scene + 1) / division)])
                c_bias_err = bootstrap(meas_averages[mag][:int((scene + 1) / division)],
                                       meas_weights[mag][:int((scene + 1) / division)], BOOTSTRAP_REPETITIONS)
                c_bias_err_err = np.std([bootstrap(meas_averages[mag][:int((scene + 1) / division)],
                                                   meas_weights[mag][:int((scene + 1) / division)], BOOTSTRAP_REPETITIONS)
                                         for _ in range(10)])
            else:
                c_bias = -1
                c_bias_err = -1
                c_bias_err_err = -1

            if num_shears == 11:
                individual_c_biases = []
                individual_c_biases_err = []
                individual_c_biases_err_err = []
                for interval in range(10):
                    if np.sum(meas_weights_small[interval][mag][:int((scene + 1) / division)]) != 0:
                        c_bias_s = np.average(meas_averages_small[interval][mag][:int((scene + 1) / division)],
                                              weights=meas_weights_small[interval][mag][:int((scene + 1) / division)])
                        c_bias_err_s = bootstrap(meas_averages_small[interval][mag][:int((scene + 1) / division)],
                                                 meas_weights_small[interval][mag][:int((scene + 1) / division)], BOOTSTRAP_REPETITIONS)
                        c_bias_err_err_s = np.std(
                            [bootstrap(meas_averages_small[interval][mag][:int((scene + 1) / division)],
                                       meas_weights_small[interval][mag][:int((scene + 1) / division)], BOOTSTRAP_REPETITIONS) for _ in
                             range(10)])
                    else:
                        c_bias_s = -1
                        c_bias_err_s = -1
                        c_bias_err_err_s = -1

                    individual_c_biases.append(c_bias_s)
                    individual_c_biases_err.append(c_bias_err_s)
                    individual_c_biases_err_err.append(c_bias_err_err_s)

            if np.sum(meas1_weights[mag][:int((scene + 1) / division)]) != 0 and np.sum(
                    meas0_weights[mag][:int((scene + 1) / division)]) != 0:

                if simulation.getboolean("same_but_shear"):
                    shear_diff = float(simulation["same_but_shear_diff"])
                else:
                    shear_diff = shears[1] - shears[0]

                bias_data = fct.bootstrap_puyol(meas1_averages[mag][:int((scene + 1) / division)],
                                                meas0_averages[mag][:int((scene + 1) / division)], BOOTSTRAP_REPETITIONS,
                                                shear_diff, meas1_weights[mag][:int((scene + 1) / division)],
                                                meas0_weights[mag][:int((scene + 1) / division)])

                bias = (np.average(meas1_averages[mag][:int((scene + 1) / division)], weights=meas1_weights[mag][:int((scene + 1) / division)])-
                        np.average(meas0_averages[mag][:int((scene + 1) / division)], weights=meas0_weights[mag][:int((scene + 1) / division)])) / shear_diff -1


                err = bias_data[1]

                err_err = np.std([fct.bootstrap_puyol(meas1_averages[mag][:int((scene + 1) / division)],
                                                      meas0_averages[mag][:int((scene + 1) / division)],
                                                      int(BOOTSTRAP_REPETITIONS / 10),
                                                      shear_diff,
                                                      meas1_weights[mag][:int((scene + 1) / division)],
                                                      meas0_weights[mag][:int((scene + 1) / division)])[1] for _ in
                                  range(10)]) / np.sqrt(10)
                # BIAS FROM RESPONSE MEAN (WEIGHTED)

                responses_update = [(meas1_averages[mag][:int((scene + 1) / division)][i] -
                                     meas0_averages[mag][:int((scene + 1) / division)][i]) / shear_diff - 1 for
                                    i in
                                    range(len(meas1_averages[mag][:int((scene + 1) / division)]))]

                results_values = (
                    np.average(responses_update), np.std(responses_update) / np.sqrt(len(responses_update)))  # No weights
            else:
                bias = -1
                err = -1
                err_err = -1
                results_values = (-1, -1)

            if num_shears != 11:
                bias_small = bias
                bias_small_err = err
                c_bias_s = c_bias
                c_bias_err_s = c_bias_err
                error_fit_m_small = error_fit_m_large
                error_fit_c_small = error_fit_c_large
                bias_small_err_err = err_err
                c_bias_err_err_s = c_bias_err_err
                popt_s = popt

            # WRITE RESULTS TO FILE
            with open(path + "output/rp_simulations/fits.txt", "a") as file:
                file.write(
                    "%s\t %d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %d\t %.4f\t %.7f\t %.7f\t %.7f\t "
                    "%.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.1f\t %d\t %.7f\t %.7f\t %.7f\t %.7f\n" %
                    ("pujol", complete_image_size, galaxy_number, scene + 1, bias, err, c_bias,
                     c_bias_err,
                     results_values[0], results_values[1], int(timeit.default_timer() - start),
                     (scene + 1) * galaxy_number * num_shears * (1 + noise_plus_meas) +
                     (scene + 1) * num_shears * scene_creation, bias_small, bias_small_err, c_bias_s, c_bias_err_s,
                     popt[0], error_fit_m_large, popt[1], error_fit_c_large, popt_s[0], error_fit_m_small, popt_s[1], error_fit_c_small,
                     magnitudes_list[counter], np.sum(meas0_weights[mag][:int((scene + 1) / division)]), err_err,
                     c_bias_err_err, bias_small_err_err,
                     c_bias_err_err_s))

            blended_fraction = len(data_complete[
                                       (data_complete["scene_index"] == scene) & (
                                               data_complete[bin_type] < upper_limit) & (
                                               data_complete[bin_type] >= lower_limit) & (
                                               data_complete["se_flag"] > 0)])

            blended_kron = len(data_complete[
                                       (data_complete["scene_index"] == scene) & (
                                               data_complete[bin_type] < upper_limit) & (
                                               data_complete[bin_type] >= lower_limit) & (
                                               data_complete["kron_blend"] > 0)])
            complete_length = len(data_complete[
                                      (data_complete["scene_index"] == scene) & (
                                              data_complete[bin_type] < upper_limit) & (
                                              data_complete[bin_type] >= lower_limit)])

            if complete_length == 0:
                complete_length = 1

            blending_fraction = 100 * blended_fraction / complete_length
            blending_fraction_kron = 100 * blended_kron / complete_length

            columns.append([complete_image_size, galaxy_number, scene + 1, bias, err, c_bias,
                      c_bias_err,
                      results_values[0], results_values[1], int(timeit.default_timer() - start),
                      (scene + 1) * galaxy_number * num_shears * (1 + noise_plus_meas) +
                      (scene + 1) * num_shears * scene_creation, bias_small, bias_small_err, c_bias_s, c_bias_err_s,
                      popt[0], error_fit_m_large, popt[1], error_fit_c_large, popt_s[0], error_fit_m_small, popt_s[1], error_fit_c_small,
                      magnitudes_list[counter], np.sum(meas0_weights[mag][:int((scene + 1) / division)]), err_err,
                      c_bias_err_err, bias_small_err_err,
                      c_bias_err_err_s, blending_fraction, blending_fraction_kron])


            counter += 1


            if num_shears == 11 and reps==REPS-1:
                with open(path + "output/rp_simulations/pujol_individual_biases.txt", "a") as file:
                    file.write("%d\t %d\t %d\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\t %.7f\n" %
                               (complete_image_size, galaxy_number, scene+1, individual_biases[0], individual_biases_err[0],
                                individual_biases[1], individual_biases_err[1],
                                individual_biases[2], individual_biases_err[2],
                                individual_biases[3], individual_biases_err[3],
                                individual_biases[4], individual_biases_err[4],
                        individual_biases[5], individual_biases_err[5],
                                individual_biases[6], individual_biases_err[6],
                            individual_biases[7], individual_biases_err[7],
                                individual_biases[8], individual_biases_err[8],
                                individual_biases[9], individual_biases_err[9]))


columns = np.array(columns, dtype=float)
pujol_results = Table([columns[:, i] for i in range(31)], names=(
    'sim_size', 'galaxy_num', 'scene_index', 'm_bias_large', 'm_bias_large_err', 'c_bias_large', 'c_bias_large_err',
    'm_bias_response', 'm_bias_response_err', 'total_runtime', 'theo_runtime', 'm_bias_small', 'm_bias_small_err',
    'c_bias_small', 'c_bias_small_err', 'm_bias_large_fit', 'm_bias_large_fit_err', 'c_bias_large_fit',
    'c_bias_large_fit_err',
    'm_bias_small_fit', 'm_bias_small_fit_err', 'c_bias_small_fit', 'c_bias_small_fit_err', bin_type, 'weight',
    'm_bias_large_err_err',
    'c_bias_large_err_err', 'm_bias_small_err_err', 'c_bias_small_err_err', 'blending_fraction', 'blending_fraction_kron'))

ascii.write(pujol_results, subfolder + 'analysis.dat',
            overwrite=True)

print(f"analysis runtime: {timeit.default_timer() - start1}")
print(f"Sorting time: {sorting_time}")
print(f"Rest: {timeit.default_timer() - start_rest}")
print(f"Davon Bootstrap fit: {time_fit}")
