# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:21:48 2021

This code reads in galaxies from the GEMS survey and takes their magnitude, half light radii and Sersic indices
to build realistic galaxy images. Those images are then additionally sheared with intrinsic ellipticity and
extrinsic shear. These stamps of individual galaxies are then placed on a larger image on random positions.

The PSF is built as a stack of monochromatic PSF's from 550 nm to 900 nm weighted by the spectrum of Vega.

Adjustable parameters are
    - pixel scale of the telescopes CCD
    - exposure time
    - Gain of the amplifier
    - Read-Out-Noise
    - Sky level
    - Zero-point magnitude
    - step size to sample the PSF
    - telescope diameter in m
    - image size in pixel in both x- and y-direction
    - how many galaxies to generate in one scene
    - RMS value for the intrinsic ellipticity and a maximum ellipticity

This script realizes the method suggested by Pujol et al. (2018) and builds additional scenes with almost the same
noise (apart from the Poisson part coming from the galaxies) but slightly different shear. Biases are then estimated
from the responses.

Input half-light-radii are given in pixel values of the GEMS survey corresponding to 0.03"

This code uses multiprocessing to distribute the workload to several cores. One process works on one galaxy and
its respective cancellation methods.

Command line code is
python3 pujol_rp.py <scene size in pixel> <number of galaxies> <n_scene per shear> <n_shear>
@author: Henning
"""

import random

from astropy.table import Table
from astropy.io import fits
from astropy.table import QTable
from astropy.io import ascii
import numpy as np
import sys
import logging
import os
import galsim
import timeit
import configparser
import functions as fct
import math
import ray
import psutil
from scipy.optimize import curve_fit
import scipy
import datetime

# ---------------------------- INITIALIZATION --------------------------------------------------------------------------
path = sys.argv[5] + "/"

# Create temporary folder for FITS files
index_fits = 0
while os.path.isdir(path + f"/output/FITS{index_fits}"):
    index_fits += 1

os.mkdir(path + f"/output/FITS{index_fits}")
os.mkdir(path + f"/output/source_extractor/{index_fits}")

# Time the program
start = timeit.default_timer()


def linear_function(x, a, b):
    return a * x + b


def do_kdtree(combined_x_y_arrays, points, k=1):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)

    return mytree.query(points, k=k)



def bootstrap(array, weights, n):
    indices = np.random.choice(np.arange(len(array)), size=(n, len(array)))
    bootstrap = np.take(array, indices, axis=0).reshape(n, -1)
    weights = np.take(weights, indices, axis=0).reshape(n, -1)

    filter = (np.sum(weights, axis=1) != 0)

    bootstrap = bootstrap[filter]
    weights = weights[filter]

    return np.std(np.average(bootstrap, axis=1, weights=weights))


# ----------------------------- READ INPUT CATALOG----------------------------------------------------------------------

''' Just to filter the table '''
# Read the table
t = Table.read(path + 'input/gems_20090807.fits')

config = configparser.ConfigParser()
config.read('config_rp.ini')

if not config['SIMULATION'].getboolean("same_but_shear"):
    # Additional randomization
    config.set('SIMULATION', 'random_seed', str(np.random.randint(1e12)))

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

MAX_NEIGHBORS = int(config['MATCHING']["max_neighbors"])
MAX_DIST = int(config['MATCHING']["max_dist"])

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])

mag_bins = int(simulation["bins_mag"])

min_mag = float(simulation["min_mag"])
max_mag = float(simulation["max_mag"])

# Introduce a mask to filter the entries
mask = (t["GEMS_FLAG"] == 4) & (np.abs(t["ST_MAG_BEST"] - t["ST_MAG_GALFIT"]) < 0.5) & (min_mag < t["ST_MAG_GALFIT"]) & \
       (t["ST_MAG_GALFIT"] < max_mag) & (0.3 < t["ST_N_GALFIT"]) & (t["ST_N_GALFIT"] < 6.0) & (
               3.0 < t["ST_RE_GALFIT"]) & \
       (t["ST_RE_GALFIT"] < 40.0)

galaxies = t[mask]

''' Build the images '''

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("real_galaxies")

logger.info("%d galaxies have been drawn from the fits file", len(galaxies["GEMS_FLAG"]))

# ------------------------------- READ IN THE CONFIG FILE --------------------------------------------------------------


num_cpu = int(simulation['num_cores'])  # psutil.cpu_count()

# Initialize Ray, which is used for the multiprocessing
ray.init(num_cpus=num_cpu, log_to_driver=False)

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

galaxy_number = int(sys.argv[2])

num_shears = int(sys.argv[4])
total_scenes_per_shear = int(sys.argv[3])

complete_image_size = int(sys.argv[1])

# Calculate sky level
sky_level = pixel_scale ** 2 * exp_time / gain * 10 ** (-0.4 * (sky - zp))

# For the PSF import the vega spectrum
data = np.genfromtxt(path + "input/vega_spectrumdat.sec")

# ------------------------------ CREATE THE PSF AND SET UP THE SHARED MEMORY -------------------------------------------

# Define the PSF profile
if psf_config["psf"] == "EUCLID":
    # Weight the contribution of each monochromatic PSF with the Vega spectrum
    psf = fct.generate_psf(lam, step_psf, 900, data, tel_diam)
elif psf_config["psf"] == "AIRY":
    psf = galsim.Airy(lam * 1.e-9 / tel_diam * 206265)  # galsim.Gaussian(fwhm=0.15, flux=1.)
elif psf_config["psf"] == "GAUSS":
    psf = galsim.Gaussian(fwhm=0.15)

# Put PSF, config and ARGV in ray's shared memory
psf_ref = ray.put(psf)
config_ref = ray.put(config)
argv_ref = ray.put(sys.argv)
# ----------- Create the catalog ---------------------------------------------------------------------------------------
responses = []
weights = []
c_biases = []
all_meas = [[], []]
input_averages = []
# matched_indices = [] # For the catalog matching

# Define the shear arrays
if num_shears == 2:
    shears = [-0.02, 0.02]
else:
    shears = [-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)]

input_positions = []
input_magnitudes = []

columns = []
# ------- DETERMINATION OF SKY BACKGROUND FOR THEORETICAL S/N ------------------------------------------------------#
rng = galsim.UniformDeviate()
# SKY NOISE
sky_image = galsim.Image(
    np.reshape(np.zeros((stamp_xsize - 1) * (stamp_ysize - 1)),
               (stamp_xsize - 1, stamp_ysize - 1)))

sky_image.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=0, sky_level=sky_level))

# POISSON GALAXY
noise = galsim.CCDNoise(rng, gain=gain, read_noise=0., sky_level=0.0)

sky_image.addNoise(noise)

# GAUSSIAN NOISE
sky_image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise / gain))

pixels = sky_image.array.copy()

edge = list(pixels[0]) + list([i[-1] for i in pixels[1:-1]]) + list(reversed(pixels[-1])) + \
       list(reversed([i[0] for i in pixels[1:-1]]))

sigma_sky = 1.4826 * np.median(np.abs(edge - np.median(edge)))

if simulation.getboolean("same_but_shear"):
    random.seed(123)
    np.random.seed(123)


positions = [[] for _ in range(total_scenes_per_shear)]
gal_list = [[] for _ in range(total_scenes_per_shear)]

for total_scene_count in range(total_scenes_per_shear):
    # ------------------------------------ CREATE THE GALAXY LIST RANDOMLY PER SCENE ----------------------------------
    count = 0

    positions[total_scene_count] = int(sys.argv[1]) * np.random.random_sample((galaxy_number, 2))
    input_positions.append(positions[total_scene_count])
    input_shears = []

    gal_below24_5 = 0
    if total_scene_count % 2 == 0:
        magnitudes = []
        for i in range(galaxy_number):
            index = random.randint(0, len(galaxies["GEMS_FLAG"]) - 1)

            ellips = fct.generate_ellipticity(ellip_rms, ellip_max, seed=total_scene_count*galaxy_number+ i +1 if simulation.getboolean("same_but_shear") else 0)
            betas = random.random() * 2 * math.pi * galsim.radians
            magnitudes.append(galaxies["ST_MAG_GALFIT"][index])
            gal_flux = exp_time / gain * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp))

            q = galaxies["ST_B_IMAGE"][index] / galaxies["ST_A_IMAGE"][index]
            gal = galsim.Sersic(galaxies["ST_N_GALFIT"][index],
                                half_light_radius=0.03 * galaxies["ST_RE_GALFIT"][index] * np.sqrt(q), flux=gal_flux)

            if galaxies["ST_MAG_GALFIT"][index] <= 24.5:
                gal_below24_5 += 1
            input_shears.append(galsim.Shear(g=ellips, beta=betas).g1)
            input_shears.append(galsim.Shear(g=ellips, beta=betas + math.pi / 2 * galsim.radians).g1)

            gal_1 = gal.shear(g=ellips, beta=betas)

            gal_2 = gal.shear(g=ellips, beta=betas + math.pi / 2 * galsim.radians)

            gal_list[total_scene_count].append(gal_1)

            gal_list[total_scene_count+1].append(gal_2)

            theo_sn = exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp)) / \
                      np.sqrt((exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp)) +
                               sky_level * gain * math.pi * (3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][index]) ** 2 +
                               (read_noise ** 2 + (gain / 2) ** 2) * math.pi * (
                                       3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][index]) ** 2))


            columns.append(
                [total_scene_count, positions[total_scene_count][i, 0], positions[total_scene_count][i, 1], galaxies["ST_MAG_GALFIT"][index], ellips,
                 betas / galsim.radians,
                 galaxies["ST_N_GALFIT"][index], 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][index], theo_sn])

            columns.append(
                [total_scene_count + 1, positions[total_scene_count][i, 0], positions[total_scene_count][i, 1], galaxies["ST_MAG_GALFIT"][index], ellips,
                 (betas + math.pi / 2 * galsim.radians) / galsim.radians,
                 galaxies["ST_N_GALFIT"][index], 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][index], theo_sn])


        print(gal_below24_5)
        print(np.average(input_shears))
    else:
        for i in range(galaxy_number):
            columns[-1 - 2 * (galaxy_number -1 -i)][1] = positions[total_scene_count][i, 0]
            columns[-1 - 2 * (galaxy_number -1 -i)][2] = positions[total_scene_count][i, 1]


    input_magnitudes.append(magnitudes)

# ------------------------------ DISTRIBUTE WORK TO RAY -----------------------------------------------------------
ids = []

for scene in range(total_scenes_per_shear):
    for m in range(num_shears):
        ids.append(fct.one_scene_pujol.remote(m, scene, gal_list[scene], positions[scene], argv_ref,
                                              config_ref, path, psf_ref, num_shears, index_fits))


# --------------------------------- MEASURE CATALOGS ------------------------------------------------------------------

order = []
scene_count = []
matching = []
magnitudes = []


columns = np.array(columns, dtype=float)
input_catalog = Table([columns[:, i] for i in range(9)],
                      names=('scene_index', 'position_x', 'position_y', 'mag', 'e', 'beta', 'n', 'hlr', 's/n'))

columns = []
while ids:
    ready, not_ready = ray.wait(ids)

    results = ray.get(ready)
    # if simulation.getboolean('matching'):
    kd_results = np.array(do_kdtree(input_positions[results[0][3]], results[0][1], k=MAX_NEIGHBORS))
    nearest_positional_neighbors = np.where(kd_results[0] <= MAX_DIST, kd_results[1], -1)

    # Filter out galaxies with no neighbours within MAX_DIST
    filter = np.sum(nearest_positional_neighbors, axis=1) != -MAX_NEIGHBORS
    nearest_positional_neighbors = nearest_positional_neighbors[filter]

    nearest_positional_neighbors = nearest_positional_neighbors.astype('int32')

    # If just no further neighbours are found within MAX_DIST, replace their entries with the nearest neighbour
    nearest_positional_neighbors = np.where(nearest_positional_neighbors == -1, np.repeat(nearest_positional_neighbors[:, 0], MAX_NEIGHBORS).reshape(nearest_positional_neighbors.shape), nearest_positional_neighbors)

    # Catch the case where no neighbour is found within the given distance (noise peaks?)
    if len(nearest_positional_neighbors) != len(results[0][0]):
        print(f"No neighbour within {MAX_DIST} px found for {len(results[0][0]) - len(nearest_positional_neighbors)} galaxies in scene {results[0][3]} at shear {results[0][2]}!")

    magnitudes_npn = np.array(input_magnitudes[results[0][3]])[nearest_positional_neighbors]

    for m in range(len(np.array(results[0][0])[filter])):
        se_flag_binary = '{:08b}'.format(int(np.array(results[0][6])[filter][m]))
        min_deviation = np.argmin(np.abs(np.subtract(magnitudes_npn[m], np.array(results[0][4])[filter][m])))
        gems_magnitude_optimized = np.array(magnitudes_npn[m])[min_deviation]
        matching_index = np.array(nearest_positional_neighbors[m])[min_deviation]
        columns.append([results[0][3], results[0][2], np.array(results[0][0])[filter][m], np.array(results[0][1])[filter][m][0], np.array(results[0][1])[filter][m][1],
                        np.array(results[0][4])[filter][m], magnitudes_npn[m][0], gems_magnitude_optimized, np.array(results[0][5])[filter][m],
                        nearest_positional_neighbors[m][0], matching_index, 0 if len(np.unique(nearest_positional_neighbors[m])) == 1
                        else 1 if (np.abs(magnitudes_npn[m][0]-magnitudes_npn[m][1]) > 2) and (len(np.unique(nearest_positional_neighbors[m])) == 2)
                        else 2, int(se_flag_binary[-2]) + int(se_flag_binary[-1]), np.array(results[0][7])[filter][m],
                        np.array(results[0][8])[filter][m], np.array(results[0][9])[filter][m], np.array(results[0][10])[filter][m]])


    ids = not_ready

columns = np.array(columns, dtype=float)
shear_results = Table([columns[:, i] for i in range(17)],
                      names=('scene_index', 'shear_index', 'meas_g1', 'position_x', 'position_y', 'mag_auto',
                             'mag_gems', 'mag_gems_optimized', 'S/N', 'matching_index', 'matching_index_optimized', 'blending_flag', 'se_flag',
                             'kron_radius', 'a_image', 'b_image', 'elongation'))
now = datetime.datetime.now()

current_time = now.strftime("%H-%M-%S")
date_object = datetime.date.today()

os.system('mkdir ' + path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}')

ascii.write(shear_results,
            path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}/shear_catalog.dat',
            overwrite=True)
ascii.write(input_catalog,
            path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}/input_catalog.dat',
            overwrite=True)
# print(all_meas[0])

ray.shutdown()
os.system('cp config_rp.ini ' + path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}/')
# DELETE ALL CATALOG AND FITS FILES TO SAVE MEMORY
os.chdir(path + "output")
os.system(f"rm -r source_extractor/{index_fits}")
if simulation.getboolean("output"):
    if not os.path.isdir(path + "output/rp_simulations/" + f'run_pujol_{date_object}_{current_time}/FITS_org'):
        os.mkdir(path + "output/rp_simulations/" + f'run_pujol_{date_object}_{current_time}/FITS_org')

    os.system('mv ' + path + f'output/FITS{index_fits}/*.fits' + ' ' + path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}/FITS_org/')

os.system(f"rm -r FITS{index_fits}")


print(f"{timeit.default_timer() - start} seconds")
