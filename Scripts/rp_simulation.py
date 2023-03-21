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

For each scene three additional scenes are generated realizing shape- and pixel noise cancellation

Input half-light-radii are given in pixel values of the GEMS survey corresponding to 0.03"

This code uses multiprocessing to distribute the workload to several cores. One process works on one galaxy and
its respective cancellation methods.

Command line code is
python3 rp_simulation.py <scene size in pixel> <number of galaxies> <n_scene per shear>
@author: Henning
"""

import random

from astropy.table import Table
from astropy.table import QTable
from astropy.io import ascii, fits
import numpy as np
import sys
import os
import logging
import galsim
import timeit
import configparser
import functions as fct
import math
import ray
import psutil
from scipy.optimize import curve_fit
import scipy
import matplotlib.pyplot as plt
import datetime


def do_kdtree(combined_x_y_arrays, points, k=1):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    # dist, indexes = mytree.query(points, k)
    return mytree.query(points, k=k)


# ---------------------------- INITIALIZATION --------------------------------------------------------------------------
print(psutil.cpu_count())

# Define local paths
path = sys.argv[4] + "/"

# Create temporary folder for FITS files
index_fits = 0
while os.path.isdir(path + f"/output/FITS{index_fits}"):
    index_fits += 1

os.mkdir(path + f"/output/FITS{index_fits}")
os.mkdir(path + f"/output/source_extractor/{index_fits}")
# Time the program
start = timeit.default_timer()

# ----------------------------- READ INPUT CATALOG----------------------------------------------------------------------

''' Just to filter the table '''
# Read the table
t = Table.read(path + 'input/gems_20090807.fits')

config = configparser.ConfigParser()
config.read('config_rp.ini')

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

shear_bins = int(simulation['shear_bins'])

shear_min = -float(sys.argv[5])
shear_max = float(sys.argv[5])

galaxy_number = int(sys.argv[2])

complete_image_size = int(sys.argv[1])

total_scenes_per_shear = int(sys.argv[3])

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

# psf = galsim.Gaussian(sigma=0.1)
logger.debug('Made PSF profile')

# Filter the PSF with a top hat at pixel scale
filter_function = galsim.Pixel(scale=pixel_scale)

psf_1 = galsim.Convolve([psf, filter_function])

# Draw the convolution on a finer pixel grid for KSB
image_sampled_psf = psf_1.drawImage(nx=ssamp_grid * stamp_xsize, ny=ssamp_grid * stamp_ysize,
                                    scale=1.0 / ssamp_grid * pixel_scale, method='no_pixel')
# image_sampled_psf = psf_1.drawImage(gband)
file_name_epsf = os.path.join(path + 'output', 'real_galaxies_psf_1.fits')
galsim.fits.write(image_sampled_psf, file_name_epsf)

psf_ref = ray.put(psf)
config_ref = ray.put(config)
argv_ref = ray.put(sys.argv)
# ----------- Create the catalog ---------------------------------------------------------------------------------------
none_measures = []
shape_measures = []
none_pixel_measures = []
shape_pixel_measures = []
input_positions = []
input_positions_2 = []

input_magnitudes = []

time = 0


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

columns = []

for scene in range(total_scenes_per_shear):
    print(scene)
    start_scene = timeit.default_timer()
    failure_counter = 0
    for m in range(shear_bins):
        # --------------------------------- CREATE GALAXY LIST FOR EACH RUN -------------------------------------------
        start_input_building = timeit.default_timer()
        count = 0
        gal_list = []
        gal_list2 = []
        magnitudes = []

        positions = int(sys.argv[1]) * np.random.random_sample((galaxy_number, 2))
        if sys.argv[6] == "RANDOM_POS" or sys.argv[6] == "RANDOM_GAL":
            positions_2 = int(sys.argv[1]) * np.random.random_sample((galaxy_number, 2))
        else:
            positions_2 = positions
        # print(positions)
        input_positions.append(positions)
        input_positions_2.append(positions_2)
        for i in range(galaxy_number):
            ellips = fct.generate_ellipticity(ellip_rms, ellip_max)
            betas = random.random() * 2 * math.pi * galsim.radians

            index = random.randint(0, len(galaxies["GEMS_FLAG"]) - 1)
            magnitudes.append(galaxies["ST_MAG_GALFIT"][index])

            gal_flux = exp_time / gain * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp))

            q = (1 - ellips) / (1 + ellips)

            # Correct for ellipticty
            gal = galsim.Sersic(galaxies["ST_N_GALFIT"][index],
                                half_light_radius=0.03 * galaxies["ST_RE_GALFIT"][index] * np.sqrt(q), flux=gal_flux)

            gal = gal.shear(g=ellips, beta=betas)

            gal_list.append(gal)

            theo_sn = exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp)) / \
                      np.sqrt((exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp)) +
                               sky_level * gain * math.pi * (3 * 0.3 * galaxies["ST_RE_GALFIT"][index]) ** 2 +
                               (read_noise ** 2 + (gain / 2) ** 2) * math.pi * (
                                           3 * 0.3 * galaxies["ST_RE_GALFIT"][index]) ** 2))

            if sys.argv[6] == "RANDOM_GAL":
                index2 = random.randint(0, len(galaxies["GEMS_FLAG"]) - 1)
                magnitudes.append(galaxies["ST_MAG_GALFIT"][index2])

                gal_flux = exp_time / gain * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index2] - zp))

                # Correct for ellipticity
                gal = galsim.Sersic(galaxies["ST_N_GALFIT"][index2],
                                    half_light_radius=0.03 * galaxies["ST_RE_GALFIT"][index2] * np.sqrt(q), flux=gal_flux)

                gal = gal.shear(g=ellips, beta=betas)

                gal_list2.append(gal)

                theo_sn2 = exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index2] - zp)) / \
                          np.sqrt((exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index2] - zp)) +
                                   sky_level * gain * math.pi * (3 * 0.3 * galaxies["ST_RE_GALFIT"][index2]) ** 2 +
                                   (read_noise ** 2 + (gain / 2) ** 2) * math.pi * (
                                           3 * 0.3 * galaxies["ST_RE_GALFIT"][index2]) ** 2))
            else:
                magnitudes.append(galaxies["ST_MAG_GALFIT"][index])
                index2 = index
                theo_sn2 = theo_sn


            for k in range(4):

                if k % 2 != 0:
                    if sys.argv[6] == "True":
                        columns.append([scene, m, k, positions[i][1], complete_image_size - positions[i][0],
                                        galaxies["ST_MAG_GALFIT"][index2], ellips,
                                        (betas + math.pi / 2 * galsim.radians) / galsim.radians,
                                        galaxies["ST_N_GALFIT"][index2], galaxies["ST_RE_GALFIT"][index2], theo_sn2])
                    else:
                        columns.append([scene, m, k, positions_2[i][0], positions_2[i][1],
                                        galaxies["ST_MAG_GALFIT"][index2], ellips,
                                        (betas + math.pi / 2 * galsim.radians) / galsim.radians,
                                        galaxies["ST_N_GALFIT"][index2], galaxies["ST_RE_GALFIT"][index2], theo_sn2])
                else:
                    columns.append([scene, m, k, positions[i][0], positions[i][1],
                                    galaxies["ST_MAG_GALFIT"][index], ellips, betas / galsim.radians,
                                    galaxies["ST_N_GALFIT"][index], galaxies["ST_RE_GALFIT"][index], theo_sn])


        if sys.argv[6] != "RANDOM_GAL":
            gal_list2 = gal_list


        input_magnitudes.append(magnitudes)
        if m == 0:
            print(timeit.default_timer() - start_input_building)
        # ----------------------------- DISTRIBUTE WORK TO RAY --------------------------------------------------------
        start_ray = timeit.default_timer()

        # Calculate how many images shall be calculated within one worker
        per_process = int(galaxy_number / num_cpu)

        rest = galaxy_number - num_cpu * per_process
        start_conv = timeit.default_timer()
        # Multiprocessing via Ray
        futures = [fct.multiple_catalog.remote(per_process, m,
                                               gal_list[k * per_process:(k + 1) * per_process], gal_list2[k * per_process:(k + 1) * per_process],
                                               positions[k * per_process:(k + 1) * per_process], positions_2[k * per_process:(k + 1) * per_process], psf_ref,
                                               config_ref, argv_ref)
                   for k in range(num_cpu)]
        if rest != 0:
            futures.append(fct.multiple_catalog.remote(rest, m,
                                                       gal_list[galaxy_number - rest:galaxy_number],
                                                       gal_list2[galaxy_number - rest:galaxy_number],
                                                       positions[galaxy_number - rest:galaxy_number],
                                                       positions_2[galaxy_number - rest:galaxy_number],
                                                       psf_ref, config_ref,
                                                       argv_ref))

        stamp_none = []
        stamp_shape = []
        stamp_none_pixel = []
        stamp_shape_pixel = []

        for i in range(galaxy_number):
            ready, not_ready = ray.wait(futures)

            for x in ray.get(ready)[0]:
                if x != -1:
                    stamp_none.append(x[0])
                    stamp_shape.append(x[1])
                    stamp_none_pixel.append(x[2])
                    stamp_shape_pixel.append(x[3])
                else:
                    failure_counter += 1

            futures = not_ready
            if not futures:
                break
        if m == 0:
            print(timeit.default_timer() - start_ray)
        time += timeit.default_timer() - start_conv
        start_catalog_building = timeit.default_timer()
        rng = galsim.UniformDeviate()

        seed1 = int(rng() * 1e6)
        seed2 = int(rng() * 1e6)

        ids = [fct.create_catalog_lf.remote(stamp_none, 0, seed1,
                                            m, scene, argv_ref, config_ref, path, psf_ref, index_fits),
               fct.create_catalog_lf.remote(stamp_shape, 1, seed2,
                                            m, scene, argv_ref, config_ref, path, psf_ref, index_fits),
               fct.create_catalog_lf.remote(stamp_none_pixel, 2, seed1,
                                            m, scene, argv_ref, config_ref, path, psf_ref, index_fits),
               fct.create_catalog_lf.remote(stamp_shape_pixel, 3, seed2,
                                            m, scene, argv_ref, config_ref, path, psf_ref, index_fits)
               ]

        while ids:
            ready, not_ready = ray.wait(ids)

            ids = not_ready

        if m == 0:
            print(timeit.default_timer() - start_catalog_building)
        # print("\n")
    print(timeit.default_timer() - start_scene)
    print("\n")
    print(f"Failed FFT's: {failure_counter}")
# print(input_distribution)
columns = np.array(columns, dtype=float)

input_catalog = Table([columns[:, i] for i in range(11)], names=(
'scene_index', 'shear_index', 'cancel_index', 'position_x', 'position_y', 'mag', 'e', 'beta', 'n', 'hlr', 's/n'))
# -------------------------- MEASURE CATALOGS -------------------------------------------------------------------------
ids = []
for scene in range(total_scenes_per_shear):
    for m in range(shear_bins):
        ids.append(fct.measure_catalog_lf.remote(m, scene, argv_ref, config_ref, path, psf_ref, index_fits))

names = ["none", "shape", "none_pixel", "shape_pixel"]
magnitudes = [[] for _ in range(4)]
magnitudes_list = [min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)]
measure_arrays = [none_measures, shape_measures, none_pixel_measures, shape_pixel_measures]
neighbour_dist = [[], []]

columns = []
while ids:
    ready, not_ready = ray.wait(ids)
    for x in ray.get(ready):

        for i in range(4):
            init_positions = input_positions[x[i][5] * shear_bins + x[i][4]]
            if i % 2 != 0:
                init_positions = input_positions_2[x[i][5] * shear_bins + x[i][4]]
                if sys.argv[6] == "True":
                    init_positions = np.array(init_positions)
                    init_positions[:, [0, 1]] = init_positions[:, [1, 0]]
                    init_positions[:, 1] = complete_image_size - init_positions[:, 1]

            kd_results = np.array(do_kdtree(init_positions, x[i][6], k=MAX_NEIGHBORS))
            nearest_positional_neighbors = np.where(kd_results[0] <= MAX_DIST, kd_results[1], -1)

            # Filter out galaxies with no neighbours within MAX_DIST
            filter = np.sum(nearest_positional_neighbors, axis=1) != -MAX_NEIGHBORS
            nearest_positional_neighbors = nearest_positional_neighbors[filter]

            nearest_positional_neighbors = nearest_positional_neighbors.astype('int32')

            # If just no further neighbours are found within MAX_DIST, replace their entries with the nearest neighbour
            nearest_positional_neighbors = np.where(nearest_positional_neighbors == -1,
                                                    np.repeat(nearest_positional_neighbors[:, 0],
                                                              MAX_NEIGHBORS).reshape(
                                                        nearest_positional_neighbors.shape),
                                                    nearest_positional_neighbors)

            # Catch the case where no neighbour is found within the given distance (noise peaks?)
            if len(nearest_positional_neighbors) != len(x[i][6]):
                print(
                    f"No neighbour within {MAX_DIST} px found for {len(x[i][6]) - len(nearest_positional_neighbors)} galaxies in scene {x[i][5]} at shear {x[i][4]}!")

            if i % 2 != 0:
                magnitudes_npn = np.array(input_magnitudes[x[i][5] * shear_bins + x[i][4]][1::2])[nearest_positional_neighbors]
            else:
                magnitudes_npn = np.array(input_magnitudes[x[i][5] * shear_bins + x[i][4]][::2])[
                    nearest_positional_neighbors]



            for gal in range(len(np.array(x[i][6])[filter])):
                min_deviation = np.argmin(np.abs(np.subtract(magnitudes_npn[gal], np.array(x[i][8])[filter][gal])))
                gems_magnitude_optimized = np.array(magnitudes_npn[gal])[min_deviation]
                matching_index = np.array(nearest_positional_neighbors[gal])[min_deviation]
                columns.append(
                    [x[i][5], x[i][4], i, x[i][0], np.array(x[i][6])[filter][gal][0], np.array(x[i][6])[filter][gal][1], np.array(x[i][7])[filter][gal], np.array(x[i][8])[filter][gal],
                     magnitudes_npn[gal][0], gems_magnitude_optimized, np.array(x[i][9])[filter][gal],
                     nearest_positional_neighbors[gal][0], matching_index, 0 if len(np.unique(nearest_positional_neighbors[gal])) == 1
                     else 1 if (np.abs(magnitudes_npn[gal][0]-magnitudes_npn[gal][1]) > 2) and (len(np.unique(nearest_positional_neighbors[gal])) == 2)
                    else 2])


    ids = not_ready

columns = np.array(columns, dtype=float)
shear_results = Table([columns[:, i] for i in range(14)], names=(
'scene_index', 'shear_index', 'cancel_index', 'input_g1', 'position_x', 'position_y', 'meas_g1', 'mag_auto',
'mag_gems', 'mag_gems_optimized', 'S/N', 'matching_index', 'matching_index_optimized', 'blending_flag'))

now = datetime.datetime.now()

current_time = now.strftime("%H-%M-%S")
date_object = datetime.date.today()

os.system('mkdir ' + path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[6]}')

ascii.write(input_catalog, path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[6]}/input_catalog.dat',
            overwrite=True)
ascii.write(shear_results, path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[6]}/shear_catalog.dat',
            overwrite=True)


# DELETE CATALOGUES AND FITS FILES TO SAVE MEMORY
os.chdir(path + "output")
os.system(f"rm -r source_extractor/{index_fits}")

if simulation.getboolean("output"):
    if not os.path.isdir(path + "output/rp_simulations/" + f'run_lf_{date_object}_{current_time}_{sys.argv[6]}/FITS_org'):
        os.mkdir(path + "output/rp_simulations/" + f"run_lf_{date_object}_{current_time}_{sys.argv[6]}/FITS_org")

    os.system('mv ' + path + f'output/FITS{index_fits}/*.fits' + ' ' + path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[6]}/FITS_org/')

os.system(f"rm -r FITS{index_fits}")

ray.shutdown()
