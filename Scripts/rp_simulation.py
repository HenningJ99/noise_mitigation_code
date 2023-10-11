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
path = sys.argv[3] + "/"

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

''' Build the images '''

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("real_galaxies")

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
ra_min_org = float(image["ra_min"])
dec_min_org = float(image["dec_min"])

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

complete_image_size = int(sys.argv[1])

total_scenes_per_shear = int(sys.argv[2])

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

psf_image = psf.drawImage(nx=stamp_xsize, ny=stamp_ysize, scale=pixel_scale)
galsim.fits.write(psf_image, "psf_cosmos.fits")

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

# ----------- Load the flagship catalog --------------------------------------------------------------------------------
hdul = fits.open("../Simulations/input/flagship.fits")
flagship = hdul[1].data
flagship["bulge_r50"] *= 1.0
flagship["disk_r50"] *= 1.0

#flagship["bulge_nsersic"] = np.where(flagship["bulge_nsersic"] * 0.9 < 0.3, 0.3, flagship["bulge_nsersic"] * 0.9)
#flagship["disk_nsersic"] = np.where(flagship["disk_nsersic"] * 0.9 < 0.3, 0.3, flagship["disk_nsersic"] * 0.9)

flagship["bulge_axis_ratio"] *= 1.0
flagship["disk_axis_ratio"] *= 1.0

patches = shear_bins * total_scenes_per_shear

CATALOG_LIMITS = np.min(flagship["dec_gal"]), np.max(flagship["dec_gal"]), np.min(flagship["ra_gal"]), np.max(
    flagship["ra_gal"])  # DEC_MIN, DEC_MAX, RA_MIN, RA_MAX

# ----------- Create the catalog ---------------------------------------------------------------------------------------
none_measures = []
shape_measures = []
none_pixel_measures = []
shape_pixel_measures = []
input_positions = []
input_positions_2 = []

input_magnitudes = []
input_redshifts = []

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

gal_list = [[] for _ in range(total_scenes_per_shear * shear_bins)]
gal_list2 = [[] for _ in range(total_scenes_per_shear * shear_bins)]
positions = [[] for _ in range(total_scenes_per_shear * shear_bins)]
positions_2 = [[] for _ in range(total_scenes_per_shear * shear_bins)]

total_scenes = total_scenes_per_shear * shear_bins
x_scenes = int(np.sqrt(total_scenes))
y_scenes = math.ceil(total_scenes / x_scenes)

angular_size = (complete_image_size - 2. * 1.5 / pixel_scale) * pixel_scale / 3600

x = np.linspace(ra_min_org, ra_min_org + x_scenes * angular_size / np.cos(dec_min_org * np.pi / 180), x_scenes+1)
y = np.linspace(dec_min_org, dec_min_org + y_scenes * angular_size, y_scenes+1)

if (np.min(x) < CATALOG_LIMITS[2]) or (np.max(x) > CATALOG_LIMITS[3]) or (np.min(y) < CATALOG_LIMITS[0]) or (
        np.max(y) > CATALOG_LIMITS[1]):
    raise ValueError("Out of catalog limits")

grid_x, grid_y = np.meshgrid(x, y)
grid_x = grid_x.flatten()
grid_y = grid_y.flatten()

grid_counter = 0
for scene in range(total_scenes_per_shear):
    start_scene = timeit.default_timer()
    failure_counter = 0
    for m in range(shear_bins):
        # --------------------------------- CREATE GALAXY LIST FOR EACH RUN -------------------------------------------
        start_input_building = timeit.default_timer()
        count = 0

        magnitudes = []
        redshifts = []

        ra_min = grid_x[grid_counter]
        ra_max = ra_min + angular_size / np.cos(dec_min_org * np.pi / 180)

        dec_min = grid_y[grid_counter]  # + (total_scenes_per_shear * m + scene) * 0.1
        dec_max = dec_min + angular_size  # + (total_scenes_per_shear * m + scene + 1) * 0.1
        print(ra_min, ra_max, dec_min, dec_max)
        mask = ((flagship["ra_gal"] <= ra_max) & (flagship["ra_gal"] > ra_min) & (flagship["dec_gal"] <= dec_max) &
                (flagship["dec_gal"] > dec_min))

        flagship_cut = flagship[mask]

        positions[scene * shear_bins + m] = np.vstack([flagship_cut["ra_gal"], flagship_cut["dec_gal"]])

        if sys.argv[5] == "RANDOM_POS" or sys.argv[5] == "RANDOM_GAL":
            positions_2[scene * shear_bins + m] = np.array(
                [ra_max - ra_min, dec_max - dec_min]) * np.random.random_sample((galaxy_number, 2)) + \
                                                  np.array([ra_min, ra_max])
        else:
            positions_2[scene * shear_bins + m] = positions[scene * shear_bins + m]

        # Convert positions from WCS to image
        canvas, wcs_astropy = fct.SimpleCanvas(ra_min, ra_max, dec_min, dec_max, pixel_scale,
                                               image_size=complete_image_size)
        full_image = canvas.copy()
        wcs = full_image.wcs

        x_gals, y_gals = wcs.toImage(positions[scene * shear_bins + m][0], positions[scene * shear_bins + m][1],
                                     units=galsim.degrees)
        positions[scene * shear_bins + m] = np.vstack([x_gals, y_gals]).T

        if sys.argv[5] == "True":
            canvas, wcs_astropy = fct.SimpleCanvas(ra_min, ra_max, dec_min, dec_max, pixel_scale, rotate=True,
                                                   image_size=complete_image_size)
            full_image = canvas.copy()
            wcs = full_image.wcs

        x_gals, y_gals = wcs.toImage(positions_2[scene * shear_bins + m][0], positions_2[scene * shear_bins + m][1],
                                     units=galsim.degrees)
        positions_2[scene * shear_bins + m] = np.vstack([x_gals, y_gals]).T

        del full_image, canvas, wcs

        input_positions.append(positions[scene * shear_bins + m])
        input_positions_2.append(positions_2[scene * shear_bins + m])

        for i in range(positions[scene * shear_bins + m].shape[0]):

            ellips = flagship_cut["bulge_axis_ratio"][i]
            betas = flagship_cut["disk_angle"][i] * galsim.degrees

            res = fct.generate_gal_from_flagship(flagship_cut, betas, exp_time, gain, zp, pixel_scale,
                                                 sky_level, read_noise, i)

            gal_list[scene * shear_bins + m].append(res[0])
            magnitudes.append(res[2])
            redshifts.append(flagship_cut["observed_redshift_gal"][i])
            theo_sn = res[1]

            if sys.argv[5] == "RANDOM_GAL":
                index2 = random.randint(0, len(flagship_cut) - 1)

                res = fct.generate_gal_from_flagship(flagship_cut, betas, exp_time, gain, zp, pixel_scale,
                                                     sky_level, read_noise, index2)

                gal_list2[scene * shear_bins + m].append(res[0])
                magnitudes.append(res[2])
                redshifts.append(flagship_cut["observed_redshift_gal"][index2])
                theo_sn2 = res[1]

            else:
                magnitudes.append(res[2])
                redshifts.append(flagship_cut["observed_redshift_gal"][i])
                index2 = i
                theo_sn2 = theo_sn

            for k in range(4):

                if k % 2 != 0:
                    columns.append([scene, m, k, positions_2[scene * shear_bins + m][i, 0],
                                    positions_2[scene * shear_bins + m][i, 1],
                                    -2.5 * np.log10(flagship_cut["euclid_vis"][index2]) - 48.6, ellips,
                                    (betas + math.pi / 2 * galsim.radians) / galsim.radians,
                                    flagship_cut["bulge_nsersic"][index2],
                                    flagship_cut["bulge_r50"][index2], flagship_cut["disk_nsersic"][index2],
                                    flagship_cut["disk_r50"][index2], flagship_cut["bulge_fraction"][index2],
                                    flagship_cut["observed_redshift_gal"][index2], theo_sn2])
                else:
                    columns.append(
                        [scene, m, k, positions[scene * shear_bins + m][i, 0], positions[scene * shear_bins + m][i, 1],
                         -2.5 * np.log10(flagship_cut["euclid_vis"][i]) - 48.6, ellips,
                         betas / galsim.radians,
                         flagship_cut["bulge_nsersic"][i],
                         flagship_cut["bulge_r50"][i], flagship_cut["disk_nsersic"][i], flagship_cut["disk_r50"][i],
                         flagship_cut["bulge_fraction"][i], flagship_cut["observed_redshift_gal"][i], theo_sn])

        if sys.argv[5] != "RANDOM_GAL":
            gal_list2[scene * shear_bins + m] = gal_list[scene * shear_bins + m]

        input_magnitudes.append(magnitudes)
        input_redshifts.append(redshifts)
        grid_counter += 1

columns = np.array(columns, dtype=float)

input_catalog = Table([columns[:, i] for i in range(15)], names=(
    'scene_index', 'shear_index', 'cancel_index', 'position_x', 'position_y', 'mag', 'e', 'beta', 'bulge_n', 'bulge_hlr', 'disk_n', 'disk_hlr', 'bulge_fraction', 'z_obs', 's/n'))
# -------------------------- DISTRIBUTE WORK TO RAY -------------------------------------------------------------------
ids = []
rng = galsim.UniformDeviate()
grid_counter = 0
for scene in range(total_scenes_per_shear):
    for m in range(shear_bins):
        seed1 = int(rng() * 1e6)
        seed2 = int(rng() * 1e6)

        ids.append(
            fct.one_scene_lf.remote(m, gal_list[scene * shear_bins + m], gal_list2[scene * shear_bins + m],
                                    positions[scene * shear_bins + m], positions_2[scene * shear_bins + m], scene,
                                    argv_ref, config_ref,
                                    path, psf_ref, 0, index_fits, seed1, grid_x[grid_counter], grid_y[grid_counter]))
        ids.append(
            fct.one_scene_lf.remote(m, gal_list[scene * shear_bins + m], gal_list2[scene * shear_bins + m],
                                    positions[scene * shear_bins + m], positions_2[scene * shear_bins + m], scene,
                                    argv_ref, config_ref,
                                    path, psf_ref, 1, index_fits, seed2, grid_x[grid_counter], grid_y[grid_counter]))
        ids.append(
            fct.one_scene_lf.remote(m, gal_list[scene * shear_bins + m], gal_list2[scene * shear_bins + m],
                                    positions[scene * shear_bins + m], positions_2[scene * shear_bins + m], scene,
                                    argv_ref, config_ref,
                                    path, psf_ref, 2, index_fits, seed1, grid_x[grid_counter], grid_y[grid_counter]))
        ids.append(
            fct.one_scene_lf.remote(m, gal_list[scene * shear_bins + m], gal_list2[scene * shear_bins + m],
                                    positions[scene * shear_bins + m], positions_2[scene * shear_bins + m], scene,
                                    argv_ref, config_ref,
                                    path, psf_ref, 3, index_fits, seed2, grid_x[grid_counter], grid_y[grid_counter]))

        grid_counter += 1

names = ["none", "shape", "none_pixel", "shape_pixel"]
magnitudes = [[] for _ in range(4)]
magnitudes_list = [min_mag + k * (max_mag - min_mag) / (mag_bins) for k in range(mag_bins + 1)]
measure_arrays = [none_measures, shape_measures, none_pixel_measures, shape_pixel_measures]
neighbour_dist = [[], []]

columns = []
while ids:
    ready, not_ready = ray.wait(ids)
    for x in ray.get(ready)[0]:

        init_positions = input_positions[x[5] * shear_bins + x[4]]
        if x[10] % 2 != 0:
            init_positions = input_positions_2[x[5] * shear_bins + x[4]]

        kd_results = np.array(do_kdtree(init_positions, x[6], k=MAX_NEIGHBORS))
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
        if len(nearest_positional_neighbors) != len(x[6]):
            print(f"No neighbour within {MAX_DIST} px found for "
                  f"{len(x[6]) - len(nearest_positional_neighbors)} galaxies in scene {x[5]} at shear {x[4]}!")

        if x[10] % 2 != 0:
            magnitudes_npn = np.array(input_magnitudes[x[5] * shear_bins + x[4]][1::2])[
                nearest_positional_neighbors]
            redshifts_npn = np.array(input_redshifts[x[5] * shear_bins + x[4]][1::2])[
                nearest_positional_neighbors]
        else:
            magnitudes_npn = np.array(input_magnitudes[x[5] * shear_bins + x[4]][::2])[
                nearest_positional_neighbors]
            redshifts_npn = np.array(input_redshifts[x[5] * shear_bins + x[4]][::2])[
                nearest_positional_neighbors]

        for gal in range(len(np.array(x[6])[filter])):
            se_flag_binary = '{:08b}'.format(int(np.array(x[-5])[filter][gal]))
            min_deviation = np.argmin(np.abs(np.subtract(magnitudes_npn[gal], np.array(x[8])[filter][gal])))
            gems_magnitude_optimized = np.array(magnitudes_npn[gal])[min_deviation]
            matching_index = np.array(nearest_positional_neighbors[gal])[min_deviation]

            if simulation.getboolean("source_extractor_morph"):
                columns.append(
                    [x[5], x[4], x[10], x[0], np.array(x[6])[filter][gal][0], np.array(x[6])[filter][gal][1],
                     np.array(x[7])[filter][gal], np.array(x[8])[filter][gal],
                     magnitudes_npn[gal][0], gems_magnitude_optimized, np.array(x[9])[filter][gal],
                     nearest_positional_neighbors[gal][0], matching_index,
                     0 if len(np.unique(nearest_positional_neighbors[gal])) == 1
                     else 1 if (np.abs(magnitudes_npn[gal][0] - magnitudes_npn[gal][1]) > 2) and (
                             len(np.unique(nearest_positional_neighbors[gal])) == 2)
                     else 2, np.array(x[11])[filter][gal], np.array(x[12])[filter][gal], np.array(x[13])[filter][gal],
                     redshifts_npn[gal][0], np.array(x[14])[filter][gal], np.array(x[15])[filter][gal][0],
                     np.array(x[15])[filter][gal][1], int(se_flag_binary[-2]) + int(se_flag_binary[-1]), np.array(x[-4])[filter][gal],
                     np.array(x[-3])[filter][gal], np.array(x[-2])[filter][gal], np.array(x[-1])[filter][gal]])
            else:
                columns.append(
                    [x[5], x[4], x[10], x[0], np.array(x[6])[filter][gal][0], np.array(x[6])[filter][gal][1],
                     np.array(x[7])[filter][gal], np.array(x[8])[filter][gal],
                     magnitudes_npn[gal][0], gems_magnitude_optimized, np.array(x[9])[filter][gal],
                     nearest_positional_neighbors[gal][0], matching_index,
                     0 if len(np.unique(nearest_positional_neighbors[gal])) == 1
                     else 1 if (np.abs(magnitudes_npn[gal][0] - magnitudes_npn[gal][1]) > 2) and (
                             len(np.unique(nearest_positional_neighbors[gal])) == 2)
                     else 2, redshifts_npn[gal][0], np.array(x[11])[filter][gal][0],
                     np.array(x[11])[filter][gal][1], int(se_flag_binary[-2]) + int(se_flag_binary[-1]),
                     np.array(x[-4])[filter][gal], np.array(x[-3])[filter][gal], np.array(x[-2])[filter][gal],
                     np.array(x[-1])[filter][gal]])


    ids = not_ready

columns = np.array(columns, dtype=float)

if simulation.getboolean("source_extractor_morph"):
    length = 26
    shear_results = Table([columns[:, i] for i in range(length)], names=(
        'scene_index', 'shear_index', 'cancel_index', 'input_g1', 'position_x', 'position_y', 'meas_g1', 'mag_auto',
        'mag_gems', 'mag_gems_optimized', 'S/N', 'matching_index', 'matching_index_optimized', 'blending_flag',
        'sersic_n',
        'sersic_re', 'sersic_e', 'matched_z', 'class_star', 'ra', 'dec', 'se_flag', 'kron_radius', 'a_image', 'b_image',
    'elongation'))
else:
    length = 22
    shear_results = Table([columns[:, i] for i in range(length)], names=(
        'scene_index', 'shear_index', 'cancel_index', 'input_g1', 'position_x', 'position_y', 'meas_g1', 'mag_auto',
        'mag_gems', 'mag_gems_optimized', 'S/N', 'matching_index', 'matching_index_optimized', 'blending_flag', 'matched_z', 'ra', 'dec', 'se_flag', 'kron_radius',
    'a_image', 'b_image', 'elongation'))


now = datetime.datetime.now()

current_time = now.strftime("%H-%M-%S")
date_object = datetime.date.today()

os.system('mkdir ' + path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[5]}')

ascii.write(input_catalog,
            path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[5]}/input_catalog.dat',
            overwrite=True)
ascii.write(shear_results,
            path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[5]}/shear_catalog.dat',
            overwrite=True)

# DELETE CATALOGUES AND FITS FILES TO SAVE MEMORY
os.chdir(path + "output")
os.system(f"rm -r source_extractor/{index_fits}")

if simulation.getboolean("output"):
    if not os.path.isdir(
            path + "output/rp_simulations/" + f'run_lf_{date_object}_{current_time}_{sys.argv[5]}/FITS_org'):
        os.mkdir(path + "output/rp_simulations/" + f"run_lf_{date_object}_{current_time}_{sys.argv[5]}/FITS_org")

    os.system(
        'mv ' + path + f'output/FITS{index_fits}/*.fits' + ' ' + path + 'output/rp_simulations/' + f'run_lf_{date_object}_{current_time}_{sys.argv[5]}/FITS_org/')

os.system(f"rm -r FITS{index_fits}")

ray.shutdown()
print(f"Runtime: {timeit.default_timer() - start:.2f} seconds")
