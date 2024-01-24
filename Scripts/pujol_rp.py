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
import pickle
import scipy
import datetime
import pyvinecopulib as pv

# ---------------------------- INITIALIZATION --------------------------------------------------------------------------
path = sys.argv[4] + "/"

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


def do_balltree(points, r=10.):
    mytree = scipy.spatial.cKDTree(points)

    return mytree.query_ball_point(points, r=r, p=2)


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
cut_size = int(image['cut_size'])

ra_min_org = float(image["ra_min"])
dec_min_org = float(image["dec_min"])

num_shears = int(sys.argv[3])
total_scenes_per_shear = int(sys.argv[2])

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
else:
    raise TypeError("PSF has to be EUCLID, AIRY or GAUSS")

# Put PSF, config and ARGV in ray's shared memory
psf_ref = ray.put(psf)
config_ref = ray.put(config)
argv_ref = ray.put(sys.argv)

# ----------- Create the PSF model if not already existent -------------------------------------------------------------
if not os.path.isfile(path + "/output/source_extractor/euclid.psf"):
    fct.create_psf_with_psfex(config, sys.argv, path + "/output/source_extractor", psf, grid=False)
elif galsim.fits.read(path + "/output/source_extractor/psf_demo.fits").nrow < complete_image_size:
    os.system(f"rm {path}/output/source_extractor/euclid.psf")
    os.system(f"rm {path}/output/source_extractor/psf_demo.fits")
    fct.create_psf_with_psfex(config, sys.argv, path + "/output/source_extractor", psf, grid=False)

if sys.argv[5] == "False":
    # ----------- Load the flagship catalog ----------------------------------------------------------------------------
    flagship = Table.read("../../simulations/input/flagship.fits")

    patches = total_scenes_per_shear

    CATALOG_LIMITS = np.min(flagship["dec_gal"]), np.max(flagship["dec_gal"]), np.min(flagship["ra_gal"]), np.max(
        flagship["ra_gal"])  # DEC_MIN, DEC_MAX, RA_MIN, RA_MAX

    # ----------- Learn the copula from GOODS --------------------------------------------------------------------------
    if not os.path.isfile(path + "/input/copula.json"):
        cop, X = fct.generate_cops(path)
        cop.to_json(filename=path + "/input/copula.json")
        pickle.dump(X, open(path + "input/reference_array.p", "wb"))
    else:
        cop = pv.Vinecop(filename=path+"/input/copula.json")
        X = pickle.load(open(path + "input/reference_array.p", "rb"))

    # ----------- Create the catalog -----------------------------------------------------------------------------------

    # Define the shear arrays
    if num_shears == 2:
        shears = [-0.02, 0.02]
    else:
        shears = [-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)]

    input_positions = []
    input_magnitudes = []
    input_redshifts = []

    columns = []
    # ------- DETERMINATION OF SKY BACKGROUND FOR THEORETICAL S/N ------------------------------------------------------
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
    magnitudes = [[] for _ in range(total_scenes_per_shear)]
    redshifts = [[] for _ in range(total_scenes_per_shear)]
    gal_list = [[] for _ in range(total_scenes_per_shear)]

    x_scenes = int(np.sqrt(total_scenes_per_shear))
    y_scenes = math.ceil(total_scenes_per_shear / x_scenes)

    angular_size = (complete_image_size - 2. * cut_size) * pixel_scale / 3600

    x = np.linspace(ra_min_org, ra_min_org + x_scenes * angular_size / np.cos(dec_min_org * np.pi / 180), x_scenes+1)
    y = np.linspace(dec_min_org, dec_min_org + y_scenes * angular_size, y_scenes+1)

    if (np.min(x) < CATALOG_LIMITS[2]) or (np.max(x) > CATALOG_LIMITS[3]) or (np.min(y) < CATALOG_LIMITS[0]) or (
            np.max(y) > CATALOG_LIMITS[1]):
        raise ValueError("Out of catalog limits")

    grid_x, grid_y = np.meshgrid(x, y)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    # ------------------------- Parallelized input producing -----------------------------------------------------------

    grid_counter = 0
    ids = []
    for total_scene_count in range(total_scenes_per_shear):
        start = timeit.default_timer()
        ra_min = grid_x[grid_counter]
        ra_max = ra_min + angular_size / np.cos(dec_min_org * np.pi / 180)

        dec_min = grid_y[grid_counter]
        dec_max = dec_min + angular_size
        print(f"RAmin = {ra_min:.3f}, RAmax = {ra_max:.3f}, DECmin = {dec_min:.3f}, DECmax = {dec_max:.3f}")
        mask = ((flagship["ra_gal"] <= ra_max) & (flagship["ra_gal"] > ra_min) & (flagship["dec_gal"] <= dec_max) &
                (flagship["dec_gal"] > dec_min))

        flagship_cut = flagship[mask]

        ids.append(fct.one_scene_pujol_input.remote(total_scene_count, argv_ref, config_ref, path, ra_min, dec_min,
                                                    flagship_cut))
        grid_counter += 1

    while ids:
        ready, not_ready = ray.wait(ids)
        for x in ray.get(ready):
            gal_list[x[0]] = x[1]
            positions[x[0]] = x[2]
            magnitudes[x[0]] = x[3]
            redshifts[x[0]] = x[4]
            columns.append(x[5])

        ids = not_ready

    columns = np.concatenate(columns, axis=0)
    columns_input = np.array(columns, dtype=float)
    input_catalog = Table([columns_input[:, i] for i in range(13)],
                          names=('scene_index', 'position_x', 'position_y', 'mag', 'e', 'beta', 'bulge_n', 'bulge_hlr',
                                 'disk_n', 'disk_hlr', 'bulge_fraction', 'z_obs', 's/n'))

    input_catalog.sort("scene_index")

    # --------------------------------- MEASURE CATALOGS ---------------------------------------------------------------
    ids = []
    grid_counter = 0
    for scene in range(total_scenes_per_shear):
        input_positions.append(positions[scene])
        input_magnitudes.append(magnitudes[scene])
        input_redshifts.append(redshifts[scene])
        for m in range(num_shears):

            ids.append(fct.one_scene_pujol.remote(m, scene, argv_ref,
                                                  config_ref, path, psf_ref, num_shears, index_fits,
                                                  gal_list[scene], positions[scene], grid_x[grid_counter],
                                                  grid_y[grid_counter]))
        grid_counter += 1

else:
    input_catalog = Table.read(sys.argv[6]+"/input_catalog.dat", format="ascii")

    # --------------------------------- MEASURE CATALOGS ---------------------------------------------------------------
    ids = []
    grid_counter = 0
    input_positions = []
    input_magnitudes = []
    input_redshifts= []
    for scene in range(total_scenes_per_shear):
        input_cut = input_catalog[(input_catalog["scene_index"] == scene)]
        positions = np.vstack([np.array(input_cut["position_x"]),
                               np.array(input_cut["position_y"])]).T

        input_positions.append(positions)

        input_magnitudes.append(np.array(input_cut["mag"]))
        input_redshifts.append(np.array(input_cut["z_obs"]))

        del positions, input_cut

        for m in range(num_shears):

            ids.append(fct.one_scene_pujol.remote(m, scene, argv_ref,
                                                  config_ref, path, psf_ref, num_shears, index_fits,
                                                  ))
        grid_counter += 1


columns = []
while ids:
    ready, not_ready = ray.wait(ids)

    results = ray.get(ready)

    # if simulation.getboolean('matching'):
    kd_results = np.array(do_kdtree(input_positions[results[0][3]], results[0][1], k=MAX_NEIGHBORS))

    # Nearest neighbor search for clustering properties
    self_query = do_balltree(results[0][1], r=float(config["MATCHING"]["dist_ball"]))
    nn_except_self = np.array(do_kdtree(results[0][1], results[0][1], k=[2])[1])

    nearest_positional_neighbors = np.where(kd_results[0] <= MAX_DIST, kd_results[1], -1)

    # Filter out galaxies with no neighbours within MAX_DIST
    filter = np.sum(nearest_positional_neighbors, axis=1) != -MAX_NEIGHBORS
    nearest_positional_neighbors = nearest_positional_neighbors[filter]

    nearest_positional_neighbors = nearest_positional_neighbors.astype('int32')

    # If just no further neighbours are found within MAX_DIST, replace their entries with the nearest neighbour
    nearest_positional_neighbors = np.where(nearest_positional_neighbors == -1,
                                            np.repeat(nearest_positional_neighbors[:, 0], MAX_NEIGHBORS).reshape(
                                                nearest_positional_neighbors.shape), nearest_positional_neighbors)

    # Catch the case where no neighbour is found within the given distance (noise peaks?)
    if len(nearest_positional_neighbors) != len(results[0][0]):
        print(
            f"No neighbour within {MAX_DIST} px found for {len(results[0][0]) - len(nearest_positional_neighbors)} "
            f"galaxies in scene {results[0][3]} at shear {results[0][2]}!")

    magnitudes_npn = np.array(input_magnitudes[results[0][3]])[nearest_positional_neighbors]
    redshifts_npn = np.array(input_redshifts[results[0][3]])[nearest_positional_neighbors]

    for m in range(len(np.array(results[0][0])[filter])):
        se_flag_binary = '{:08b}'.format(int(np.array(results[0][-5])[filter][m]))

        min_deviation = np.argmin(np.abs(np.subtract(magnitudes_npn[m], np.array(results[0][4])[filter][m])))
        gems_magnitude_optimized = np.array(magnitudes_npn[m])[min_deviation]
        matching_index = np.array(nearest_positional_neighbors[m])[min_deviation]
        if simulation.getboolean("source_extractor_morph"):
            columns.append(
                [results[0][3], results[0][2], np.array(results[0][0])[filter][m], np.array(results[0][1])[filter][m][0],
                 np.array(results[0][1])[filter][m][1],
                 np.array(results[0][4])[filter][m], magnitudes_npn[m][0], gems_magnitude_optimized,
                 np.array(results[0][5])[filter][m],
                 nearest_positional_neighbors[m][0], matching_index,
                 0 if len(np.unique(nearest_positional_neighbors[m])) == 1
                 else 1 if (np.abs(magnitudes_npn[m][0] - magnitudes_npn[m][1]) > 2) and (
                             len(np.unique(nearest_positional_neighbors[m])) == 2)
                 else 2, np.array(results[0][6])[filter][m], np.array(results[0][7])[filter][m],
                 np.array(results[0][8])[filter][m], redshifts_npn[m][0], np.array(results[0][9])[filter][m],
                 np.array(results[0][10])[filter][m], int(se_flag_binary[-2]) + int(se_flag_binary[-1]),
                 np.array(results[0][-4])[filter][m], np.array(results[0][-3])[filter][m],
                 np.array(results[0][-2])[filter][m],
                 np.array(results[0][-1])[filter][m], len(self_query[filter][m])-1,
                 np.array(results[0][4])[nn_except_self[filter][m][0]]])
        else:
            columns.append(
                [results[0][3], results[0][2], np.array(results[0][0])[filter][m],
                 np.array(results[0][1])[filter][m][0],
                 np.array(results[0][1])[filter][m][1],
                 np.array(results[0][4])[filter][m], magnitudes_npn[m][0], gems_magnitude_optimized,
                 np.array(results[0][5])[filter][m],
                 nearest_positional_neighbors[m][0], matching_index,
                 0 if len(np.unique(nearest_positional_neighbors[m])) == 1
                 else 1 if (np.abs(magnitudes_npn[m][0] - magnitudes_npn[m][1]) > 2) and (
                         len(np.unique(nearest_positional_neighbors[m])) == 2)
                 else 2, redshifts_npn[m][0], int(se_flag_binary[-2]) + int(se_flag_binary[-1]),
                 np.array(results[0][-4])[filter][m], np.array(results[0][-3])[filter][m],
                 np.array(results[0][-2])[filter][m],
                 np.array(results[0][-1])[filter][m], len(self_query[filter][m])-1,
                 np.array(results[0][4])[nn_except_self[filter][m][0]]])

    ids = not_ready

columns = np.array(columns, dtype=float)

if simulation.getboolean("source_extractor_morph"):
    shear_results = Table([columns[:, i] for i in range(25)],
                          names=('scene_index', 'shear_index', 'meas_g1', 'position_x', 'position_y', 'mag_auto',
                                 'mag_gems', 'mag_gems_optimized', 'S/N', 'matching_index', 'matching_index_optimized',
                                 'blending_flag', 'sersic_n', 'sersic_re', 'sersic_e', 'matched_z', 'class_star',
                                 'aspect_ratio', 'se_flag', 'kron_radius', 'a_image', 'b_image', 'elongation',
                                 'n_neigh', 'mag_neigh'))
else:
    shear_results = Table([columns[:, i] for i in range(20)],
                          names=('scene_index', 'shear_index', 'meas_g1', 'position_x', 'position_y', 'mag_auto',
                                 'mag_gems', 'mag_gems_optimized', 'S/N', 'matching_index', 'matching_index_optimized',
                                 'blending_flag', 'matched_z', 'se_flag', 'kron_radius', 'a_image', 'b_image',
                                 'elongation', 'n_neigh', 'mag_neigh'))

now = datetime.datetime.now()

current_time = now.strftime("%H-%M-%S")
date_object = datetime.date.today()

if sys.argv[5] == "False":
    os.system('mkdir ' + path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}')

    ascii.write(shear_results,
                path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}/shear_catalog.dat',
                overwrite=True)
    ascii.write(input_catalog,
                path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}/input_catalog.dat',
                overwrite=True)
    # print(all_meas[0])

    os.system('cp config_rp.ini ' + path + 'output/rp_simulations/' + f'run_pujol_{date_object}_{current_time}/')

    os.system(f'cp {path}/output/source_extractor/default.sex ' + path + 'output/rp_simulations/'
              + f'run_pujol_{date_object}_{current_time}/')

    if simulation.getboolean("output"):
        if not os.path.isdir(path + "output/rp_simulations/" + f'run_pujol_{date_object}_{current_time}/FITS_org'):
            os.mkdir(path + "output/rp_simulations/" + f'run_pujol_{date_object}_{current_time}/FITS_org')

        os.system(
            'mv ' + path + f'output/FITS{index_fits}/*.fits' + ' ' + path + 'output/rp_simulations/'
            + f'run_pujol_{date_object}_{current_time}/FITS_org/')

else:
    ascii.write(shear_results,
                sys.argv[6]+'/shear_catalog_remeas.dat',
                overwrite=True)
    os.system(f'cp {path}/output/source_extractor/default.sex ' + sys.argv[6] + "/default_remeas.sex")


# DELETE ALL CATALOG AND FITS FILES TO SAVE MEMORY
os.chdir(path + "output")
os.system(f"rm -r FITS{index_fits}")
os.system(f"rm -r source_extractor/{index_fits}")

ray.shutdown()

print(f"{timeit.default_timer() - start} seconds")
