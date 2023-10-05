# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:21:48 2021

This code reads in galaxies from the GEMS survey and takes their magnitude, half light radii and Sersic indices 
to build realistic galaxy images. Those images are then additionally sheared with intrinsic ellipticity and 
extrinsic shear. The noise is build with the CCD noise module of galsim.

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
    - how many objects to generate
    - RMS value for the intrinsic ellipticity and a maximum ellipticity

Then these galaxies are duplicated and rotated to cancel shape noise. Having one realisation with added noise and
one with subtracted noise yields an additional method to cancel noise. 

Input half-light-radii are given in pixel values of the GEMS survey corresponding to 0.03"

This code uses multiprocessing to distribute the workload to several cores. One process works on one galaxy and
its respective cancellation methods.

Command line code is
python3 grid_simulation.py <object_number> <average_num> <nx_tiles> <pixel noise times> <shear_interval> <path>
@author: Henning
"""

from astropy.table import Table
import numpy as np
import sys
import os
import math
import logging
import galsim
import random
import timeit
from astropy.table import QTable
from astropy.io import ascii
import configparser
import pickle
import ray
import functions as fct
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime


def linear_function(x, a, b):
    return a * x + b


# ---------------------------- INITIALIZATION --------------------------------------------------------------------------


path = sys.argv[6] + "/"  # Read out the path from the command line

# Time the program
start = timeit.default_timer()

# Read in the configuration file
config = configparser.ConfigParser()
config.read('config_grid.ini')

# Additional randomization
config.set('SIMULATION', 'random_seed', str(np.random.randint(1e12)))

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])
# ----------------------------- READ INPUT CATALOG----------------------------------------------------------------------

''' Just to filter the table '''
# Read the table
t = Table.read(path + 'input/gems_20090807.fits')

# Introduce a mask to filter the entries
mask = (t["GEMS_FLAG"] == 4) & (np.abs(t["ST_MAG_BEST"] - t["ST_MAG_GALFIT"]) < 0.5) & (
        float(simulation["min_mag"]) < t["ST_MAG_GALFIT"]) & \
       (t["ST_MAG_GALFIT"] < float(simulation["max_mag"])) & (0.3 < t["ST_N_GALFIT"]) & (t["ST_N_GALFIT"] < 6.0) & (
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
ray.init(num_cpus=num_cpu)

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

object_number = int(sys.argv[1])

average_num = int(sys.argv[2])

# For each galaxy build different orientations and noise
ring_num = int(sys.argv[3])
per_stamp = 2 * ring_num  # Two different noise realisations
rotation_angle = 180.0 / ring_num  # How much is the galaxy turned
nx_tiles = ring_num

if int(sys.argv[4]) == 0:
    ny_tiles = 1
else:
    ny_tiles = int(sys.argv[4]) * 2

stamp_xsize = int(image['stamp_xsize'])
stamp_ysize = int(image['stamp_ysize'])
ssamp_grid = int(image['ssamp_grid'])

shift_radius = float(image['shift_radius'])

shear_min = -float(sys.argv[5])
shear_max = float(sys.argv[5])

# Setup result table
shear_results = QTable(
    names=('input_g1', 'input_g2', 'meas_g1_mod', 'meas_g1_mod_err', 'meas_g1_mod_err_err', 'meas_g2_mod',
           'meas_g2_mod_err', 'meas_g2_mod_err_err', 'n_pairs', 'mag'))

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
    psf = galsim.Airy(lam * 1.e-9 / tel_diam * 206265)
elif psf_config["psf"] == "GAUSS":
    psf = galsim.Gaussian(fwhm=0.15)

# Filter the PSF with a top hat at pixel scale
filter_function = galsim.Pixel(scale=pixel_scale)

psf_1 = galsim.Convolve([psf, filter_function])

# Draw the convolution on a finer pixel grid for KSB
image_sampled_psf = psf_1.drawImage(nx=ssamp_grid * stamp_xsize, ny=ssamp_grid * stamp_ysize,
                                    scale=1.0 / ssamp_grid * pixel_scale, method='no_pixel')

image_epsf = galsim.ImageF(stamp_xsize, stamp_ysize, scale=pixel_scale)
psf.drawImage(image_epsf)

# Save the PSF in a fits file
file_name_epsf = os.path.join(path + 'output', 'real_galaxies_psf_1.fits')
galsim.fits.write(image_sampled_psf, file_name_epsf)

# Put the parts used by all workers in the shared memory of ray
psf_ref = ray.put(psf)
argv_ref = ray.put(sys.argv)
sampl_psf_ref = ray.put(image_sampled_psf)
config_ref = ray.put(config)

# Error count for KSB
error_count = 0

# --------------------------- CREATE THE INPUTS FOR THE WORKERS --------------------------------------------------------

# If the same galaxies shall be used, initialize their parameters
if simulation.getboolean('same_galaxies'):

    # If there is already a pickled file evident
    if os.path.isfile(path + "input/galaxies.p"):
        pickled = pickle.load(open(path + "input/galaxies.p", "rb"))

        indices = pickled[0]
        ellips = pickled[1]
        betas = pickled[2]

    # Elsewise this needs to be created
    else:

        indices = np.random.randint(0, len(galaxies["GEMS_FLAG"]) - 1, average_num)

        ellips = np.random.rayleigh(ellip_rms, average_num)
        for k in range(len(ellips)):
            ellips[k] = fct.generate_ellipticity(ellip_rms, ellip_max)

        betas = np.random.random(average_num) * 2 * math.pi * galsim.radians

        pickle.dump((indices, ellips, betas), open(path + "input/galaxies.p", "wb"))

images = []  # Collects the output images if wished
gal_list = []  # List of galsim objects to pass to the workers
input_shear = []  # Intrinsic shear to calculate selection bias if wanted
columns = []  # Columns for the input table, which is saved

# Create the galaxy list, from which the workers get their galaxies
for k in range(object_number):
    if simulation.getboolean('same_galaxies'):
        # Repeat the same galaxy properties each "average_number" times

        # Calculate galaxy flux
        gal_flux = exp_time / gain * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][indices[k % average_num]] - zp))

        ellipticity = ellips[k % average_num]
        angle = betas[k % average_num]

        q = galaxies["ST_B_IMAGE"][indices[k % average_num]] / galaxies["ST_A_IMAGE"][indices[k % average_num]]
        # Define the galaxy profile (correct here for different pixel sizes of VIS and GEMS)
        gal = galsim.Sersic(galaxies["ST_N_GALFIT"][indices[k % average_num]],
                            half_light_radius=0.03 * galaxies["ST_RE_GALFIT"][indices[k % average_num]] * np.sqrt(q),
                            flux=gal_flux)

        gal_list.append(gal.shear(g=ellipticity, beta=angle))

        input_shear.append([galsim.Shear(g=ellipticity, beta=angle).g1, galsim.Shear(g=ellipticity, beta=angle).g2])

        # Calculate the theoretical signal to noise from the CCD equation
        theo_sn = exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][indices[k % average_num]] - zp)) / \
                  np.sqrt((exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][indices[k % average_num]] - zp)) +
                           sky_level * gain * math.pi * (
                                   3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][indices[k % average_num]]) ** 2 +
                           (read_noise ** 2 + (gain / 2) ** 2) * math.pi * (
                                   3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][indices[k % average_num]]) ** 2))

        columns.append([k, galaxies["ST_MAG_GALFIT"][indices[k % average_num]],
                        0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][indices[k % average_num]],
                        galaxies["ST_N_GALFIT"][indices[k % average_num]],
                        ellips[(k - 1) % average_num] if k % 2 != 0 else ellips[k % average_num],
                        betas[(k - 1) % average_num] / galsim.radians
                        if k % 2 != 0 else betas[k % average_num] / galsim.radians,
                        input_shear[-1][0], input_shear[-1][1], theo_sn])

    else:
        # Everthing is random
        index = np.random.randint(0, len(galaxies["GEMS_FLAG"]) - 1)

        gal_flux = exp_time / gain * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp))

        ellips = fct.generate_ellipticity(ellip_rms, ellip_max)
        betas = random.random() * 2 * math.pi * galsim.radians

        q = galaxies["ST_B_IMAGE"][index] / galaxies["ST_A_IMAGE"][index]
        # Correct for ellipticity
        gal = galsim.Sersic(galaxies["ST_N_GALFIT"][index],
                            half_light_radius=0.03 * galaxies["ST_RE_GALFIT"][index] * np.sqrt(q), flux=gal_flux)

        gal_list.append(gal.shear(g=ellips, beta=betas))
        input_shear.append([galsim.Shear(g=ellips, beta=betas).g1, galsim.Shear(g=ellips, beta=betas).g2])

        theo_sn = exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp)) / \
                  np.sqrt((exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][index] - zp)) +
                           sky_level * gain * math.pi * (3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][index]) ** 2 +
                           (read_noise ** 2 + (gain / 2) ** 2) * math.pi * (
                                   3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][index]) ** 2))

        columns.append(
            [k, galaxies["ST_MAG_GALFIT"][index], 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][index],
             galaxies["ST_N_GALFIT"][index], ellips, betas / galsim.radians, input_shear[-1][0], input_shear[-1][1],
             theo_sn])

# print(np.average(np.array(input_shear)[:, 0]))

# Convert columns to a numpy array and crate the input table from it
columns = np.array(columns, dtype=float)
input_catalog = Table([columns[:, i] for i in range(9)],
                      names=('galaxy_id', 'mag', 'hlr', 'n', 'e', 'beta', 'intr_g1', 'intr_g2', 'theo_sn'))

# ------------------------- DISTRIBUTE WORK FOR MULTIPROCESSING -------------------------------------------------------

# Calculate how many images shall be calculated within one worker
per_process = int(object_number / num_cpu)

# There might be some leftover if the object number is not divisible by the number of CPU's
rest = object_number - num_cpu * per_process

# Multiprocessing via Ray
futures = [fct.multiple_worker.remote(per_process, range(k * per_process, (k + 1) * per_process),
                                      gal_list[k * per_process:(k + 1) * per_process], psf_ref, sampl_psf_ref,
                                      config_ref, argv_ref, input_shear[k * per_process:(k + 1) * per_process]) for k in
           range(num_cpu)]
if rest != 0:
    futures.append(fct.multiple_worker.remote(rest, range(object_number - rest, object_number),
                                              gal_list[object_number - rest:object_number], psf_ref, sampl_psf_ref,
                                              config_ref, argv_ref, input_shear[object_number - rest:object_number]))
# futures = [worker.remote(k,gal_list[k],sampl_psf_ref) for k in range(object_number)]


# ------------------------ READ OUT THE GENERATED MEASUREMENTS --------------------------------------------------------

interval = (shear_max - shear_min) / ((object_number / average_num) - 1)
binning = [-0.1 + 0.2 / 19 * k for k in range(20)]  # Creates values between -0.1 and 0.1

columns = []
failure_counter = 0
timings = [[], [], []]
for i in range(object_number):
    ready, not_ready = ray.wait(futures)

    for x in ray.get(ready)[0]:
        if x != -1:
            if simulation.getboolean("output"):
                images.append(x[6])

            timings[0].append(x[1][0])
            timings[1].append(x[1][1])
            timings[2].append(x[1][2])
            for m in range(len(x[-1])):
                columns.append(
                    [x[0], int(x[0] / average_num), x[2], shear_min + int(x[0] / average_num) * interval,
                     binning[int(x[2])], x[-4][m], x[-3][m], x[-2][m], x[-1][m],
                     x[4][m], x[3], x[5][m]])
                # shear_catalog.add_row(column)
        else:
            failure_counter += 1
    # del(ready[0])
    futures = not_ready
    if not futures:
        break

timing_draw = np.sort(np.concatenate(timings[0]))[:-num_cpu]
timing_meas = np.sort(np.concatenate(timings[1]))[:-num_cpu]
timing_noise = np.sort(np.concatenate(timings[2]))[:-num_cpu]

print(np.mean(timing_draw), np.mean(timing_meas), np.mean(timing_noise))
print(f"Failure count: {failure_counter}")

# Convert columns to numpy array and create the output table
columns = np.array(columns, dtype=float)
shear_catalog = Table([columns[:, i] for i in range(12)], names=(
    'galaxy_id', 'shear_index_g1', 'shear_index_g2', 'input_g1', 'input_g2', 'meas_g1', 'meas_g2', 'meas_g1_sel', 'meas_g2_sel', 'mag_meas', 'mag_input',
    'S/N'))

# Save the output files in a folder labeled by the current date and time
now = datetime.datetime.now()

current_time = now.strftime("%H-%M-%S")
date_object = datetime.date.today()

os.system('mkdir ' + path + 'output/grid_simulations/' + f'run_lf_{date_object}_{current_time}')

ascii.write(shear_catalog, path + 'output/grid_simulations/' + f'run_lf_{date_object}_{current_time}/shear_catalog.dat',
            overwrite=True)

ascii.write(input_catalog, path + 'output/grid_simulations/' + f'run_lf_{date_object}_{current_time}/input_catalog.dat',
            overwrite=True)

if not os.path.isdir(path + "output/grid_simulations/" + f'run_lf_{date_object}_{current_time}/' + 'fits'):
    os.mkdir(path + "output/grid_simulations/" + f'run_lf_{date_object}_{current_time}/' + 'fits')
file_name = os.path.join(path + "output/grid_simulations/" + f'run_lf_{date_object}_{current_time}/' + 'fits',
                         'real_galaxies.fits')
file_name_epsf = os.path.join(path + "output/grid_simulations/" + f'run_lf_{date_object}_{current_time}/' + 'fits',
                              'real_galaxies_psf.fits')

# Now write the images to a multi-extension fits file.  Each image will be in its own HDU.
if simulation.getboolean("output"):
    galsim.fits.writeMulti(images, file_name)

psfs = [image_sampled_psf, image_epsf]
galsim.fits.writeMulti(psfs, file_name_epsf)

ray.shutdown()
print(f"Runtime: {timeit.default_timer() - start:.2f} seconds")
