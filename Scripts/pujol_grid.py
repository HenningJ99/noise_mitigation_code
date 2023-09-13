# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:45:18 2021

This script implements the method suggested by Pujol et al. (2018) to estimate the bias of a shear measurements.
Therefore it uses two images of the same galaxy with slightly different shears and uses the response to this shear
change to estimate the bias.

This script is used to study the impact of different types of noise on the result 

Multiprocessing is used here and each process works on one galaxy and its differently sheared versions with possible
different noise realizations

In the config file one can change key parameters of this script

Command line code to execute is

python3 pujol_grid.py <object_number> <noise repetitions>
@author: Henning
"""

from astropy.table import Table
import numpy as np
import sys
import os
import math
import logging
import galsim
import timeit
from astropy.table import QTable
from astropy.io import ascii
import configparser
import pickle
import functions as fct
import ray
import datetime


def bootstrap(array, weights, n):
    indices = np.random.choice(np.arange(len(array)), size=(n, len(array)))
    bootstrap = np.take(array, indices, axis=0).reshape(n, -1)
    weights = np.take(weights, indices, axis=0).reshape(n, -1)

    return np.std(np.average(bootstrap, axis=1, weights=weights))

# ----------------------------- SETUP (READ CONFIGS ETC.) ------------------------------------------------------------ #


# Time the program
start = timeit.default_timer()

# Read Config File
config = configparser.ConfigParser()
config.read('config_grid.ini')

# Additional randomization
config.set('SIMULATION', 'random_seed', str(np.random.randint(1e8)))

image = config['IMAGE']
simulation = config['SIMULATION']
psf_config = config['PSF']

BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])

num_cpu = int(simulation['num_cores']) #psutil.cpu_count()
print(num_cpu)
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

object_number = int(sys.argv[1])  # int(simulation['object_number'])

stamp_xsize = int(image['stamp_xsize'])
stamp_ysize = int(image['stamp_ysize'])
ssamp_grid = int(image['ssamp_grid'])

ellip_rms = float(simulation['ellip_rms'])
ellip_max = float(simulation['ellip_max'])

SN_cut = float(simulation['SN_cut'])

mag_bins = int(simulation["bins_mag"])
time_bins = int(simulation["time_bins"])

mag_min = float(simulation["min_mag"])
mag_max = float(simulation["max_mag"])

noise_repetitions = int(sys.argv[2])  # int(simulation['noise_repetition'])

num_shears = int(sys.argv[4])
if num_shears == 2:
    shears = [-0.02, 0.02]
else:
    shears = [-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)]

# Calculate sky level
sky_level = pixel_scale ** 2 * exp_time / gain * 10 ** (-0.4 * (sky - zp))

# Get working path from command line
path = sys.argv[5]+"/"

''' Just to filter the table '''
# Read the table
t = Table.read(path + 'input/gems_20090807.fits')

# Introduce a mask to filter the entries
mask = (t["GEMS_FLAG"] == 4) & (np.abs(t["ST_MAG_BEST"] - t["ST_MAG_GALFIT"]) < 0.5) & (mag_min < t["ST_MAG_GALFIT"]) & \
       (t["ST_MAG_GALFIT"] < mag_max) & (0.3 < t["ST_N_GALFIT"]) & (t["ST_N_GALFIT"] < 6.0) & (3.0 < t["ST_RE_GALFIT"]) & \
       (t["ST_RE_GALFIT"] < 40.0)

galaxies = t[mask]

''' Build the images '''

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("real_galaxies")

logger.info("%d galaxies have been drawn from the fits file", len(galaxies["GEMS_FLAG"]))

# For the PSF import the vega spectrum
data = np.genfromtxt(path + "input/vega_spectrumdat.sec")

# -------------------------------------- SETUP GALAXIES AND PSF ------------------------------------------------------ #

''' PSF PART'''
# Define the PSF profile
if psf_config["psf"] == "EUCLID":
    # Weight the contribution of each monochromatic PSF with the Vega spectrum
    psf = fct.generate_psf(lam, step_psf, 900, data, tel_diam)
elif psf_config["psf"] == "AIRY":
    psf = galsim.Airy(lam * 1.e-9 / tel_diam * 206265) # galsim.Gaussian(fwhm=0.15, flux=1.)
elif psf_config["psf"] == "GAUSS":
    psf = galsim.Gaussian(fwhm=0.15)

logger.debug('Made PSF profile')

# Filter the PSF with a top hat at pixel scale
filter_function = galsim.Pixel(scale=pixel_scale)

psf_1 = galsim.Convolve([psf, filter_function])

# Draw the convolution on a finer pixel grid for KSB
if simulation["shear_meas"] == "KSB_HENK":
    ssamp_grid = 5

image_sampled_psf = psf_1.drawImage(nx=ssamp_grid * stamp_xsize, ny=ssamp_grid * stamp_ysize,
                                    scale=1.0 / ssamp_grid * pixel_scale, method='no_pixel')
image_epsf = galsim.ImageF(stamp_xsize, stamp_ysize, scale=pixel_scale)
psf.drawImage(image_epsf)
# image_sampled_psf = psf_1.drawImage(gband)
file_name_epsf = os.path.join(path + 'output', 'real_galaxies_psf_1.fits')
galsim.fits.write(image_sampled_psf, file_name_epsf)
# Put the important parts used by all workers in ray's shared memory
argv_ref = ray.put(sys.argv)
config_ref = ray.put(config)
psf_ref = ray.put(psf)
sampl_psf_ref = ray.put(image_sampled_psf)

images = []

# Write RMS into table
RMS = QTable(names=('MAG', 'SERSIC', 'RE', 'SNR', 'ELLIPTICITY', 'BETA', 'RMS'))




# If the same galaxies shall be used, initialize their parameters
if simulation.getboolean('same_galaxies'):
    if os.path.isfile(path + "input/galaxies.p"):
        pickled = pickle.load(open(path + "input/galaxies.p", "rb"))

        indices = pickled[0]
        ellips = pickled[1]
        betas = pickled[2]

    else:

        indices = np.random.randint(0, len(galaxies["GEMS_FLAG"]) - 1, object_number)

        ellips = np.random.rayleigh(ellip_rms, object_number)
        for k in range(len(ellips)):
            ellips[k] = fct.generate_ellipticity(ellip_rms, ellip_max)

        betas = np.random.random(object_number) * 2 * math.pi * galsim.radians

        pickle.dump((indices, ellips, betas), open(path + "input/galaxies.p", "wb"))
else:
    indices = np.random.randint(0, len(galaxies["GEMS_FLAG"]) - 1, object_number)

    ellips = np.random.rayleigh(ellip_rms, object_number)
    for k in range(len(ellips)):
        ellips[k] = fct.generate_ellipticity(ellip_rms, ellip_max)

    betas = np.random.random(object_number) * 2 * math.pi * galsim.radians

gal_list = []
input_shears_compl = []
columns = []
# gal_below25_5 = 0
for k in range(object_number):
    q = galaxies["ST_B_IMAGE"][indices[k]] / galaxies["ST_A_IMAGE"][indices[k]]
    theo_sn = exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][indices[k]] - zp)) / \
              np.sqrt((exp_time * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][indices[k]] - zp)) +
                       sky_level * gain * math.pi * (3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][indices[k]]) ** 2 +
                       (read_noise ** 2 + (gain / 2) ** 2) * math.pi * (
                                   3 * 0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][indices[k]]) ** 2))
    if k % 2 != 0:
        gal_list.append(gal.shear(g=ellips[k-1], beta=betas[k-1] + math.pi / 2 * galsim.radians))

        input_shears_compl.append(galsim.Shear(g=ellips[k-1], beta=betas[k-1] + math.pi / 2 * galsim.radians).g1)

        columns.append([k, galaxies["ST_MAG_GALFIT"][indices[k]],
                        0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][indices[k]], ellips[k-1],
                        galaxies["ST_N_GALFIT"][indices[k]], (betas[k-1] + math.pi / 2 *galsim.radians) / galsim.radians,
                        input_shears_compl[-1], theo_sn])
    else:
        # if galaxies["ST_MAG_GALFIT"][indices[k]] <= 25.5 and galaxies["ST_MAG_GALFIT"][indices[k]] >= 24.5:
        #     gal_below25_5 += 1

        # Calculate galaxy flux
        gal_flux = exp_time / gain * 10 ** (-0.4 * (galaxies["ST_MAG_GALFIT"][indices[k]] - zp))

        # Define the galaxy profile (correct here for different pixel sizes of VIS and GEMS)
        gal = galsim.Sersic(galaxies["ST_N_GALFIT"][indices[k]],
                            half_light_radius=0.03 * galaxies["ST_RE_GALFIT"][indices[k]] * np.sqrt(q), flux=gal_flux)

        gal_list.append(gal.shear(g=ellips[k], beta=betas[k]))

        input_shears_compl.append(galsim.Shear(g=ellips[k], beta=betas[k]).g1)

        columns.append([k, galaxies["ST_MAG_GALFIT"][indices[k]],
                        0.3 * np.sqrt(q) * galaxies["ST_RE_GALFIT"][indices[k]], ellips[k],
                        galaxies["ST_N_GALFIT"][indices[k]], betas[k] / galsim.radians, input_shears_compl[-1],
                        theo_sn])

    #input_catalog.add_row(column)

print(np.average(input_shears_compl))
# print(gal_below25_5)
columns = np.array(columns, dtype=float)
input_catalog = Table([columns[:, i] for i in range(8)], names=('galaxy_id', 'mag', 'hlr', 'e', 'n', 'beta', 'intr_g1', 'theo_sn'))
# ------------------------DISTRIBUTE WORK ----------------------------------------------------------------------------
# Calculate how many images shall be calculated within one worker
per_process = int(object_number / num_cpu)

rest = object_number - num_cpu * per_process

futures = [fct.multiple_puyol.remote(per_process, range(k * per_process, (k + 1) * per_process),
                                     [galsim.Shear(g=ellips[k], beta=betas[k]).g1 for k in
                                      range(k * per_process, (k + 1) * per_process)],
                                     gal_list[k * per_process:(k + 1) * per_process], sampl_psf_ref, psf_ref,
                                     config_ref, argv_ref) for k in range(num_cpu)]
# print(ray.get(futures))
if rest != 0:
    futures.append(fct.multiple_puyol.remote(rest, range(object_number - rest, object_number),
                                             [galsim.Shear(g=ellips[k], beta=betas[k]).g1 for k in
                                              range(object_number - rest, object_number)],
                                             gal_list[object_number - rest:object_number], sampl_psf_ref, psf_ref,
                                             config_ref, argv_ref))

# ------------------------- ANALYSIS -----------------------------------------------------------------------------------

results_compl = []

meas_compl = []
meas_1_compl = []
meas_2_compl = []
gal_mag = []
columns = []
failure_count = 0
for i in range(object_number):
    ready, not_ready = ray.wait(futures)

    for x in ray.get(ready)[0]:
        if x != -1:
            images.append(x[6])
            for p in range(len(x[-1])):
                columns.append([x[-4], x[-3], x[-2], x[-1][p], x[-5][p], x[0], x[1], x[2], x[3], x[4], x[5]])
                #shear_catalog.add_row(column)
        else:
            failure_count += 1
    futures = not_ready
    if not futures:
        break
print(f"Failure count: {failure_count}")
columns = np.array(columns, dtype=float)
shear_catalog = Table([columns[:,i] for i in range(11)], names=("galaxy_id", "mag_inp", "mag_meas", "meas_g1", "S/N", "R11", "R11_err", "R11_len", "alpha", "alpha_err", "alpha_len"))

now = datetime.datetime.now()

current_time = now.strftime("%H-%M-%S")
date_object = datetime.date.today()

os.system('mkdir ' + path + 'output/grid_simulations/' + f'run_puj_{date_object}_{current_time}')

ascii.write(shear_catalog, path + 'output/grid_simulations/' + f'run_puj_{date_object}_{current_time}/shear_catalog.dat',
            overwrite=True)

ascii.write(input_catalog, path + 'output/grid_simulations/' + f'run_puj_{date_object}_{current_time}/input_catalog.dat',
            overwrite=True)

file_name = os.path.join(path + "output/grid_simulations/" + f'run_puj_{date_object}_{current_time}', 'real_galaxies.fits')
if simulation.getboolean("output"):
    galsim.fits.writeMulti(images, file_name)

ray.shutdown()
print(timeit.default_timer() - start)