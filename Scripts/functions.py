"""
This module contains most of the functionality of the whole pipeline by outsourcing the main functions.

"""

import numpy as np
import galsim
import ray
import random
import math
import os
import timeit
from galsim.errors import GalSimFFTSizeError
import matplotlib.pyplot as plt


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)

    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    try:
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.sqrt(np.average((values - average) ** 2, weights=weights)) / np.sqrt(np.sum(weights))
    except ZeroDivisionError:
        average = -1
        variance = 0

    return average, variance


def jackknife(array):
    """
    This function takes an array and returns the estimated 1 sigma error via the jackknife method
    """
    array = list(array)
    length = len(array)
    temp = []
    for i in range(len(array)):
        b = array.copy()
        assert length != 1, "Can't jackknife this! Array consists of only 1 argument"
        b.pop(i)
        temp.append(np.average(b))

    return np.sqrt(len(array) - 1) * np.std(temp)


def bootstrap_puyol(array_1, array_2, n, shear_diff, weights_1, weights_2):
    """
    Takes the two arrays of positive and negative shear and returns the standard deviation of the
    resulting response (see Pujol paper eq.33)
    """

    # Random indices to build the bootstrap samples
    indices = np.random.choice(np.arange(len(array_1)), size=(n, len(array_1)))

    # Take from input arrays at the given indices
    array_1 = np.take(array_1, indices, axis=0)
    array_2 = np.take(array_2, indices, axis=0)

    weights_1 = np.take(weights_1, indices, axis=0)
    weights_2 = np.take(weights_2, indices, axis=0)

    # Filter out samples with no measurements
    filter = (np.sum(weights_1, axis=1) != 0) & (np.sum(weights_2, axis=1) != 0)
    array_1 = array_1[filter]
    array_2 = array_2[filter]

    weights_1 = weights_1[filter]
    weights_2 = weights_2[filter]

    # Calculate mean and standard deviation of the responses calculated from the bootstrap samples
    if len(array_1) != 0:
        error = np.std(
            (np.average(array_1, axis=1, weights=weights_1) - np.average(array_2, axis=1,
                                                                         weights=weights_2)) / shear_diff - 1)

        mean = np.mean(
            (np.average(array_1, axis=1, weights=weights_1) - np.average(array_2, axis=1,
                                                                         weights=weights_2)) / shear_diff - 1)
    else:
        error = -1
        mean = -1

    return mean, error


def bootstrap_puyol_grid_galaxy(array_1, array_2, n, shear_diff):
    """
    Takes the two arrays of positive and negative shear and returns the standard deviation of the
    resulting response (see Pujol paper eq.33). In this case the input consists of arrays of arrays, where each
    individual array belongs to one galaxy.
    """

    indices = np.random.choice(len(array_1), size=(n, len(array_1)))

    bootstrap_samples = np.take(array_1, indices, axis=0).reshape(n, -1)
    bootstrap_samples_2 = np.take(array_2, indices, axis=0).reshape(n, -1)

    error = np.std((np.mean(bootstrap_samples, axis=1, where=bootstrap_samples != 0) - np.mean(bootstrap_samples_2,
                                                                                               axis=1,
                                                                                               where=bootstrap_samples_2 != 0)) / shear_diff - 1)
    mean = np.mean((np.mean(bootstrap_samples, axis=1, where=bootstrap_samples != 0) - np.mean(bootstrap_samples_2,
                                                                                               axis=1,
                                                                                               where=bootstrap_samples_2 != 0)) / shear_diff - 1)

    return mean, error


@ray.remote
def generate_bootstrap_samples(meas_g1, per_worker, shear_diff, num_shears):
    """
    This function is used to parallelize the generation of bootstrap samples for the grid analysis.

    It returns the mean responses and the mean measured shears for each bootstrap sample in a list
    """
    samples = []
    samples_c = []
    for _ in range(per_worker):
        # Generate indices with a step size of 2 times the number of different shears, since always 2 galaxies are related.
        indices = np.random.choice(np.arange(0, len(meas_g1), 2 * num_shears),
                                   size=(int(len(meas_g1) / (2 * num_shears))))

        # Add the indices of the galaxies, which belong to the galaxies drawn before.
        indices_c = np.append(indices, [indices + i for i in range(1, 2 * num_shears)])
        if num_shears > 2:
            indices_plus = np.append(indices + 1, [indices + i for i in range(2, num_shears)])

            indices_plus = np.append(indices_plus, indices_plus + num_shears)

            indices_minus = np.append(indices, [indices + i for i in range(1, num_shears - 1)])

            indices_minus = np.append(indices_minus, indices_minus + num_shears)

        else:
            indices_plus = np.append(indices + 1, indices + 3)
            indices_minus = np.append(indices, indices + 2)

        # Build the bootstrap arrays by taking from the table at the drawn indices
        bootstrap_plus = np.take(meas_g1, indices_plus)
        bootstrap_minus = np.take(meas_g1, indices_minus)

        bootstrap_c = np.take(meas_g1, indices_c)

        # Append the sample lists with the means of the bootstrap samples
        samples_c.append(np.mean(bootstrap_c))

        samples.append((np.mean(bootstrap_plus) - np.mean(bootstrap_minus)) / shear_diff - 1)

    return samples, samples_c


def bootstrap_puyol_grid_galaxy_new(meas_g1, n, shear_diff, num_shears, num_cpu):
    """
    This function is the parallelized version of the pujol analysis on the grid.

    It takes the shear table and outputs the uncertainties for both biases.
    """
    samples = []
    samples_c = []

    # Only use multiprocessing when arrays become large
    if len(meas_g1) > 100000:

        ray.init(num_cpus=num_cpu)

        # Put the table in shared memory
        table_ref = ray.put(meas_g1)
        futures = [generate_bootstrap_samples.remote(table_ref, int(n / num_cpu), shear_diff, num_shears)
                   for _ in range(num_cpu)]

        for i in range(num_cpu):
            ready, not_ready = ray.wait(futures)

            for x in ray.get(ready):
                samples.append(x[0])
                samples_c.append(x[1])

            futures = not_ready
            if not futures:
                break

        ray.shutdown()

    else:
        # Generate indices with a step size of 2 times the number of different shears,
        # since always 2 galaxies are related.

        indices = np.random.choice(np.arange(0, len(meas_g1), 2 * num_shears),
                                   size=(int(len(meas_g1) / (2 * num_shears)), n))

        # Add the indices of the galaxies, which belong to the galaxies drawn before.
        indices_c = np.append(indices, np.array([indices + i for i in range(1, 2 * num_shears)]).reshape(-1, n), axis=0)
        if num_shears > 2:
            indices_plus = np.append(indices + 1, np.array([indices + i for i in range(2, num_shears)]).reshape(-1, n),
                                     axis=0)

            indices_plus = np.append(indices_plus, indices_plus + num_shears, axis=0)

            indices_minus = np.append(indices, np.array([indices + i for i in range(1, num_shears - 1)]).reshape(-1, n),
                                      axis=0)

            indices_minus = np.append(indices_minus, indices_minus + num_shears, axis=0)

        else:
            indices_plus = np.append(indices + 1, indices + 3, axis=0)
            indices_minus = np.append(indices, indices + 2, axis=0)

        # Build the bootstrap arrays by taking from the table at the drawn indices
        bootstrap_plus = np.take(meas_g1, indices_plus, axis=0)

        bootstrap_minus = np.take(meas_g1, indices_minus, axis=0)

        bootstrap_c = np.take(meas_g1, indices_c, axis=0)

        # Append the sample lists with the means of the bootstrap samples
        samples_c = np.mean(bootstrap_c, axis=0)

        samples = (np.mean(bootstrap_plus, axis=0) - np.mean(bootstrap_minus, axis=0)) / shear_diff - 1

    try:
        samples = np.concatenate(samples)
        samples_c = np.concatenate(samples_c)
    except ValueError:
        pass

    mean = np.mean(samples)
    error = np.std(samples)
    error_c = np.std(samples_c)

    # Calculate the uncertainty of the uncertainty by splitting the sample in 10 and using the standard deviation
    error_error = np.std([np.std(np.array_split(samples, 10)[i]) for i in range(10)])
    error_error_c = np.std([np.std(np.array_split(samples_c, 10)[i]) for i in range(10)])
    mean_as_estimate = np.mean([np.std(np.array_split(samples, 10)[i]) for i in range(10)])

    return [mean, error, error_c, error_error / np.sqrt(10), error_error_c / np.sqrt(10), mean_as_estimate]


def bootstrap(array, n):
    """
    Takes an array and returns the standard deviation estimated via bootstrap
    """
    samples = []
    for _ in range(n):
        indices = np.random.choice(np.arange(len(array)), size=(len(array)))
        bootstrap = np.take(array, indices)

        samples.append(np.mean(bootstrap))

    return np.std(samples)


def bootstrap_rg_analysis(table, n, every, quantity):
    """
    Bootstrap function for the parallelized grid analysis.

    It takes the shear catalog and outputs the uncertainty and the uncertainty on the uncertainty.
    """
    samples = []

    # Bootstrap implemented as a for loop
    for _ in range(n):
        # Draw random indices to build bootstrap samples
        indices = np.random.choice(np.arange(0, len(table), every), size=(int(len(table) / every)))

        if every != 1:
            # Append indices with the respective indices of shape noise- and pixel noise cancellation
            indices = np.append(indices, [indices + i for i in range(1, every)])

        # Build bootstrap sample by taking at indices
        bootstrap = np.take(table[quantity], indices)

        # Save the mean of the sample in a list
        samples.append(np.mean(bootstrap))

    error = np.std(samples)
    error_err = np.std([np.std(np.array_split(samples, 10)[i]) for i in range(10)])

    return [error, error_err / np.sqrt(10)]


def generate_ellipticity(ellip_rms, ellip_max, seed=0):
    """
    Takes the desired standard deviation of the ellipticity and the maximum ellipticity.
    Returns the generated ellipticity from the rayleigh distribution.
    """

    if seed != 0:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    assert ellip_max > ellip_rms
    tmp = 1.0
    while tmp > ellip_max:
        tmp = rng.rayleigh(ellip_rms)
    return tmp


def generate_psf(lam_min, step_psf, lam_max, data, tel_diam):
    """
    Takes the starting wavelength, the step size, the maximum wavelength, the reference spectrum and telescope
    diameter to generate a galsim psf object.
    Returns this galsim psf object.
    """

    # Estimate the total flux of the spectrum by numerically integrating
    total_flux = np.trapz(data[:, 1][(data[:, 0] > 10 * lam_min) & (data[:, 0] < 10 * lam_max)],
                          data[:, 0][(data[:, 0] > 10 * lam_min) & (data[:, 0] < 10 * lam_max)])
    lam = lam_min

    # Determine the weights by dividing the flux in the wavelength interval by the total flux
    weight = np.trapz(data[:, 1][(data[:, 0] > (lam - step_psf / 2) * 10) & (data[:, 0] < (lam + step_psf / 2) * 10)],
                      data[:, 0][(data[:, 0] > (lam - step_psf / 2) * 10) & (
                              data[:, 0] < (lam + step_psf / 2) * 10)]) / total_flux
    lam_over_dia = lam * 1.e-9 / tel_diam * 206265
    psf = weight * galsim.OpticalPSF(lam_over_dia, obscuration=0.29, nstruts=6)

    # Add the monochromatic optical PSF's to one combined weighted PSF
    while lam <= 900:
        lam += step_psf
        weight = np.trapz(
            data[:, 1][(data[:, 0] > (lam - step_psf / 2) * 10) & (data[:, 0] < (lam + step_psf / 2) * 10)],
            data[:, 0][(data[:, 0] > (lam - step_psf / 2) * 10) & (data[:, 0] <
                                                                   (lam + step_psf / 2) * 10)]) / total_flux
        lam_over_dia = lam * 1.e-9 / tel_diam * 206265
        psf += weight * galsim.OpticalPSF(lam_over_dia, obscuration=0.29, nstruts=6)

    return psf


@ray.remote
def multiple_worker(ntimes, indices, ellip_gal, psf, image_sampled_psf, config, argv, input_shears):
    """
    This function defines the work for one core for the fit method on a grid.
    It calls the function handling individual images n times.
    The return value is a list of all return values from the individual image analysis
    """
    results = [worker(indices[k], ellip_gal[k], psf, image_sampled_psf, config, argv, input_shears[k]) for k in
               range(ntimes)]
    return results


def worker(k, ellip_gal, psf, image_sampled_psf, config, argv, input_shear):
    """
    This function does the main work for the linear fit method on a grid. It builds images and measures the shear
    with KSB. If at least one stamp could be measured (or was above the SN cut) it returns important information like
    the galaxy id, the measured ellipticities and the measured signal-to-noise.

    Otherwise -1 is returned
    """

    simulation = config['SIMULATION']
    image = config['IMAGE']

    stamp_xsize = int(image['stamp_xsize'])
    stamp_ysize = int(image['stamp_ysize'])
    ssamp_grid = int(image['ssamp_grid'])

    shift_radius = float(image['shift_radius'])

    shear_min = -float(argv[5])
    shear_max = float(argv[5])

    ring_num = int(argv[3])  # int(simulation['ring_num'])
    rotation_angle = 180.0 / ring_num  # How much is the galaxy turned
    nx_tiles = ring_num

    if int(argv[4]) == 0:
        ny_tiles = 1
    else:
        ny_tiles = int(argv[4]) * 2

    random_seed = int(simulation['random_seed'])
    pixel_scale = float(image['pixel_scale'])
    exp_time = float(image['exp_time'])
    gain = float(image['gain'])
    read_noise = float(image['read_noise'])
    sky = float(image['sky'])
    zp = float(image['zp'])

    # Setup the random number generator for the CCD noise
    if simulation.getboolean("same_galaxies"):
        rng = galsim.UniformDeviate(random_seed + 1 + k % int(argv[2]))
    else:
        rng = galsim.UniformDeviate()

    rng1 = rng.duplicate()
    if simulation.getboolean("same_noise_and_shift"):
        rng2 = rng.duplicate()
        rng3 = rng.duplicate()

    object_number = int(argv[1])

    average_num = int(argv[2])

    sky_level = pixel_scale ** 2 * exp_time / gain * 10 ** (-0.4 * (sky - zp))
    row = 0

    timings = [[], [], []]

    gal_image = galsim.ImageF(stamp_xsize * nx_tiles - 1, stamp_ysize * ny_tiles - 1, scale=pixel_scale)
    column = ny_tiles

    ''' These arrays store the KSB measurements for each of the stamps. The lists are 2 dimensional in order to select 
    shape noise pairs or don't do depending on what shall be tested. '''

    meas_g1 = [[0 for _ in range(nx_tiles)] for _ in range(column)]
    meas_g2 = [[0 for _ in range(nx_tiles)] for _ in range(column)]
    meas_g1_sel = [[0 for _ in range(nx_tiles)] for _ in range(column)]
    meas_g2_sel = [[0 for _ in range(nx_tiles)] for _ in range(column)]
    meas_SNR = [[0 for _ in range(nx_tiles)] for _ in range(column)]
    meas_mag = [[0 for _ in range(nx_tiles)] for _ in range(column)]

    # Apply a random shift_radius. This can be drawn from a square or also a circle
    dx = [0 for _ in range(ring_num)]
    dy = [0 for _ in range(ring_num)]
    for i in range(ring_num):
        if image["shift_type"] == "CIRCLE":
            r = shift_radius * np.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            dx[i] = r * np.cos(theta)
            dy[i] = r * np.sin(theta)
        else:
            dx[i] = 2 * shift_radius * random.random() - shift_radius
            dy[i] = 2 * shift_radius * random.random() - shift_radius

    # Setup the g1 values
    interval = (shear_max - shear_min) / ((object_number / average_num) - 1)

    g1 = shear_min + int(k / average_num) * interval

    fails = [[0, 0], [0, 0]]

    if simulation["g2"] == "ZERO":
        g2 = 0
        bin_index = int(k / average_num)  # np.digitize(g1, binning)
    elif simulation["g2"] == "GAUSS":
        binning = [shear_min + interval * k for k in range(20)]
        # Draw as long as a value between -0.1 and 0.1 is created
        draw = -1
        while (draw > shear_max) or (draw < shear_min):
            draw = np.random.normal(loc=0.0, scale=0.03)

        bin_index = np.digitize(draw, binning)

        g2 = binning[bin_index] - 0.005
    elif simulation["g2"] == "UNIFORM":
        binning = [shear_min + interval * k for k in range(20)]
        bin_index = np.random.choice(np.arange(20))

        g2 = binning[bin_index]

    ''' This is the main loop building the images and measuring them. This loops through the image first in y-direction 
    and then in x-direction. The convolution happens only in the first horizontal line, while all the other additions
     vertically work with those images. '''

    for ix in range(nx_tiles):
        for iy in range(ny_tiles):

            # Define the bounds of one image (up to 4 versions)
            b = galsim.BoundsI(ix * stamp_xsize + 1, (ix + 1) * stamp_xsize - 1,
                               iy * stamp_ysize + 1, (iy + 1) * stamp_ysize - 1)

            if iy == 0:
                sub_gal_image = gal_image[b]

                # Rotate the galaxy if a new column is started
                if iy == 0 and ix != 0:
                    ellip_gal = ellip_gal.rotate(rotation_angle * galsim.degrees)
                    input_shear = np.array(input_shear) * -1
                # Extrinsic shear
                this_gal = ellip_gal.shear(g1=g1, g2=g2)

                # Sub-pixel Shift
                if image.getboolean('shift_galaxies'):
                    if simulation.getboolean("same_noise_and_shift"):
                        if ix == 0:
                            this_gal = this_gal.shift(dx[0], dy[0])
                        if ix == 1:
                            this_gal = this_gal.shift(-dy[0], dx[0])
                    else:
                        this_gal = this_gal.shift(dx[ix], dy[ix])

                # Convolution of shifted and sheared galaxy with the PSF
                final = galsim.Convolve([this_gal, psf])

                # Draw the image and time how long it takes
                start = timeit.default_timer()
                try:
                    final.drawImage(sub_gal_image)
                except GalSimFFTSizeError:
                    return -1

                sub_gal_image = galsim.Image(
                    np.abs(sub_gal_image.array.copy()))  # Avoid slightly negative values after FFT
                # image = final.drawImage(scale=pixel_scale)
                timings[0].append(timeit.default_timer() - start)

                # CCD noise GALAXY
                if simulation.getboolean("same_noise_and_shift") and ix == 1:
                    if simulation.getboolean("two_in_one"):
                        noise = galsim.noise.CCDNoiseHenning(rng1, gain=gain, read_noise=read_noise,
                                                             sky_level=sky_level, inv=True)
                    else:
                        noise = galsim.noise.CCDNoiseHenning(rng2, gain=gain, read_noise=read_noise,
                                                             sky_level=sky_level)
                else:
                    noise = galsim.noise.CCDNoiseHenning(rng, gain=gain, read_noise=read_noise, sky_level=sky_level)

                without_noise = sub_gal_image.array.copy()
                start_noise = timeit.default_timer()
                sub_gal_image.addNoise(noise)
                timings[2].append(timeit.default_timer() - start_noise)
                with_noise = sub_gal_image.array.copy()
                gal_image[b] = sub_gal_image

            elif iy == 1:
                gal_image[b] = galsim.Image(without_noise)
                sub_gal_image = gal_image[b]

                if simulation.getboolean("same_noise_and_shift") and ix == 1:
                    noise = galsim.noise.CCDNoiseHenning(rng3, gain=gain, read_noise=read_noise,
                                                         sky_level=sky_level, inv=True)
                else:
                    noise = galsim.noise.CCDNoiseHenning(rng1, gain=gain, read_noise=read_noise,
                                                         sky_level=sky_level, inv=True)

                sub_gal_image.addNoise(noise)

            # Subsample the image to make the measurement easier for KSB
            subsampled_image = sub_gal_image.subsample(ssamp_grid, ssamp_grid)

            # Find S/N and estimated shear
            start = timeit.default_timer()
            params = galsim.hsm.HSMParams(ksb_sig_factor=1.0)
            results = galsim.hsm.EstimateShear(subsampled_image, image_sampled_psf, shear_est="KSB", strict=False,
                                               hsmparams=params)
            timings[1].append(timeit.default_timer() - start)

            if results.error_message == "":
                adamflux = results.moments_amp
                adamsigma = results.moments_sigma / ssamp_grid
                pixels = sub_gal_image.array.copy()

                # Define the edge of the stamp to measure the sky background there
                edge = list(pixels[0]) + list([i[-1] for i in pixels[1:-1]]) + list(reversed(pixels[-1])) + \
                       list(reversed([i[0] for i in pixels[1:-1]]))

                sigma_sky = 1.4826 * np.median(np.abs(edge - np.median(edge)))

                signal_to_noise = adamflux * gain / np.sqrt(
                    gain * adamflux + np.pi * (3 * adamsigma * np.sqrt(2 * np.log(2))) ** 2 * (
                            gain * sigma_sky) ** 2)


                # Use this for a normal run
                meas_g1[iy][ix] = results.corrected_g1
                meas_g2[iy][ix] = results.corrected_g2

                # Use this for selection bias checks
                meas_g1_sel[iy][ix] = input_shear[0] + g1
                meas_g2_sel[iy][ix] = input_shear[1] + g2

                meas_SNR[iy][ix] = signal_to_noise
                gal_mag_inp = -2.5 * np.log10(gain / exp_time * ellip_gal.original.flux) + zp  # Input magnitude
                if adamflux > 0:
                    meas_mag[iy][ix] = -2.5 * np.log10(gain / exp_time * adamflux) + zp  # Measured magnitude

                gal_hlr = ellip_gal.original.half_light_radius

            else:
                fails[ix][iy] += 1
                if simulation.getboolean("selection"):
                    meas_g1[iy].clear()
                    meas_g2[iy].clear()

    # Flatten the measurement array in the end to make the analysis.
    meas_g1 = np.concatenate(meas_g1)
    meas_g2 = np.concatenate(meas_g2)
    meas_g1_sel = np.concatenate(meas_g1_sel)
    meas_g2_sel = np.concatenate(meas_g2_sel)
    meas_SNR = np.concatenate(meas_SNR)
    meas_mag = np.concatenate(meas_mag)

    # Only use those stamps for analysis which have SN >= SN_cut and could be measured at all.
    if np.sum(meas_g1) != 0:
        # if len(fails) != 0:
        # np.savetxt(sys.stdout,np.reshape(np.array(fails),(-1,2)),fmt="%i")
        return k, timings, bin_index, gal_mag_inp, meas_mag, meas_SNR, gal_image if simulation.getboolean(
            "output") else "", np.concatenate(fails), meas_g1, meas_g2, meas_g1_sel, meas_g2_sel
    else:
        return -1


@ray.remote
def multiple_puyol(ntimes, indices, input_g1, ellip_gal, image_sampled_psf, psf, config, argv):
    """
    This function defines one core for the Pujol method. It calls the function working on individual images
    n times and returns a list of the individual image results. See one_galaxy function
    """
    results = [one_galaxy(indices[k], input_g1[k], ellip_gal[k], image_sampled_psf, psf, config, argv) for k in
               range(ntimes)]
    return results


def one_galaxy(k, input_g1, ellip_gal, image_sampled_psf, psf, config, argv):
    """
    This function does the main work for the Pujol method. It builds the images and measures the shear using KSB.
    If none of the two stamps could be measured or is above the SN cut, then -1 is returned. Otherwise important
    information like signal-to-noise and the measured ellipticities are returned.
    """

    image = config['IMAGE']
    simulation = config['SIMULATION']

    random_seed = int(simulation['random_seed'])

    noise_type = argv[3]  # simulation['noise_kind']

    pixel_scale = float(image['pixel_scale'])
    exp_time = float(image['exp_time'])
    gain = float(image['gain'])
    read_noise = float(image['read_noise'])
    sky = float(image['sky'])
    zp = float(image['zp'])

    stamp_xsize = int(image['stamp_xsize'])
    stamp_ysize = int(image['stamp_ysize'])
    ssamp_grid = int(image['ssamp_grid'])

    shift_radius = float(image['shift_radius'])

    SN_cut = float(simulation['SN_cut'])

    noise_repetitions = int(argv[2])  # int(simulation['noise_repetition'])

    # Calculate sky level
    sky_level = pixel_scale ** 2 * exp_time / gain * 10 ** (-0.4 * (sky - zp))

    # Magnitude of the galaxy
    gal_mag = -2.5 * np.log10(gain / exp_time * ellip_gal.original.flux) + zp
    # print(k, ellip_gal.original.flux, ellip_gal.original.half_light_radius)
    # How many different shears
    num_shears = int(argv[4])

    # Increase the input random seed to have different noise realizations for each worker
    random_seed += k * noise_repetitions

    # The used shears for the 2 images (vary them due to shear variation bias)
    delta_shear = 0.04
    shear_max = 0.1

    sheared_images = []
    noise_arrays = []

    if simulation.getboolean("different_shears"):
        shears = []

        shears.append(-shear_max + random.random() * (2 * shear_max - delta_shear))
        shears.append(shears[0] + delta_shear)
    elif num_shears == 2:
        shears = [-0.02, 0.02]
    else:
        shears = [-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)]

    R_11 = []
    alpha = []
    SNR = []
    # create the overall image
    gal_image = galsim.ImageF(stamp_xsize * num_shears - 1, stamp_ysize * noise_repetitions - 1,
                              scale=pixel_scale)

    count = 0

    # meas = [[0 for _ in range(num_shears - 1)] for _ in range(2)]

    ''' This is the main loop building the images and measuring them. The outer loop handles the noise repetitions, 
    while the inner loop treats the 2 or 11 images of the same galaxy with different shear '''
    meas_g1 = [0 for _ in range(num_shears * noise_repetitions)]
    SNR = [0 for _ in range(num_shears * noise_repetitions)]
    for m in range(noise_repetitions):
        if m % 2 == 0:
            sheared_images = []
            noise_arrays = []

        # Options to add a second shear component without analyzing it explicitely
        if simulation["g2"] == "ZERO":
            draw = 0
        elif simulation["g2"] == "GAUSS":
            # Draw as long as a value between -0.1 and 0.1 is created
            draw = -1
            while (draw > 0.1) or (draw < -0.1):
                draw = np.random.normal(loc=0.0, scale=0.03)
        elif simulation["g2"] == "UNIFORM":
            draw = -1
            while (draw > 0.1) or (draw < -0.1):
                draw = np.random.random() * 0.2 - 0.1

        g2 = draw

        # Apply a random shift_radius. This is drawn uniformly from a circle with radius "shift_radius"
        if image["shift_type"] == "CIRCLE":

            r = shift_radius * np.sqrt(random.random())
            theta = random.random() * 2 * math.pi
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)

        else:
            # random.seed(k*2)
            dx = 2 * shift_radius * random.random() - shift_radius
            dy = 2 * shift_radius * random.random() - shift_radius
            # print(k, dx, dy)

        # Loop for the two images with opposite shears
        for i in range(num_shears):

            rng = galsim.UniformDeviate(random_seed + k + m + 1)

            # Determine part of the image
            b = galsim.BoundsI(i * stamp_xsize + 1, (i + 1) * stamp_xsize - 1,
                               m * stamp_ysize + 1, (m + 1) * stamp_ysize - 1)

            sub_gal_image = gal_image[b]

            # Only for the first noise repetitions do the convolutions.
            if m % 2 == 0:
                if m == 2 and i == 0:
                    ellip_gal = ellip_gal.rotate(90 * galsim.degrees)

                # Extrinsic shear
                this_gal = ellip_gal.shear(g1=shears[i], g2=g2)

                if image.getboolean('shift_galaxies'):
                    this_gal = this_gal.shift(dx, dy)

                # Convolution
                final = galsim.Convolve([this_gal, psf])

                # Draw image
                try:
                    final.drawImage(sub_gal_image)  # TAKES A LOT OF TIME
                except GalSimFFTSizeError:
                    return -1

                sub_gal_image = galsim.Image(
                    np.abs(sub_gal_image.array.copy()))  # Avoid slightly negative values after FFT
                sheared_images.append(sub_gal_image.array.copy())

                # --------- NOISE HANDLING -----------------#

                if noise_type == "CCD":
                    # Add CCD Noise
                    noise = galsim.CCDNoise(rng, gain=gain, read_noise=read_noise, sky_level=sky_level)
                    sub_gal_image.addNoise(noise)

                elif noise_type == "same_CCD":
                    noise = galsim.CCDNoise(rng, gain=gain, read_noise=read_noise, sky_level=sky_level)
                    sub_gal_image.addNoise(noise)

                    if i == 0:
                        noise_image = sub_gal_image.array.copy()
                    elif i == 1:
                        noise_array = noise_image - sheared_images[0]
                        gal_image[b] = galsim.Image(
                            noise_array * np.divide(sheared_images[1], sheared_images[0]) + sheared_images[1])

                elif noise_type == "GAUSS":
                    noise = galsim.CCDNoise(rng, gain=0, read_noise=3 * read_noise / gain, sky_level=0.)
                    sub_gal_image.addNoise(noise)

                elif noise_type == "VARIABLE":
                    noise = galsim.VariableGaussianNoise(rng, galsim.Image(np.sqrt(sub_gal_image.array.copy() +
                                                                                   sky_level)))
                    noise_gauss = galsim.GaussianNoise(rng, sigma=read_noise / gain)
                    sub_gal_image.addNoise(noise)
                    sub_gal_image.addNoise(noise_gauss)

                elif noise_type == "CCD_SIM":
                    # POISSON GALAXY
                    rng = galsim.UniformDeviate(random_seed + k + m + 1)
                    noise = galsim.noise.CCDNoiseHenning(rng, gain=gain, read_noise=read_noise, sky_level=sky_level)

                    sub_gal_image = gal_image[b]
                    sub_gal_image.addNoise(noise)

            else:
                gal_image[b] = galsim.Image(sheared_images[i] - noise_arrays[i])

            # If the same CCD noise shall be used for the two images

            # Measure with KSB
            subsampled_image = sub_gal_image.subsample(ssamp_grid, ssamp_grid)

            # Find S/N and estimated shear
            params = galsim.hsm.HSMParams(ksb_sig_factor=1.0)  # KSB size 3times half light radius
            results = galsim.hsm.EstimateShear(subsampled_image, image_sampled_psf, shear_est="KSB", strict=False,
                                               hsmparams=params)

            if results.error_message == "":
                adamflux = results.moments_amp
                adamsigma = results.moments_sigma / ssamp_grid

                pixels = sub_gal_image.array.copy()

                edge = list(pixels[0]) + list([i[-1] for i in pixels[1:-1]]) + list(reversed(pixels[-1])) + \
                       list(reversed([i[0] for i in pixels[1:-1]]))

                sigma_sky = 1.4826 * np.median(np.abs(edge - np.median(edge)))

                if noise_type != "GAUSS":

                    signal_to_noise = adamflux * gain / np.sqrt(
                        gain * adamflux + np.pi * (3 * adamsigma * np.sqrt(2 * np.log(2))) ** 2 * (
                                gain * sigma_sky) ** 2)

                elif noise_type == "GAUSS":

                    signal_to_noise = adamflux / np.sqrt(
                        np.pi * (3 * adamsigma * np.sqrt(2 * np.log(2))) ** 2 * sigma_sky ** 2)

                meas_g1[m * num_shears + i] = results.corrected_g1
                # if i == 0:
                #     meas[0][i] = results.corrected_g1
                # elif i == (num_shears - 1):
                #     meas[1][i - 1] = results.corrected_g1
                # else:
                #     meas[0][i] = results.corrected_g1
                #     meas[1][i - 1] = results.corrected_g1
                SNR[m * num_shears + i] = signal_to_noise

                if adamflux > 0:
                    gal_mag_meas = -2.5 * np.log10(gain / exp_time * adamflux) + zp  # Measured magnitude
                else:
                    gal_mag_meas = -1

            else:
                count += 1

                gal_mag_meas = -1

        for i in range(num_shears - 1):
            if meas_g1[m * num_shears + i] != 0 and meas_g1[m * num_shears + i + 1] != 0:
                R_11_value = (meas_g1[m * num_shears + i + 1] - meas_g1[m * num_shears + i]) / (
                    np.abs(shears[0] - shears[1]))

                R_11.append(R_11_value)
                c_bias = meas_g1[m * num_shears + i] - shears[i] * R_11_value - input_g1
                alpha.append(c_bias)
                alpha.append(meas_g1[m * num_shears + i + 1] - shears[i + 1] * R_11_value - input_g1)

    if len(R_11) != 0:
        return (
            np.average(R_11) - 1, np.std(R_11) / np.sqrt(len(R_11)), len(R_11), np.average(alpha),
            np.std(alpha) / np.sqrt(len(alpha)), len(alpha), gal_image if simulation.getboolean("output") else "", SNR,
            k, gal_mag, gal_mag_meas,
            meas_g1)
    else:
        return 0, 0, 0, 0, 0, 0, gal_image if simulation.getboolean(
            "output") else "", SNR, k, gal_mag, gal_mag_meas, meas_g1


@ray.remote
def multiple_puyol_whole_matrix(ntimes, indices, input_g1, ellip_gal, image_sampled_psf, psf, config, argv):
    """
    This function defines one core for the Pujol method with the full response matrix.
    It calls the function working on individual images
    n times and returns a list of the individual image results. See one_galaxy function
    """
    results = [one_galaxy_whole_matrix(indices[k], input_g1[k], ellip_gal[k], image_sampled_psf, psf, config, argv) for
               k in
               range(ntimes)]
    return results


def one_galaxy_whole_matrix(k, input_g1, ellip_gal, image_sampled_psf, psf, config, argv):
    """
    This function does the main work for the Pujol method with the full response matrix.
    It builds the images and measures the shear using KSB.
    If none of the two stamps could be measured or is above the SN cut, then -1 is returned. Otherwise a 7 entries long
    tuple is returned looking like

    (average_m, error_m, average_alpha, error_alpha, meas_minus, meas_plus, gal_image)
    """

    image = config['IMAGE']
    simulation = config['SIMULATION']

    random_seed = int(simulation['random_seed'])

    noise_type = argv[3]  # simulation['noise_kind']

    pixel_scale = float(image['pixel_scale'])
    exp_time = float(image['exp_time'])
    gain = float(image['gain'])
    read_noise = float(image['read_noise'])
    sky = float(image['sky'])
    zp = float(image['zp'])

    stamp_xsize = int(image['stamp_xsize'])
    stamp_ysize = int(image['stamp_ysize'])
    ssamp_grid = int(image['ssamp_grid'])

    shift_radius = float(image['shift_radius'])

    SN_cut = float(simulation['SN_cut'])

    noise_repetitions = int(argv[2])  # int(simulation['noise_repetition'])

    # Calculate sky level
    sky_level = pixel_scale ** 2 * exp_time / gain * 10 ** (-0.4 * (sky - zp))

    # How many different shears
    num_shears = 3

    # Increase the input random seed to have different noise realizations for each worker
    random_seed += k * noise_repetitions

    # The used shears for the 2 images (vary them due to shear variation bias)
    delta_shear = 0.04
    shear_max = 0.1

    sheared_images = []
    noise_arrays = []

    if simulation.getboolean("different_shears"):
        shears = []

        shears.append(-shear_max + random.random() * (2 * shear_max - delta_shear))
        shears.append(shears[0] + delta_shear)
    elif num_shears == 2:
        shears_g1 = [-0.02, 0.02]
    elif num_shears == 3:
        offset = float(argv[4])
        shears_g1 = random.choice(
            [[0.02 + offset, -0.02 + offset, 0.02 + offset], [-0.02 + offset, 0.02 + offset, -0.02 + offset]])
        shears_g2 = random.choice(
            [[-0.02 + offset, -0.02 + offset, 0.02 + offset], [0.02 + offset, 0.02 + offset, -0.02 + offset]])
        # shears_g1 = [-0.02, 0.02, 0.02]
        # shears_g2 = [-0.02, -0.02, 0.02]
    else:
        shears_g1 = [-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)]

    R_11 = []
    alpha = []
    SNR = []
    # create the overall image
    gal_image = galsim.ImageF(stamp_xsize * num_shears - 1, stamp_ysize * noise_repetitions - 1,
                              scale=pixel_scale)

    count = 0

    # Apply a random shift_radius. This is drawn uniformly from a circle with radius "shift_radius"
    if image["shift_type"] == "CIRCLE":

        r = shift_radius * np.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)

    else:
        # random.seed(k*2)
        dx = 2 * shift_radius * random.random() - shift_radius
        dy = 2 * shift_radius * random.random() - shift_radius
        # print(k, dx, dy)

    ''' This is the main loop building the images and measuring them. The outer loop handles the noise repetitions, 
    while the inner loop treats the 2 images of the same galaxy with different shear '''

    for m in range(noise_repetitions):
        meas_g1 = [0 for _ in range(num_shears)]
        meas_g2 = [0 for _ in range(num_shears)]

        # Loop for the two images with opposite shears
        for i in range(num_shears):

            rng = galsim.UniformDeviate(random_seed + k + m + 1)

            # Determine part of the image
            b = galsim.BoundsI(i * stamp_xsize + 1, (i + 1) * stamp_xsize - 1,
                               m * stamp_ysize + 1, (m + 1) * stamp_ysize - 1)

            sub_gal_image = gal_image[b]

            # Only for the first noise repetitions do the convolutions.
            if m == 0:
                # Extrinsic shear
                this_gal = ellip_gal.shear(g1=shears_g1[i], g2=shears_g2[i])

                if image.getboolean('shift_galaxies'):
                    this_gal = this_gal.shift(dx, dy)

                # Convolution
                final = galsim.Convolve([this_gal, psf])

                # Draw image
                final.drawImage(sub_gal_image)  # TAKES A LOT OF TIME

                sub_gal_image = galsim.Image(
                    np.abs(sub_gal_image.array.copy()))  # Avoid slightly negative values after FFT

                sheared_images.append(sub_gal_image.array.copy())

                # --------- NOISE HANDLING -----------------#

                if noise_type == "CCD":
                    # Add CCD Noise
                    noise = galsim.CCDNoise(rng, gain=gain, read_noise=read_noise, sky_level=sky_level)
                    sub_gal_image.addNoise(noise)

                elif noise_type == "same_CCD":
                    noise = galsim.CCDNoise(rng, gain=gain, read_noise=read_noise, sky_level=sky_level)
                    sub_gal_image.addNoise(noise)

                    if i == 0:
                        noise_image = sub_gal_image.array.copy()
                    elif i == 1:
                        noise_array = noise_image - sheared_images[0]
                        gal_image[b] = galsim.Image(
                            noise_array * np.divide(sheared_images[1], sheared_images[0]) + sheared_images[1])

                elif noise_type == "GAUSS":
                    noise = galsim.CCDNoise(rng, gain=0, read_noise=3 * read_noise / gain, sky_level=0.)
                    sub_gal_image.addNoise(noise)

                elif noise_type == "VARIABLE":
                    noise = galsim.VariableGaussianNoise(rng, galsim.Image(np.sqrt(sub_gal_image.array.copy() +
                                                                                   sky_level)))
                    noise_gauss = galsim.GaussianNoise(rng, sigma=read_noise / gain)
                    sub_gal_image.addNoise(noise)
                    sub_gal_image.addNoise(noise_gauss)

                elif noise_type == "CCD_SIM":
                    # SKY NOISE
                    sky_image = galsim.Image(
                        np.reshape(np.zeros((stamp_xsize - 1) * (stamp_ysize - 1)), (stamp_xsize - 1, stamp_ysize - 1)))

                    sky_image.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=0, sky_level=sky_level))

                    sky_noise = sky_image.array.copy()

                    # POISSON GALAXY
                    rng = galsim.UniformDeviate(random_seed + k + m + 1)
                    noise = galsim.CCDNoise(rng, gain=gain, read_noise=0., sky_level=0.0)

                    if i == 0:
                        without_noise = sub_gal_image.array.copy()

                        sub_gal_image.addNoise(noise)

                        poisson_noise = sub_gal_image.array.copy() - without_noise

                    # GAUSSIAN NOISE

                    rng = galsim.UniformDeviate(random_seed + k + m + 1)

                    sub_gal_image.addNoise(galsim.GaussianNoise(rng, sigma=read_noise / gain))

                    if i == 0:
                        gal_image[b] = galsim.Image(sub_gal_image.array.copy() + sky_noise)
                        sub_gal_image = gal_image[b]
                    if i != 0:
                        gal_image[b] = galsim.Image(sub_gal_image.array.copy() + sky_noise + poisson_noise *
                                                    np.sqrt(np.divide(sheared_images[i], sheared_images[0])))
                        sub_gal_image = gal_image[b]

                    noise_arrays.append(sub_gal_image.array.copy() - sheared_images[i])
            else:
                gal_image[b] = galsim.Image(sheared_images[i] - noise_arrays[i])

            # If the same CCD noise shall be used for the two images

            # Measure with KSB
            subsampled_image = sub_gal_image.subsample(ssamp_grid, ssamp_grid)

            # Find S/N and estimated shear

            results = galsim.hsm.EstimateShear(subsampled_image, image_sampled_psf, shear_est="KSB", strict=False)

            if results.error_message == "":
                adamflux = results.moments_amp
                adamsigma = results.moments_sigma / ssamp_grid

                pixels = sub_gal_image.array.copy()

                edge = list(pixels[0]) + list([i[-1] for i in pixels[1:-1]]) + list(reversed(pixels[-1])) + \
                       list(reversed([i[0] for i in pixels[1:-1]]))

                sigma_sky = 1.4826 * np.median(np.abs(edge - np.median(edge)))

                if noise_type != "GAUSS":

                    signal_to_noise = adamflux * gain / np.sqrt(
                        gain * adamflux + np.pi * (3 * adamsigma * np.sqrt(2 * np.log(2))) ** 2 * (
                                gain * sigma_sky) ** 2)

                elif noise_type == "GAUSS":

                    signal_to_noise = adamflux / np.sqrt(
                        np.pi * (3 * adamsigma * np.sqrt(2 * np.log(2))) ** 2 * sigma_sky ** 2)

                if signal_to_noise >= SN_cut and results.corrected_g1 > -20 and results.corrected_g1 < 20:
                    meas_g1[i] = results.corrected_g1
                    meas_g2[i] = results.corrected_g2

                    SNR.append(signal_to_noise)

            else:
                count += 1

    response = [[-1, -1], [-1, -1]]
    if meas_g1[0] != 0 and meas_g1[1] != 0:
        response[0][0] = np.sign(shears_g1[0] - offset) * (meas_g1[0] - meas_g1[1]) / 0.04
        response[1][0] = np.sign(shears_g1[0] - offset) * (meas_g2[0] - meas_g2[1]) / 0.04

    if meas_g1[0] != 0 and meas_g1[2] != 0:
        response[1][1] = np.sign(shears_g2[2] - offset) * (meas_g2[2] - meas_g2[0]) / 0.04
        response[0][1] = np.sign(shears_g2[2] - offset) * (meas_g1[2] - meas_g1[0]) / 0.04

    meas = [meas_g1, meas_g2]
    # print(response)
    signs = [np.sign(shears_g1[0] - offset), np.sign(shears_g2[2] - offset), np.sign(shears_g2[1])]
    return np.concatenate(response), gal_image, signs, meas




@ray.remote
def one_scene_lf(m, gal, gal2, positions, positions2, scene, argv, config, path, psf, index, index_fits, seed):
    """
    This function places one galaxy anywhere in the larger image
    """

    image = config['IMAGE']
    simulation = config['SIMULATION']

    timings = []

    exp_time = float(image['exp_time'])
    read_noise = float(image['read_noise'])
    sky = float(image['sky'])
    zp = float(image['zp'])

    gain = float(image['gain'])
    pixel_scale = float(image['pixel_scale'])

    shear_min = -float(argv[5])
    shear_max = float(argv[5])

    stamp_xsize = int(image['stamp_xsize'])
    stamp_ysize = int(image['stamp_ysize'])

    shear_bins = int(simulation['shear_bins'])

    complete_image_size = int(argv[1])

    ssamp_grid = int(image['ssamp_grid'])
    stamp_xsize = int(image['stamp_xsize'])
    stamp_ysize = int(image['stamp_ysize'])

    cut_size = int(image['cut_size'])

    bin_type = simulation['bin_type']

    rng = galsim.UniformDeviate()
    rng1 = rng.duplicate()

    stamp = []
    for i in range(len(positions)):
        g1 = shear_min + m * (shear_max - shear_min) / (shear_bins - 1)

        if simulation["g2"] == "ZERO":
            draw = 0
        elif simulation["g2"] == "GAUSS":
            # Draw as long as a value between -0.1 and 0.1 is created
            draw = -1
            while (draw > 0.1) or (draw < -0.1):
                draw = np.random.normal(loc=0.0, scale=0.03)
        elif simulation["g2"] == "UNIFORM":
            draw = -1
            while (draw > 0.1) or (draw < -0.1):
                draw = np.random.random() * 0.2 - 0.1

        g2 = draw

        x = positions[i][0]
        y = positions[i][1]

        image_pos = galsim.PositionD(x, y)
        image_pos_2 = galsim.PositionD(positions2[i][0], positions2[i][1])

        # Add extrinsic shear
        normal_gal = gal[i].shear(g1=g1, g2=g2)

        # Convolve the sheared version with the
        final = galsim.Convolve([normal_gal, psf])

        # Rotate the unsheared galaxy
        rotated_galaxy = gal2[i].rotate(90 * galsim.degrees)

        # Add the shear also that rotated galaxy
        rotated_galaxy = rotated_galaxy.shear(g1=g1, g2=g2)

        # Convolve the rotated version with the PSF
        rotated_final = galsim.Convolve([rotated_galaxy, psf])

        if index % 2 == 0:
            # Draw the normal version on a stamp
            try:
                stamp_norm = final.drawImage(center=image_pos, nx=stamp_xsize, ny=stamp_ysize, scale=pixel_scale)
            except GalSimFFTSizeError:
                continue

            stamp_norm = galsim.Image(np.abs(stamp_norm.array.copy()), bounds=stamp_norm.bounds)  # Avoid slightly negative values after FFT
            # Draw the rotated version on a stamp
            stamp.append(stamp_norm)

        else:
            try:
                stamp_rotated = rotated_final.drawImage(center=image_pos, nx=stamp_xsize, ny=stamp_ysize, scale=pixel_scale)
            except GalSimFFTSizeError:
                continue

            if argv[6] == "True":
                stamp_rotated = galsim.Image(np.abs(stamp_rotated.array.copy()), bounds=galsim.BoundsI(
                    galsim.PositionI(stamp_rotated.bounds.ymin, complete_image_size - stamp_rotated.bounds.xmax),
                    galsim.PositionI(stamp_rotated.bounds.ymax,
                                     complete_image_size - stamp_rotated.bounds.xmin)))  # Avoid slightly negative values after FFT
            else:
                stamp_rotated = galsim.Image(np.abs(stamp_rotated.array.copy()), bounds=stamp_rotated.bounds)

            stamp.append(stamp_rotated)

    """
        This function creates the large scenes for the linear fit method  from individual stamps and detects sources
        with SourceExtractor.

        The output consists of the source catalogs and the image fits files.
        """


    rng = galsim.UniformDeviate(seed)

    # Calculate sky level
    sky_level = pixel_scale ** 2 * exp_time / gain * 10 ** (-0.4 * (sky - zp))

    # Build the two large images for none and shape and add sky level and Gaussian RON to them
    image = galsim.Image(complete_image_size, complete_image_size, scale=pixel_scale)

    SOURCE_EXTRACTOR_DIR = path + "output/source_extractor"

    def sex(image, output):
        """Construct a sextractor command and run it."""
        # creates a sextractor line e.g sex img.fits -catalog_name -checkimage_name
        os.chdir(SOURCE_EXTRACTOR_DIR)
        if "vol" in path:
            com = "sextractor " + image + " -c " + SOURCE_EXTRACTOR_DIR + "/default.sex" + \
                  " -CATALOG_NAME " + output
        else:
            com = "source-extractor " + image + " -c " + SOURCE_EXTRACTOR_DIR + "/default.sex" + \
                  " -CATALOG_NAME " + output

        # print(com)
        res = os.system(com)

        return res

    # ----------------------------- ANALYSE THE CATALOGS --------------------------------------------------------------

    for i in range(len(stamp)):
        # Find the overlapping bounds between the large image and the individual stamp.
        bounds = stamp[i].bounds & image.bounds

        image[bounds] += stamp[i][bounds]

    image.addNoise(galsim.noise.CCDNoiseHenning(rng, gain=gain, read_noise=read_noise, sky_level=sky_level,
                                                inv=True if index > 1 else False))

    image.write(path + f"output/FITS{index_fits}/catalog_" + f"{scene}_{m}_{index}.fits")

    # --------------------------- SOURCE EXTRACTOR ----------------------------------------------------------------

    IMAGE_DIRECTORY = path + f"output/FITS{index_fits}/catalog_" + f"{scene}_{m}_{index}.fits"

    sex(IMAGE_DIRECTORY, SOURCE_EXTRACTOR_DIR + f"/{index_fits}/" + f"{scene}_{m}_{index}.cat")

    """
        This function does measure the detected sources from SourceExtractor with KSB.

        It returns mainly the measured ellipticities, positions and S/N.
        """

    g1 = shear_min + m * (shear_max - shear_min) / (shear_bins - 1)

    # Filter the PSF with a top hat at pixel scale
    filter_function = galsim.Pixel(scale=pixel_scale)

    psf_1 = galsim.Convolve([psf, filter_function])

    # Draw the convolution on a finer pixel grid for KSB
    image_sampled_psf = psf_1.drawImage(nx=ssamp_grid * stamp_xsize, ny=ssamp_grid * stamp_ysize,
                                        scale=1.0 / ssamp_grid * pixel_scale, method='no_pixel')

    measurements = []


    x_pos = []
    y_pos = []
    # start = timeit.default_timer()
    gal_image = galsim.fits.read(path + f"output/FITS{index_fits}/catalog_" + f"{scene}_{m}_{index}.fits")

    data = np.genfromtxt(path + f"output/source_extractor/{index_fits}/" + f"{scene}_{m}_{index}.cat")

    filter = np.where((data[:, 7] > cut_size) & (data[:, 7] < complete_image_size - cut_size) &
                                (data[:, 8] > cut_size) & (data[:, 8] < complete_image_size - cut_size))

    x_cen = data[:, 7][filter]
    y_cen = data[:, 8][filter]
    mag_auto = data[:, 1][filter]

    flag = data[:, 11][filter]
    kron_radius = data[:, 12][filter]
    a_image = data[:, 13][filter]
    b_image = data[:, 14][filter]
    elongation = data[:, 16][filter]

    measures = []
    magnitudes = []
    s_n = []
    flags = []
    # images = []
    kron_radii = []
    a_images = []
    b_images = []
    elongations = []

    edge = list(gal_image.array[0]) + list([i[-1] for i in gal_image.array[1:-1]]) + list(
        reversed(gal_image.array[-1])) + \
           list(reversed([i[0] for i in gal_image.array[1:-1]]))

    sigma_sky = 1.4826 * np.median(np.abs(edge - np.median(edge)))

    for i in range(len(x_cen)):

        b = galsim.BoundsI(int(x_cen[i]) - cut_size, int(x_cen[i]) + cut_size, int(y_cen[i]) - cut_size,
                           int(y_cen[i]) + cut_size)

        sub_gal_image = gal_image[b & gal_image.bounds]

        subsampled_image = sub_gal_image.subsample(ssamp_grid, ssamp_grid)

        # Find S/N and estimated shear
        results = galsim.hsm.EstimateShear(subsampled_image, image_sampled_psf, shear_est="KSB", strict=False)

        if results.error_message == "":
            adamflux = results.moments_amp
            adamsigma = results.moments_sigma / ssamp_grid

            # pixels = sub_gal_image.array.copy()
            mag_adamom = zp - 2.5 * np.log10(adamflux * gain / exp_time)

            signal_to_noise = adamflux * gain / np.sqrt(
                gain * adamflux + np.pi * (3 * adamsigma * np.sqrt(2 * np.log(2))) ** 2 * (
                        gain * sigma_sky) ** 2)

            measures.append(results.corrected_g1)
            s_n.append(signal_to_noise)
            if bin_type == "MAG_ADAMOM":
                magnitudes.append(mag_adamom)
            else:
                magnitudes.append(mag_auto[i])
            x_pos.append(x_cen[i])
            y_pos.append(y_cen[i])
            flags.append(flag[i])
            kron_radii.append(kron_radius[i])
            a_images.append(a_image[i])
            b_images.append(b_image[i])
            elongations.append(elongation[i])

    error_specific = bootstrap(measures, int(simulation['bootstrap_repetitions']))

    measurements.append(
        [g1, np.average(measures), error_specific, len(measures), m, scene, np.dstack((x_pos, y_pos))[0],
         measures, magnitudes, s_n, flags, index, kron_radii, a_images, b_images, elongations])

    return measurements


@ray.remote
def one_scene_pujol(m, total_scene_count, gal, positions, argv, config, path, psf, num_shears, index_fits):
    """
    This function creates the stamps for the response method for one specific galaxy.
    """
    image = config['IMAGE']
    simulation = config['SIMULATION']

    complete_image_size = int(argv[1])
    random_seed = int(simulation['random_seed'])
    exp_time = float(image['exp_time'])
    gain = float(image['gain'])
    read_noise = float(image['read_noise'])
    sky = float(image['sky'])
    zp = float(image['zp'])
    pixel_scale = float(image['pixel_scale'])
    ssamp_grid = int(image['ssamp_grid'])
    stamp_xsize = int(image['stamp_xsize'])
    stamp_ysize = int(image['stamp_ysize'])
    cut_size = int(image['cut_size'])
    bin_type = simulation['bin_type']

    stamps = []
    for i in range(len(positions)):
        x = positions[i][0]
        y = positions[i][1]

        image_pos = galsim.PositionD(x, y)


        if num_shears == 2:
            shears = [-0.02, 0.02]
        else:
            shears = [-0.1 + 0.2 / (num_shears - 1) * k for k in range(num_shears)]

        if simulation.getboolean("same_but_shear"):
            if argv[6] == "rand":
                central_shear = np.random.rand() * float(simulation["variable_field_mag"]) - float(
                    simulation["variable_field_mag"]) / 2
            elif argv[6] == "zero":
                central_shear = 0
            shears = [central_shear + k * float(simulation["same_but_shear_diff"]) / 2 for k in
                      [-1, 1]]  # Slightly vary the (random) shear field

        if simulation["g2"] == "ZERO":
            draw = 0
        elif simulation["g2"] == "GAUSS":
            # Draw as long as a value between -0.1 and 0.1 is created
            draw = -1
            while (draw > 0.1) or (draw < -0.1):
                draw = np.random.normal(loc=0.0, scale=0.03)
        elif simulation["g2"] == "UNIFORM":
            draw = -1
            while (draw > 0.1) or (draw < -0.1):
                draw = np.random.random() * 0.2 - 0.1

        g2 = draw


        # Extrinsic shear
        normal_gal = gal[i].shear(g1=shears[m], g2=g2)

        # Convolution with the PSF
        final = galsim.Convolve([normal_gal, psf])

        # Drawing on the stamp
        try:
            stamp = final.drawImage(center=image_pos, nx=stamp_xsize, ny=stamp_ysize, scale=pixel_scale)
        except GalSimFFTSizeError:
            return -1

        stamp = galsim.Image(np.abs(stamp.array.copy()),
                             bounds=stamp.bounds)  # Avoid slightly negative values after FFT
        # without_noise = stamp.array.copy()
        # stamp.addNoise(galsim.noise.CCDNoiseHenning(rng, gain=gain, read_noise=0., sky_level=0.))

        stamps.append(stamp)


    # Calculate sky level
    sky_level = pixel_scale ** 2 * exp_time / gain * 10 ** (-0.4 * (sky - zp))

    image_none = galsim.Image(complete_image_size, complete_image_size, scale=pixel_scale)

    for i in range(len(stamps)):
        # Find the overlapping bounds between the large image and the individual stamp.
        bounds = stamps[i].bounds & image_none.bounds

        # Add this to the corresponding location in the large image.
        image_none[bounds] += stamps[i][bounds]

    # Ensure the same seed for the versions belonging to one run
    rng = galsim.UniformDeviate(random_seed + 1 + 2 * total_scene_count)
    image_none.addNoise(galsim.noise.CCDNoiseHenning(rng, gain=gain, read_noise=read_noise, sky_level=sky_level))

    # Write the image to the output directory
    image_none.write(path + f"output/FITS{index_fits}/catalog_none_pujol_{total_scene_count}_{m}.fits")

    # --------------------------------- SOURCE EXTRACTOR --------------------------------------------------------------

    IMAGE_DIRECTORY_NONE = path + f"output/FITS{index_fits}/catalog_none_pujol_{total_scene_count}_{m}.fits"
    SOURCE_EXTRACTOR_DIR = path + "output/source_extractor"

    def sex(image, output):
        """Construct a sextractor command and run it."""
        # creates a sextractor line e.g sex img.fits -catalog_name -checkimage_name
        os.chdir(SOURCE_EXTRACTOR_DIR)
        if "vol" in path:
            com = "sextractor " + image + " -c " + SOURCE_EXTRACTOR_DIR + "/default.sex" + " -CATALOG_NAME " + output
            # print(com)
            res = os.system(com)
        else:
            com = "source-extractor " + image + " -c " + SOURCE_EXTRACTOR_DIR + "/default.sex" + \
                  " -CATALOG_NAME " + output
            # print(com)
            res = os.system(com)
        return res

    sex(IMAGE_DIRECTORY_NONE, SOURCE_EXTRACTOR_DIR + f"/{index_fits}/none_pujol_{total_scene_count}_{m}.cat")



    """
    This function measures the detected sources for the response method using KSB

    It reads in the source catalogs, extracts stamps around the detected centre, and does the measurement.
    The output consists of the measured ellipticities, the detected positions and the S/N.
    """

    meas = []
    x_pos = []
    y_pos = []


    # Filter the PSF with a top hat at pixel scale
    filter_function = galsim.Pixel(scale=pixel_scale)

    psf_1 = galsim.Convolve([psf, filter_function])

    # Draw the convolution on a finer pixel grid for KSB
    image_sampled_psf = psf_1.drawImage(nx=ssamp_grid * stamp_xsize, ny=ssamp_grid * stamp_ysize,
                                        scale=1.0 / ssamp_grid * pixel_scale, method='no_pixel')

    # ----------------- ANALYSE THE CATALOGS --------------------------------------------------------------------------
    gal_image = galsim.fits.read(path + f"output/FITS{index_fits}/catalog_none_pujol_{total_scene_count}_{m}.fits")

    data = np.genfromtxt(path + f"output/source_extractor/{index_fits}/" + f"none_pujol_{total_scene_count}_{m}.cat")

    filter = np.where((data[:, 7] > cut_size) & (data[:, 7] < complete_image_size - cut_size) & (data[:, 8] > cut_size) & (
                data[:, 8] < complete_image_size - cut_size))
    # Exclude the outer pixels of each large scenes because the stamps would be incomplete
    x_cen = data[:, 7][filter]
    y_cen = data[:, 8][filter]
    mag_auto = data[:, 1][
        filter]

    flag = data[:, 11][filter]
    kron_radius = data[:, 12][filter]
    a_image = data[:, 13][filter]
    b_image = data[:, 14][filter]
    elongation = data[:, 16][filter]

    measures = []
    magnitudes = []
    S_N = []
    flags = []
    # images = []
    kron_radii = []
    a_images = []
    b_images = []
    elongations = []

    edge = list(gal_image.array[0]) + list([i[-1] for i in gal_image.array[1:-1]]) + list(
        reversed(gal_image.array[-1])) + \
           list(reversed([i[0] for i in gal_image.array[1:-1]]))

    sigma_sky = 1.4826 * np.median(np.abs(edge - np.median(edge)))

    # Loop through all detected positions
    for i in range(len(x_cen)):

        # Determine image boundaris
        b = galsim.BoundsI(int(x_cen[i]) - cut_size, int(x_cen[i]) + cut_size, int(y_cen[i]) - cut_size,
                           int(y_cen[i]) + cut_size)

        sub_gal_image = gal_image[b & gal_image.bounds]

        # Subsample the image for the measurement again
        subsampled_image = sub_gal_image.subsample(ssamp_grid, ssamp_grid)

        # Find S/N and estimated shear
        results = galsim.hsm.EstimateShear(subsampled_image, image_sampled_psf, shear_est="KSB", strict=False)

        if results.error_message == "":
            adamflux = results.moments_amp
            adamsigma = results.moments_sigma / ssamp_grid

            # pixels = sub_gal_image.array.copy()
            mag_adamom = zp - 2.5 * np.log10(adamflux * gain / exp_time)

            signal_to_noise = adamflux * gain / np.sqrt(
                gain * adamflux + np.pi * (3 * adamsigma * np.sqrt(2 * np.log(2))) ** 2 * (
                        gain * sigma_sky) ** 2)

            measures.append(results.corrected_g1)
            x_pos.append(x_cen[i])
            y_pos.append(y_cen[i])

            if bin_type == "MAG_ADAMOM":
                magnitudes.append(mag_adamom)
            else:
                magnitudes.append(mag_auto[i])

            meas.append(results.corrected_g1)
            S_N.append(signal_to_noise)
            flags.append(flag[i])
            kron_radii.append(kron_radius[i])
            a_images.append(a_image[i])
            b_images.append(b_image[i])
            elongations.append(elongation[i])

    return meas, np.dstack((x_pos, y_pos))[0], m, total_scene_count, magnitudes, S_N, flags, kron_radii, a_images, b_images, elongations








@ray.remote
def one_shear_analysis_old(m, config, argv, data_complete, input_g1, input_g2, magnitudes, every, binning):
    """
    This function does do the analysis for one shear of the grid catalog.

    It returns the Intrinsic Shapes, the measured shears, the input shears and all uncertainties.
    """

    simulation = config["SIMULATION"]
    mag_bins = int(simulation["bins_mag"])
    BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])
    columns = []

    shear_min = -float(argv[5])
    shear_max = float(argv[5])

    object_number = int(argv[1])
    average_num = int(argv[2])

    if simulation["bin_type"] == "GEMS":
        bin_type = "mag_input"
    elif simulation["bin_type"] == "MEAS":
        bin_type = "mag_meas"

    for i in range(int(simulation["time_bins"])):
        for mag in range(mag_bins + 1):

            if mag == mag_bins:
                upper_limit = float(simulation["max_mag"])
                lower_limit = float(simulation["min_mag"])
            else:
                upper_limit = magnitudes[mag + 1]
                lower_limit = magnitudes[mag]

            array_g1 = data_complete[(data_complete["shear_index"] == m)].copy()
            array_g2 = data_complete[(data_complete["shear_index"] == m)].copy()
            input_array_g1 = input_g1[(data_complete["shear_index"] == m)].copy()
            input_array_g2 = input_g2[(data_complete["shear_index"] == m)].copy()

            # Mask the arrays to include only the wanted magnitude range
            array_g1["meas_g1"].mask = (array_g1["meas_g1"].mask) | (array_g1[bin_type] <= lower_limit) | (
                    array_g1[bin_type] >= upper_limit)
            array_g2["meas_g2"].mask = (array_g2["meas_g2"].mask) | (array_g2[bin_type] <= lower_limit) | (
                    array_g2[bin_type] >= upper_limit)

            # Same masks for the input arrays
            input_array_g1.mask = array_g1["meas_g1"].mask
            input_array_g2.mask = array_g2["meas_g2"].mask

            # Split the array for time bins. Make sure that the split is done at complete galaxies and not cutting of
            # certain versions of a galaxy
            array_g1 = array_g1[0: int(len(array_g1) / (i + 1)) - int((len(array_g1) / (i + 1)) % every)]
            array_g2 = array_g2[0: int(len(array_g2) / (i + 1)) - int((len(array_g2) / (i + 1)) % every)]
            input_array_g1 = input_array_g1[0: int(len(array_g1) / (i + 1)) - int((len(array_g1) / (i + 1)) % every)]
            input_array_g2 = input_array_g2[0: int(len(array_g2) / (i + 1)) - int((len(array_g2) / (i + 1)) % every)]

            # A few try excepts to handle empty array_g1s occuring for example for the faintest magnitudes
            if len(array_g1) == 0:
                av_g1_mod = 0
                err_g1_mod = 0
                err_g1_mod_err = 0
                intrinsic_av_g1 = 0

            else:
                av_g1_mod = np.average(array_g1["meas_g1"])
                intrinsic_av_g1 = np.average(input_array_g1)
                bootstrap_g1_mod = bootstrap_rg_analysis(array_g1, BOOTSTRAP_REPETITIONS, every, "meas_g1")

                err_g1_mod = bootstrap_g1_mod[0]

                err_g1_mod_err = bootstrap_g1_mod[1]

            if len(array_g2) == 0:
                av_g2_mod = 0
                err_g2_mod = 0
                err_g2_mod_err = 0
                intrinsic_av_g2 = 0
            else:
                av_g2_mod = np.average(array_g2["meas_g2"])
                intrinsic_av_g2 = np.average(input_array_g2)
                bootstrap_g2_mod = bootstrap_rg_analysis(array_g2, BOOTSTRAP_REPETITIONS, every, "meas_g2")

                err_g2_mod = bootstrap_g2_mod[0]
                err_g2_mod_err = bootstrap_g2_mod[1]

            if mag == mag_bins:
                columns.append([m,
                                shear_min + (shear_max - shear_min) / ((object_number / average_num) - 1) * m,
                                shear_min + (shear_max - shear_min) / ((object_number / average_num) - 1) * m,
                                av_g1_mod, err_g1_mod, err_g1_mod_err, av_g2_mod, err_g2_mod, err_g2_mod_err,
                                len(array_g1), magnitudes[mag], intrinsic_av_g1, intrinsic_av_g2])
            else:
                columns.append([m,
                                shear_min + (shear_max - shear_min) / ((object_number / average_num) - 1) * m,
                                shear_min + (shear_max - shear_min) / ((object_number / average_num) - 1) * m,
                                av_g1_mod, err_g1_mod, err_g1_mod_err, av_g2_mod, err_g2_mod, err_g2_mod_err,
                                len(array_g1), magnitudes[mag + 1], intrinsic_av_g1, intrinsic_av_g2])

    return columns


@ray.remote
def one_scene_analysis(data_complete, path, complete_image_size, galaxy_number, scene, min_mag, max_mag, mag_bins,
                       magnitudes_list, bin_type, shear_min, shear_max, shear_bins):
    """
    This function does the analysis for one scene for the randomly positioned galaxies.

    It returns input shear and measured shear with uncertainties.
    """
    columns = []
    with open(path + f"output/rp_simulations/catalog_results_{complete_image_size}_{galaxy_number}_{scene}.txt",
              "w") as file:
        for j in range(4):
            for i in range(shear_bins):
                for mag in range(mag_bins + 1):
                    if mag == mag_bins:
                        lower_limit = min_mag
                        upper_limit = max_mag
                    else:
                        lower_limit = magnitudes_list[mag]
                        upper_limit = magnitudes_list[mag + 1]

                    meas = data_complete["meas_g1"][
                        (data_complete["scene_index"] == scene) & (data_complete["shear_index"] == i) & (
                                data_complete["cancel_index"] == j) & (data_complete[bin_type] > lower_limit) & (
                                data_complete[bin_type] < upper_limit)]

                    g1 = shear_min + i * (shear_max - shear_min) / (shear_bins - 1)

                    if len(meas) != 0:
                        file.write("%.4f\t %.6f\t %.6f\t %d\t %.1f\n" %
                                   (g1, np.average(meas), np.std(meas) / np.sqrt(len(meas)),
                                    len(meas), magnitudes_list[mag]))
                        columns.append([g1, i, scene, j, np.average(meas), np.std(meas) / np.sqrt(len(meas)),
                                        len(meas), magnitudes_list[mag]])
                    else:
                        file.write("%.4f\t %.6f\t %.6f\t %d\t %.1f\n" %
                                   (g1, -1, -1, 0, magnitudes_list[mag]))
                        columns.append([g1, i, scene, j, -1, -1, 0, magnitudes_list[mag]])

    return columns


def bootstrap_new(array, weights, n):
    """
    Takes an array and returns the standard deviation estimated via bootstrap
    """
    samples = []
    for _ in range(n):
        # print(len(array), np.sum(weights))
        indices = np.random.choice(np.arange(len(array)), size=(len(array)))
        bootstrap = np.take(array, indices)
        bootstrap_weights = np.take(weights, indices)

        samples.append(np.average(bootstrap, weights=bootstrap_weights))

    error = np.std(samples)
    split = np.array_split(samples, 10)
    error_err = np.std([np.std(split[i]) for i in range(10)])

    return error, error_err


@ray.remote
def one_shear_analysis(m, config, argv, meas_comp, meas_weights, meas_comp_bs, meas_weights_bs, magnitudes, index):
    simulation = config["SIMULATION"]
    mag_bins = int(simulation["bins_mag"])
    BOOTSTRAP_REPETITIONS = int(simulation["bootstrap_repetitions"])
    columns = []

    shear_min = -float(argv[5])
    shear_max = float(argv[5])

    object_number = int(argv[1])
    average_num = int(argv[2])

    if simulation["bin_type"] == "GEMS":
        bin_type = "mag_input"
    elif simulation["bin_type"] == "MEAS":
        bin_type = "mag_meas"

    if index == "shear_index_g1":
        shear = "g1"
    else:
        shear = "g2"

    columns = []
    for i in reversed(range(int(simulation["time_bins"]))):
        for mag in range(mag_bins + 1):

            if mag == mag_bins:
                upper_lim = float(simulation["max_mag"])
                lower_lim = float(simulation["min_mag"])
            else:
                upper_lim = magnitudes[mag + 1]
                lower_lim = magnitudes[mag]

            value = meas_comp["meas_" + shear][(meas_comp[index] == m) & (meas_comp["binned_time"] <= i) & (
                    meas_comp["binned_mag"] > lower_lim) & (meas_comp["binned_mag"] < upper_lim)].data
            intr = meas_comp["intr_" + shear][
                (meas_comp[index] == m) & (meas_comp["binned_time"] <= i) & (
                        meas_comp["binned_mag"] > lower_lim) & (meas_comp["binned_mag"] < upper_lim)].data

            selection = meas_comp["meas_" + shear +"_sel"][(meas_comp[index] == m) & (meas_comp["binned_time"] <= i) & (
                    meas_comp["binned_mag"] > lower_lim) & (meas_comp["binned_mag"] < upper_lim)].data
            # A few try excepts to handle empty array_g1s occuring for example for the faintest magnitudes
            if len(value) == 0:
                av_mod = 0
                err_mod = 0
                err_mod_err = 0
                av_sel = 0
                err_sel = 0
                err_sel_err = 0
                intrinsic_av = 0
                length = 0

            else:
                av_mod = np.average(value, weights=meas_weights["meas_" + shear][
                    (meas_comp[index] == m) & (meas_comp["binned_time"] <= i) & (
                            meas_comp["binned_mag"] > lower_lim) & (
                            meas_comp["binned_mag"] < upper_lim)].data)  # np.average(array_g1["meas_g1"])
                length = np.sum(meas_weights["meas_" + shear][
                                    (meas_comp[index] == m) & (meas_comp["binned_time"] <= i) & (
                                            meas_comp["binned_mag"] > lower_lim) & (
                                            meas_comp["binned_mag"] < upper_lim)].data)
                bootstrap_res = bootstrap_new(meas_comp_bs["meas_" + shear][(meas_comp_bs[index] == m) & (
                        meas_comp_bs["binned_time"] <= i) & (meas_comp_bs["binned_mag"] > lower_lim) & (
                                                                                    meas_comp_bs[
                                                                                        "binned_mag"] < upper_lim)].data,
                                              meas_weights_bs["meas_" + shear][(meas_comp_bs[index] == m) & (
                                                      meas_comp_bs["binned_time"] <= i) & (meas_comp_bs[
                                                                                               "binned_mag"] > lower_lim) & (
                                                                                       meas_comp_bs[
                                                                                           "binned_mag"] < upper_lim)].data,
                                              BOOTSTRAP_REPETITIONS)

                err_mod = bootstrap_res[0]
                err_mod_err = bootstrap_res[1]
                intrinsic_av = np.average(intr, weights=meas_weights["meas_" + shear][
                    (meas_comp[index] == m) & (meas_comp["binned_time"] <= i) & (
                            meas_comp["binned_mag"] > lower_lim) & (
                            meas_comp["binned_mag"] < upper_lim)].data)

                av_sel = np.average(selection, weights=meas_weights["meas_" + shear +"_sel"][
                    (meas_comp[index] == m) & (meas_comp["binned_time"] <= i) & (
                            meas_comp["binned_mag"] > lower_lim) & (
                            meas_comp["binned_mag"] < upper_lim)].data)  # np.average(array_g1["meas_g1"])

                bootstrap_res = bootstrap_new(meas_comp_bs["meas_" + shear +"_sel"][(meas_comp_bs[index] == m) & (
                        meas_comp_bs["binned_time"] <= i) & (meas_comp_bs["binned_mag"] > lower_lim) & (
                                                                                    meas_comp_bs[
                                                                                        "binned_mag"] < upper_lim)].data,
                                              meas_weights_bs["meas_" + shear +"_sel"][(meas_comp_bs[index] == m) & (
                                                      meas_comp_bs["binned_time"] <= i) & (meas_comp_bs[
                                                                                               "binned_mag"] > lower_lim) & (
                                                                                       meas_comp_bs[
                                                                                           "binned_mag"] < upper_lim)].data,
                                              BOOTSTRAP_REPETITIONS)

                err_sel = bootstrap_res[0]
                err_sel_err = bootstrap_res[1]

            columns.append([m,
                            shear_min + (shear_max - shear_min) / ((object_number / average_num) - 1) * m,
                            av_mod, err_mod, err_mod_err, av_sel, err_sel, err_sel_err,
                            length, upper_lim, intrinsic_av, i, mag])
    return columns
