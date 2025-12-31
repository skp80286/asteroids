#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:23:18 2021
@author: kritti
"""

""" 
Reworked on Sun Mar 09 20:50:30 2025 
@Co-author: Yogesh 
"""

##################################################################################################################
############################################# IMPORT THE PACKAGES ################################################
##################################################################################################################
import argparse
import math
import os
import subprocess
import sys
from datetime import date

import astropy.units as u
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip, sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from image_registration.chi2_shifts import chi2_shift
from photutils.aperture import (
    EllipticalAperture,
    SkyCircularAnnulus,
    SkyCircularAperture,
    aperture_photometry,
)
from photutils.background import Background2D, MedianBackground, ModeEstimatorBackground
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
from photutils.utils import calc_total_error
from scipy import ndimage
from scipy.optimize import curve_fit

##################################################################################################################
############################################# FUNCTION DEFINITIONS ###############################################
##################################################################################################################


def generate_2D_psf(psf_1D, theta, length, size, target):
    """
    Function to generate a synthetic 2D PSF model using the 1D PSF model, position angle, streak length
    and size.

    """
    print("Getting the synthetic 2D PSF of the trailing background stars......")
    print("Length of streak in pixels is " + str(length))
    print("Generating the synthetic 2D PSF model...... Convolving......")
    data = np.zeros((size, size), dtype=np.float64)
    xc, yc = int(size / 2), int(size / 2)  # mid point of streak
    half = int(length / 2)  # half the streak length
    for i in range(half):
        data[xc + i][yc] += 100.0
        data[xc - i][yc] += 100.0
    psf_1D_prime = np.pad(
        psf_1D, ((1, 0), (0, 1)), "constant"
    )  # preprocessing required for convolution
    psf_2D = convolve(data, psf_1D_prime, boundary="fill", normalize_kernel=True)  # convolve
    psf_2D = Cutout2D(
        psf_2D,
        position=(int(size / 2), int(size / 2)),
        size=(length + 30, 30),
        mode="trim",
    ).data
    hdu = fits.PrimaryHDU(data=psf_2D)
    hdu.writeto(target + ".psf.fits", overwrite=True)
    print("Got the 2D PSF model......")
    return psf_2D


def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def half_max_x(x, y):
    half = np.max(y) / 2.0
    fwhm = 0
    signs = np.sign(np.add(y, -half))
    zero_crossings = signs[0:-2] != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]
    if np.size(zero_crossings_i) > 1:
        return [
            lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half),
        ]
    else:
        return fwhm


def get_psf_fwhm(psfModelData):
    x1 = np.arange(len(psfModelData))
    y1 = psfModelData
    psf_fwhm = 0
    hmx_1 = half_max_x(x1, y1)
    if hmx_1 != 0:
        psf_fwhm = hmx_1[1] - hmx_1[0]
        # print("fwhm:", psf_fwhm)
    # psf_fwhm = 2*(psf_fwhm_1*psf_fwhm_2)/(psf_fwhm_1+psf_fwhm_2)
    return psf_fwhm


def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def getFWHM(calImg, ang, width, count):
    # Util image is used to get the counts along the streak length
    utilImg = np.zeros((calImg.shape[0], calImg.shape[1]))

    fwhm = 0
    # All the pixels of util image are zero except fot the ones running along the middle 9 rows and all columns
    # This is basically a horizontal line of pixel width 9 and length equal to the length of the image
    if utilImg.shape[0] != 0 and utilImg.shape[1] != 0:
        for i in range(9):
            for j in range(calImg.shape[1]):
                utilImg[math.floor(calImg.shape[0] / 2) + i - 4][j] = 1

        # Rotate the util image by desired angle so that the horizontal line of ones run perpendicular to the streak
        rotImg = imutils.rotate(utilImg, angle=90 + (180 - ang))

        # Take the original counts of pixels along the perpendicular direction by multiplying the util image and original cutout of streak
        # all other points will give zero except fot the direction that runs perpendicular to streak
        psfImg = np.zeros((utilImg.shape[0], calImg.shape[1]))
        for i in range(psfImg.shape[0]):
            for j in range(psfImg.shape[1]):
                psfImg[i][j] = rotImg[i][j] * calImg[i][j]

        psfFinal = np.zeros((psfImg.shape[0], psfImg.shape[1]))

        psfFinal = imutils.rotate(psfImg, angle=-90 - (180 - ang))

        psfList = []
        row_start = math.floor(psfFinal.shape[0] / 2) - 2
        row_end = math.floor(psfFinal.shape[0] / 2) + 2

        # Take median of all rows so as to reduce the random fluctuations in the counts perpendicular to streak
        for k in range(6, psfFinal.shape[1] - 6):
            median_val = 0
            med_arr = []
            for l in range(5):
                med_arr.append(psfFinal[row_start + l][k])
            if np.sum(med_arr) != 0:
                psfList.append(np.median(np.array(med_arr)))

        psfArray = np.array(psfList)
        max_ind = np.argmax(psfList)
        # To fit gaussian, just take the projection array
        # Starting and ending values are removed as the counts profile essentially drops to zero across that
        # Also, we can very well fit the Gaussian even after rejecting these
        psfArr_Fin = psfArray[max_ind - 10 : max_ind + 10]
        x_val = np.linspace(0, len(np.array(psfArr_Fin)), len(np.array(psfArr_Fin)))
        ax_row = int(count / 5)
        ax_col = count % 5
        # To check whether if the removal of the starting and ending elemets made the array length zero
        if len(psfArr_Fin) > 0:
            psfArr_Fin_nor = psfArr_Fin - np.min(psfArr_Fin)
            # ax[ax_row, ax_col].scatter(x_val, psfArr_Fin_nor)
            if len(psfArr_Fin_nor) > 3:
                p0 = [np.mean(x_val), np.std(x_val), np.max(psfArr_Fin)]
                try:
                    parameters, covariance = curve_fit(
                        gaussian, x_val, psfArr_Fin_nor, p0=p0
                    )
                    sigma = parameters[1]
                    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
                except Exception as e:
                    # There can be error in Gaussian fit if sigma = 0
                    sigma = 0
                    fwhm = 3.8
                # print("******FWHM Value*******", fwhm)
            else:
                fwhm = 0
    return fwhm


def generateFWHMNS(im_path, filepath, streakLen):
    imgs = []
    finalFWHMList = []
    finalPaList = []
    c = 0
    for f in im_path:
        HDUList = fits.open(os.path.join(filepath, f))
        imgs = HDUList[0].data
        print(imgs.shape)
        header = HDUList[0].header
        if "OBS-TYPE" not in header:
            header["OBS-TYPE"] = "NS"
        obs_Type = header["OBS-TYPE"]
        if "NS" in obs_Type:
            fwhmArr = []
            detector_pixels_width = header["NAXIS2"]
            detector_pixels_length = header["NAXIS1"]
            # img_needed = imgs[500:3000,500:3000]
            img_needed = imgs[10:1400, 10:1050]
            img_needed = imgs[
                50 : detector_pixels_length - 50, 50 : detector_pixels_width - 50
            ]

            # _, median, std = sigma_clipped_stats(img_needed[500:1600,500:1600])    #by me mean replaced with _ , takes less space
            _, median, std = sigma_clipped_stats(
                img_needed[
                    100 : detector_pixels_length - 100,
                    100 : detector_pixels_width - 100,
                ]
            )
            data_clip = np.clip(img_needed, median - 2 * std, median + 10 * std)
            # transforming the image in grayscale for streak detection
            img_needed_n = 255 * (
                (data_clip - np.min(data_clip))
                / (np.max(data_clip) - np.min(data_clip))
            )
            mean_n, median_n, std_n = sigma_clipped_stats(
                img_needed_n[500:1600, 500:1600]
            )  # none of these are used
            try:
                cv2.imwrite(os.path.join(filepath, "forPSF_jpg.jpg"), img_needed_n)
                im_data = cv2.imread(
                    os.path.join(filepath, "forPSF_jpg.jpg"), cv2.IMREAD_GRAYSCALE
                )
            except:
                fwhm = 0
                angle = 0
                print(
                    "Failed to read/write the utility image for Pa calculation for NS"
                )
                break
            angle_arr = []
            # _, thr = cv2.threshold(im_data, np.max(im_data)-50, np.max(im_data), type=cv2.THRESH_BINARY)
            _, thr = cv2.threshold(
                im_data,
                np.max(im_data) * 0.5,
                np.max(im_data) * 0.6,
                type=cv2.THRESH_BINARY,
            )
            # Find contours
            contours, hierarchy = cv2.findContours(thr, 1, 2)  # by me
            # print(imgs.shape)
            count = 0
            # print("No of Contours", len(contours),"No of contours")   #by me
            # Iterate over contours
            # print("Getting FWHM and pa for Image:", os.path.join(filepath, f))     #by me
            for cnt in contours:
                # Get area of blob
                area = cv2.contourArea(cnt)
                # Only work with decent size blobs - ignore smallest ones
                if area > streakLen * 2:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(im_data, [box], 0, (0, 155, 255), 2)
                    (x, y), (width, height), angle = rect
                    # print(width,height)
                    # Explanation to this can be found on source detection with open CV
                    # width and height check is just to ensure at which orientation the rotated rectange is pointed to
                    if width < height:
                        angle = 90 + angle
                    else:
                        angle = angle

                    # print(f'angle: {angle}')
                    angle_arr.append(angle)
                    (x_d, y_d, w_d, h_d) = cv2.boundingRect(cnt)
                    cv2.rectangle(
                        im_data, (x_d, y_d), (x_d + w_d, y_d + h_d), (0, 255, 0), 2
                    )
                    # Get indices to get the counts near mid point of the detected source
                    if (x_d > 10) and (y_d > 10):
                        orig_img_x_start = x_d - 10
                        orig_img_x_end = x_d + w_d + 10
                        orig_img_y_start = y_d - 10
                        orig_img_y_end = y_d + h_d + 10
                        print(
                            "orig_img_x_start, orig_img_x_end, orig_img_y_start, orig_img_y_end",
                            orig_img_x_start,
                            orig_img_x_end,
                            orig_img_y_start,
                            orig_img_y_end,
                        )
                        # This is the cutout of the image near the streak midpoint to check for saturation
                        for_saturation_test = img_needed[
                            ((y_d + y_d + h_d) // 2) - 4 : ((y_d + y_d + h_d) // 2) + 4,
                            ((x_d + x_d + w_d) // 2) - 4 : ((x_d + x_d + w_d) // 2) + 4,
                        ]
                        median_pixel = np.median(for_saturation_test)
                        # This is to ensure that the sources with saturated values/(near saturated values) near the mid point of the streak should be avoided
                        if median_pixel < 4000:
                            fwhm = getFWHM(
                                img_needed[
                                    orig_img_y_start:orig_img_y_end,
                                    orig_img_x_start:orig_img_x_end,
                                ],
                                angle,
                                width,
                                count,
                            )
                            fwhmArr.append(fwhm)

                    count += 1
            # To ensure no absurd value of FWHM is added (All of the detected blobs need not be true sources)
            fwhmArr = [i for i in fwhmArr if i != 0]
            angle_arr = [j for j in angle_arr if (j != 90 or j != 0)]
            if (not math.isnan(np.median(fwhmArr))) and np.median(fwhmArr) > 0:
                finalFWHMList.append(np.median(fwhmArr))
                finalPaList.append(np.median(angle_arr))
    median_Pa = np.median(np.array(finalPaList))
    median_fwhm = np.median(np.array(finalFWHMList))

    cv2.imwrite("out.png", im_data)
    try:
        os.remove("saved_1_jpg.jpg")
    except:
        pass
    return median_fwhm, median_Pa


def get_bkg(img_data, dist=25):
    """
    Function to get the background of an image.

    """
    data = img_data
    bkg_estimator = (
        MedianBackground()
        #ModeEstimatorBackground()
    )  # astreaks paper has used Mode as background estimator, check
    mesh_size = int(1.5*dist)
    #mesh_size = 50
    #if mesh_size < 50: mesh_size = 50
    print(f"Estimating the background of the image. Mesh size:{mesh_size}, estimator={bkg_estimator}")
    bkg = Background2D(data, (mesh_size, mesh_size), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    print("Background estimation done......")
    return bkg


def get_sources(img_data2, psf_2D, dist, filename=""):
    """
    Function to detect sources in the image using image segmentation.

    """
    bkg = get_bkg(img_data2, dist)
    img_data = img_data2 - bkg.background
    hdu = fits.PrimaryHDU(img_data)
    hdu.writeto(filename.replace('.fits', f'.bkgsub.fits'), overwrite=True)

    kernel = psf_2D
    convolved_data = convolve(img_data, kernel, normalize_kernel=True)
    hdu = fits.PrimaryHDU(convolved_data)
    hdu.writeto(filename.replace('.fits', f'.convolved.fits'), overwrite=True)

    for factor in [2]:
        print(f'thresholding_sigma={thresholding_sigma*factor}')
        threshold = (
            thresholding_sigma * bkg.background_rms * factor
        )  # the scaling depends on camera?
        print(f'get_sources: threshold={threshold}')

        print("Doing Image Segmentation......")
        segment_map = detect_sources(convolved_data, threshold, npixels=10)
        print(f'segment_map : {segment_map}')
        print("Doing Image Deblending......")
        segm_deblend = deblend_sources(
            convolved_data,
            segment_map,
            npixels=10,
            nlevels=lev,
            contrast=0.001,
            progress_bar=False,
        )
        _, _, sigma = sigma_clipped_stats(img_data)
        effective_gain = gain
        error = calc_total_error(img_data, sigma, effective_gain)
        cat = SourceCatalog(img_data, segm_deblend, error=error)
        columns = ["xcentroid", "ycentroid", "segment_flux", "segment_fluxerr", "area"]
        tbl = cat.to_table(columns=columns)
        tbl["xcentroid"].info.format = "{:.6f}"
        tbl["ycentroid"].info.format = "{:.6f}"
        tbl["segment_flux"].info.format = "{:.6f}"
        tbl["segment_fluxerr"].info.format = "{:.6f}"
        tbl["area"].info.format = "{:.6f}"
        ## Exclude zero and negative flux values
        # eps = 1e-10  # small positive value
        # tbl["segment_flux"] = np.where(tbl["segment_flux"] <= 0, eps, tbl["segment_flux"])
        print(f'size of table before removing rows with negative segment_flux: {len(tbl)}')
        tbl = tbl[tbl["segment_flux"] > 0]
        print(f'size of table after removing rows with negative segment_flux: {len(tbl)}')
        tbl["int_mag"] = -2.5 * np.log10(tbl["segment_flux"])
        tbl["int_mag_err"] = 2.5 * np.log10(
            1 + (tbl["segment_fluxerr"] / tbl["segment_flux"])
        )
        print(str(len(tbl)) + " sources detected......")
    apertures = []
    for obj in cat:
        position = np.transpose((obj.xcentroid, obj.ycentroid))
        theta = obj.orientation.to(u.rad).value
        a = (dist) / 2
        b = 0.3 * a
        print(f"Creating aperture: (x,y,a,b,theta) = ({obj.xcentroid}, {obj.ycentroid}, {a}, {b}, {theta}")
        apertures.append(
            EllipticalAperture(position, a, b, theta=theta)
        )
        break
    area = apertures[0].area
    return tbl, area, segm_deblend


def inject_sources(img_data, tbl):
    """
    Function to inject sources into the synthetic image......

    """
    print("Generating the synthetic image using the sources detected......")
    mean_bkg, median_bkg, sigma_bkg = sigma_clipped_stats(img_data)
    img_data = np.pad(img_data, ((pix_pad, pix_pad), (pix_pad, pix_pad)), "constant")
    syn_data = np.random.normal(mean_bkg, sigma_bkg, np.shape(img_data))
    x, y = np.meshgrid(np.linspace(-30, 30, 60), np.linspace(-30, 30, 60))
    d = np.sqrt(x * x + y * y)
    mu = 0
    area_new = np.pi * ((fwhm / pix_scale) ** 2)
    g = np.exp(-((d - mu) ** 2 / (2.0 * (15 * pix_scale / fwhm) ** 2)))
    g /= np.sum(g)
    src_y = np.asarray(tbl["xcentroid"]) + pix_pad
    src_x = np.asarray(tbl["ycentroid"]) + pix_pad
    src_sum = np.asarray(tbl["segment_flux"])
    area = np.asarray(tbl["area"])
    count = 0
    for i in range(len(src_x)):
        try:
            syn_data[
                int(src_x[i]) - 30 : int(src_x[i]) + 30,
                int(src_y[i]) - 30 : int(src_y[i]) + 30,
            ] += g * src_sum[i] * area_new / (area[i]) ** 0.5
            count += 1
        except:
            continue
    print("Injected " + str(count) + " sources in the synthetic image")
    return syn_data


def get_ast_mag(img_data_old, tbl, img_wcs_header, x_ast, y_ast):
    bkg = get_bkg(img_data_old)
    img_data = img_data_old - bkg.background
    data = np.pad(img_data, ((pix_pad, pix_pad), (pix_pad, pix_pad)), mode="constant")
    x = np.asarray(tbl["xcentroid"])
    y = np.asarray(tbl["ycentroid"])
    wcs = WCS(img_wcs_header)
    ra_, dec_ = wcs.all_pix2world([x + pix_pad], [y + pix_pad], 0)
    ra_c, dec_c = wcs.all_pix2world(
        [int((np.shape(data)[1]) / 2)], [int((np.shape(data)[0]) / 2)], 0
    )
    catNum = "II/349"
    v = Vizier(
        columns=["*"],
        column_filters={"rmag": "<%.2f" % maxmag, "Nd": ">6", "e_rmag": "<<1.086/3"},
        row_limit=-1,
    )
    Q = v.query_region(
        SkyCoord(ra=ra_c[0], dec=dec_c[0], unit=(u.deg, u.deg)),
        radius=str(boxsize) + "m",
        catalog=catNum,
        cache=False,
    )
    nan_mask = np.isnan(ra_) | np.isnan(dec_)
    ra_ = ra_[~nan_mask]
    dec_ = dec_[~nan_mask]

    print("Results from querying vizier......")
    # print(Q[0])
    # img_coord = SkyCoord(ra=ra_[0], dec=dec_[0], unit='degree')
    img_coord = SkyCoord(ra=ra_, dec=dec_, unit="degree")
    vz_coord = SkyCoord(ra=Q[0]["RAJ2000"], dec=Q[0]["DEJ2000"], unit="degree")
    idx_image, idx_ps1, d2d, d3d = vz_coord.search_around_sky(
        img_coord, photoDistThresh * u.arcsec
    )
    print("Found %d good cross-matches" % len(idx_image))
    # print("Doing Zero Point Calculations......")
    offsets = np.asarray(Q[0]["rmag"][idx_ps1]) - np.asarray(tbl["int_mag"][idx_image])
    offsets = sigma_clip(np.asarray(offsets), sigma=2, masked=False, axis=None)
    zero_mean, zero_med, zero_std = sigma_clipped_stats(offsets)
    zeroPoints = {"zp_mean": zero_mean, "zp_median": zero_med, "zp_std": zero_std}
    # print("{} stars used for ZP calculation".format(len(offsets)))
    # print("ZP_mean: " + str(zero_mean))
    # print("ZP_median: " + str(zero_med))
    # print("ZP_sigma: " + str(zero_std))
    corr_mags, corr_mags_err, ps1_mags = [], [], []
    for i in range(len(idx_image)):
        magCorrected = tbl["int_mag"][idx_image[i]] + zeroPoints["zp_median"]
        magCorrectedErr = np.sqrt(
            zeroPoints["zp_std"] ** 2 + tbl["int_mag_err"][idx_image[i]] ** 2
        )
        corr_mags.append(
            round(tbl["int_mag"][idx_image[i]] + zeroPoints["zp_median"], 2)
        )
        corr_mags_err.append(
            round(
                np.sqrt(
                    zeroPoints["zp_std"] ** 2 + tbl["int_mag_err"][idx_image[i]] ** 2
                ),
                2,
            )
        )
    for j in range(len(idx_ps1)):
        ps1_mags.append(round(Q[0]["rmag"][idx_ps1[j]], 2))
    #        print(idx_ps1["mag"][i])
    #        print('Corrected magnitude of %.2f +/- %.2f'%(magCorrected, magCorrectedErr))
    # print("ZP_median = " + str(zero_med))

    # print(ps1_mags)
    print("Working on getting the photometry of asteroid......")

    try:
        #   print(x_ast+pix_pad, y_ast+pix_pad)
        ra_, dec_ = wcs.all_pix2world(y_ast + pix_pad, x_ast + pix_pad, 0)
        # print(str(ra_), str(dec_))
        position = SkyCoord(ra=ra_, dec=dec_, unit=u.deg, frame="icrs")
        apertures = SkyCircularAperture(position, r=apr_radius * u.pix)
        pix_apertures = apertures.to_pixel(wcs)
        phot_table = aperture_photometry(data, pix_apertures)
        for col in phot_table.colnames:
            phot_table[col].info.format = "%.8g"
        # print(phot_table)
        # Create the annulus aperture
        annulus_aperture = SkyCircularAnnulus(
            position, r_in=R_in * u.pix, r_out=R_out * u.pix
        )
        pix_annulus_aperture = annulus_aperture.to_pixel(wcs)
        # Measuring the flux inside an aperture annulus
        error = np.sqrt(data)
        annulus_phot_table = aperture_photometry(
            data, pix_annulus_aperture, error=error
        )
        for col in annulus_phot_table.colnames:
            annulus_phot_table[col].info.format = "%.8g"
        # print(annulus_phot_table)
        bkg_mean = annulus_phot_table["aperture_sum"] / pix_annulus_aperture.area
        bkg_flux = bkg_mean * pix_apertures.area
        # subtract the background flux for this aperture from the source flux
        source_flux = phot_table["aperture_sum"] - bkg_flux
        mag_err_ = 2.5 * np.log10(
            1 + annulus_phot_table["aperture_sum_err"] / source_flux
        )
        mag_err = np.sqrt(mag_err_**2 + zero_std**2)
        source_mag = zero_med - 2.5 * np.log10(source_flux)
        print(
            "Found source magnitude of %.2f +/- %0.2f for aperture of radius %d pixels"
            % (source_mag, mag_err, apr_radius)
        )
    except:
        source_mag, mag_err = -99, -99
    return source_mag, mag_err, zero_med, zero_std


def get_ast_coords(img_data, ast_x, ast_y, cut_size):
    """
    Function to get the coordinates of an asteroid.
    """
    position = (ast_x + pix_pad, ast_y + pix_pad)
    size = (cut_size, cut_size)
    asteroid = Cutout2D(img_data, position, size, mode="trim").data
    x_c, y_c = ast_x + pix_pad, ast_y + pix_pad
    x_m, y_m = np.where(np.asarray(asteroid) == np.max(np.asarray(asteroid)))
    position_m = (x_m, y_m)
    coord_x = x_m - (int(cut_size) / 2) + x_c
    coord_y = y_m - (int(cut_size) / 2) + y_c
    coord_x, coord_y = ast_x, ast_y
    return coord_x, coord_y


def get_ra_dec(img_wcs_header, coord_x, coord_y):
    """
    Function to map the (x, y) coords to RA-Dec.
    """
    wcs = WCS(img_wcs_header)
    ra, dec = wcs.all_pix2world(coord_x, coord_y, 0)
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    # print(c.ra.hms[2])
    # print("RA = " + str(int(c.ra.hms[0])) + ":" + str(int(c.ra.hms[1])) + ":" + str(round(c.ra.hms[2], 2)))
    # print("Dec = " + str(int(c.dec.dms[0])) + ":" + str(np.abs(int(c.dec.dms[1]))) + ":" + str(np.abs(round(c.dec.dms[2], 1))))
    return c


def get_MPC_report_line(img_header, target, coord, mag):
    """
    Function to get the MPC Report ile Line for an image.

    """
    print(f"\n\n### get_MPC_report_line... ###\n\n")
    if float(mag[0]) > 1e-6: 
        source_mag_str = f'{mag[0]:02.2f}r'
        print(f"Magnitude is non-zero: {mag[0]}")
    else: 
        print(f'Magnitude is zero')
        source_mag_str = ' '*7
    exp_time = float(img_header["EXPTIME"])  # seconds
    time = img_header["DATE-OBS"].split("T")[1].split(":")
    time_format = (
        int(time[0])
        + (int(time[1]) / 60)
        + ((float(time[2]) + (exp_time * 0.5)) / 3600)
    ) / 24
    date = img_header["DATE-OBS"].split("T")[0].split("-")
    UT_date = int(date[2]) + time_format
    date_MPC_format = f'{date[0]} {date[1]} {UT_date:.6f}'
    MPC_report_line = (
        "     "
        + target
        + "  C"
        + date_MPC_format
        + f'{int(coord.ra.hms[0]):02}'
        + " "
        + f'{int(coord.ra.hms[1]):02}'
        + " "
        + f'{float(coord.ra.hms[2]):02.3f}'
        + f'{int(coord.dec.dms[0]):+02}'
        + " "
        + f'{np.abs(int(coord.dec.dms[1])):02}'
        + " "
        + f'{np.abs(coord.dec.dms[2]):02.2f}'
        + (" " * 9)
        + source_mag_str
        + (" "*6) + "N51"
    )
    print(f"Prepared report line: {MPC_report_line}")
    return MPC_report_line


##################################################################################################################
############################################# PARSE INPUT FROM USER ##############################################
##################################################################################################################


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--ast", help="asteroid, name of the target to be processed for astrometry"
)
parser.add_argument(
    "--date",
    default=date.today().strftime("%Y%m%d"),
    help="date, date of observations, default is the current date",
)
parser.add_argument(
    "--cam",
    default="andor",
    help="camera (andor/apogee), camera with which data is acquired, default is andor",
)
parser.add_argument(
    "--dir_path",
    default=os.path.abspath("."),
    help="dir_path, path to the date directory, default is home",
)


########### theta has the value of streak pa ############
args = parser.parse_args()
asteroid = args.ast
date = args.date
camera = args.cam
dir_path = args.dir_path

##################################################################################################################
############################################# PATH DEFINITIONS ###################################################
##################################################################################################################


abs_curpath = os.path.abspath(".")
print(abs_curpath)
ast_req = os.path.join(abs_curpath, "astrometry_req")
ast_conf = pd.read_csv(os.path.join(ast_req, "astrometry_conf.csv"))
targetinfo = pd.read_csv(os.path.join(ast_req, "target_info.csv"))
try:
    rockinfo = pd.read_csv(os.path.join(ast_req, asteroid + ".csv"))
except:
    print("Using the asteroid coordinates entered by the measurer")
print(dir_path, date, asteroid)
curpath = os.path.join(dir_path, date, asteroid)
fs = [f for f in os.listdir(curpath) if f.endswith("RA.fits")]
fspath = curpath


##################################################################################################################
####################################### HYPERPARAMETERS SPECIFIC TO CAMERA #######################################
##################################################################################################################

psf_1D_file = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["PSF_1D"])[0]
####### psf_1D_file a standard PSF? #######
grid_x = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["GRID_X"])[0]
grid_y = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["GRID_Y"])[0]
lev = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["LEVELS"])[0]
con = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["CONTRAST"])[0]
gain = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["GAIN"])[0]
fwhm = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["FWHM"])[0]
pix_scale = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["PIX_SCALE"])[0]
pix_pad = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["PIX_PADDED"])[0]
boxsize = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["BOX_SIZE"])[0]
maxmag = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["MAX_MAG"])[0]
thresholding_sigma = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["NSIGMA"])[0]
photoDistThresh = np.asarray(
    ast_conf[ast_conf["CAMERA"] == camera]["PHOTO_DIST_THRESH"]
)[0]
apr_radius = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["APR_RADIUS"])[0]
R_in = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["R_IN"])[0]
R_out = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["R_OUT"])[0]
psf_1D = fits.open(os.path.join(ast_req, psf_1D_file))[0].data
L = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["L_PIX_SCALE"])[0]
H = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["H_PIX_SCALE"])[0]
R = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["RADIUS"])[0]
psf_2D_init_size = np.asarray(
    ast_conf[ast_conf["CAMERA"] == camera]["PSF_2D_INIT_SIZE"]
)[0]


##################################################################################################################
###################################### PARAMETERS FOR ASTEROID OBSERVED ##########################################
##################################################################################################################
ns_files = []
for filename in fs:
    try:
        with fits.open(os.path.join(fspath, filename)) as hdul:
            header = hdul[0].header
            if header.get("OBS-TYPE", "") == "NS":
                ns_files.append(filename)
    except Exception as e:
        print(f"Error checking {filename}: {e}")

# Collect x_ast and y_ast for NS files only
coordinates = {}
print("\n--- Enter coordinates for NS images ---")
for filename in ns_files:
    while True:
        try:
            x = int(input(f"Enter x_ast for {filename}: "))
            y = int(input(f"Enter y_ast for {filename}: "))
            coordinates[filename] = (x, y)
            #ra_rate = float(input(f"Enter ra_rate for {filename}: "))
            #dec_rate = float(input(f"Enter dec_rate for {filename}: "))
            break
        except ValueError:
            print("Invalid input. Please enter integers.")


for filename in fs:
    file_path = os.path.join(fspath, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    try:
        with fits.open(file_path) as hdul:
            obs_type = hdul[0].header["OBS-TYPE"]
            if obs_type != "S":
                f = file_path
                break
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

if f is not None:
    print(f"Selected file: {f}", flush=True)
else:
    print("No file found with OBS-TYPE != 'S'")

f_header = fits.open(f)[0].header
exp_time = f_header["EXPTIME"] / 60  # minutes
print("exp time: ", exp_time)
RA = f_header["TARRA"]
DEC = f_header["TARDEC"]
print("DEC:", DEC)
dec = float(f_header["TARDEC"])
print("DEC:", dec)
ra_rate = float(f_header["RA-RATE"])
dec_rate = float(f_header["DEC-RATE"])
if ra_rate == 0 and dec_rate == 0:
    while True:
        try:
            ra_rate = float(input(f"Enter ra_rate for {filename}: "))
            dec_rate = float(input(f"Enter dec_rate for {filename}: "))
            break
        except ValueError:
            print("Invalid input. Please enter integers.")

#ra_rate = float(0.11)
#dec_rate = float(-0.91)
net_rate = np.sqrt(dec_rate**2 + (ra_rate * np.cos(np.radians(dec))) ** 2)
vel = net_rate * 60
print(f"ra_rate = {ra_rate}, dec_rate = {dec_rate} ")
print("#################################\n velocity = " + str(vel) + '"/min')
print(
    f"#################################\n pixel scale: {pix_scale} \n#################################", flush=True
)

dist_arcsec = vel * exp_time
dist = int((dist_arcsec) / pix_scale)  # pixel #streak length
print(f"streak length: {dist} \n#################################")

os.chdir(curpath)
fwhm, pa = generateFWHMNS(fs, curpath, dist)
# print("*****************The fwhm is:", fwhm, "*****************")

if fwhm < 1.5:
    fwhm = 1.5
elif fwhm > 10:
    fwhm = 10

if pa > 45 and pa < 135:
    fwhm_final = pix_scale * fwhm * (1.0 / np.sin((np.pi * pa) / 180.0))
else:
    fwhm_final = pix_scale * fwhm * (1.0 / np.cos((np.pi * pa) / 180.0))

sigma = fwhm / 2.3548
psf = Gaussian2DKernel(sigma)
hdul = fits.HDUList([fits.PrimaryHDU(np.arange(100))])
hdul[0].data = np.pad(psf, pad_width=(501 - psf.shape[0]) // 2)[:-1, :-1]
hdul.writeto("psf1D.fits", overwrite=True)
theta = pa
try:
    psf_1D = fits.open(os.path.join(curpath, "psf1D.fits"))[0].data
except:
    print("failed to load the PSF profile image generated from NS")
    psf_1D = fits.open(os.path.join(ast_req, psf_1D_file))[0].data
psf_2D_gen = generate_2D_psf(psf_1D, theta, int(dist), psf_2D_init_size, asteroid)
psf_2D_rot = ndimage.rotate(psf_2D_gen, 90 - theta, reshape=True)
psf_2D = psf_2D_rot / np.sum(psf_2D_rot)
psf_2D = np.array(psf_2D, dtype=float)


def make_odd_trim(array):
    rows, cols = array.shape
    if rows % 2 == 0:
        array = array[:-1, :]
    if cols % 2 == 0:
        array = array[:, :-1]
    return array


psf_2D_odd = make_odd_trim(psf_2D)
hdu = fits.PrimaryHDU(psf_2D_odd)
hdu.writeto(filename.replace('.fits', f'.psf_2D_odd.fits'), overwrite=True)


os.chdir(fspath)
rock_coordinates = pd.DataFrame(
    columns=[
        "ImageName",
        "ReportImg",
        "psfPath",
        "x_phy",
        "y_phy",
        "ra",
        "dec",
        "ZeroPoint",
        "FWHM",
    ]
)


# ##################################################################################################################
# ############################################## PROCESSING BEGINS #################################################
# ##################################################################################################################

for f in fs:
    obs_type = fits.open(f)[0].header["OBS-TYPE"]
    if obs_type != "S":
        print("Processing " + str(f) + " for astrometry.....")
        target = f
        name = f.split(".")[0]
        x_ast, y_ast = coordinates[f]
        filename = os.path.join(fspath, f)
        img_data = fits.open(os.path.join(fspath, f))[0].data
        img_header = fits.open(os.path.join(fspath, f))[0].header
        tbl, area, segm_deblend = get_sources(img_data, psf_2D_odd, dist, filename)
        ascii.write(
            tbl, target[:-3] + ".csv", format="csv", fast_writer=False, overwrite=True
        )
        path = os.path.join(fspath, target[:-3] + ".csv")
        syn_data = inject_sources(img_data, tbl)
        hdu = fits.PrimaryHDU(data=syn_data, header=img_header)
        image_name = f.replace(".fits", ".syn.fits")
        hdu.writeto(image_name, overwrite=True)
        print("Running Solve-Field!!!!!")
        command = (
            f"solve-field --ra {RA} --dec {DEC} --radius {R} -L {L} -H {H} -u arcsecperpix --scale-low {0.75*pix_scale} --scale-high {1.25*pix_scale} --no-background-subtraction --sigma 2 --odds-to-solve 1e8 --tweak-order 3 {image_name} --config /home/growth/sameer/astreaks/astrometry.cfg --overwrite --verbose"
        )
        print("Executing command: %s" % command)
        rval = os.system(command)
        print("Solved " + image_name + "for WCS.")
        # remove junk files
        new_files = os.listdir()
        print(new_files)
        try:
            [os.remove(f) for f in os.listdir() if f.endswith("match")]
            [os.remove(f) for f in os.listdir() if f.endswith("xyls")]
            [os.remove(f) for f in os.listdir() if f.endswith("rdls")]
            [os.remove(f) for f in os.listdir() if f.endswith("png")]
            [os.remove(f) for f in os.listdir() if f.endswith("axy")]
            [os.remove(f) for f in os.listdir() if f.endswith("corr")]
            [os.remove(f) for f in os.listdir() if f.endswith("png")]
            [os.remove(f) for f in os.listdir() if f.endswith("solved")]
            [os.remove(f) for f in os.listdir() if f.endswith("wcs")]
        except:
            continue

        img_data = fits.open(target)[0].data
        obs_type = fits.open(target)[0].header["OBS-TYPE"]
        print(image_name)
        img_wcs_header = fits.open(image_name.replace(".fits", ".new"))[0].header
        img_data_padded = np.pad(
            img_data, ((pix_pad, pix_pad), (pix_pad, pix_pad)), "constant"
        )
        mag, mag_err, zp, zp_err = get_ast_mag(
            img_data, tbl, img_wcs_header, x_ast, y_ast
        )
        img_wcs_header["FWHM"] = fwhm_final
        img_wcs_header["ZP"] = zp
        hdu = fits.PrimaryHDU(data=img_data_padded, header=img_wcs_header)
        hdu.writeto(name + ".proc.wcs.fits", overwrite=True)
        coordsfile = os.path.join(target[:-3] + ".csv")
        coordsdata = pd.read_csv(coordsfile)
        psfPath = os.path.join(curpath, target + ".psf.fits")
        if obs_type != "S":
            new_row = pd.DataFrame(
                {
                    "ImageName": [os.path.join(fspath, name + ".proc.wcs.fits")],
                    "ReportImg": [os.path.join(fspath, image_name)],
                    "psfPath": [os.path.join(curpath, "psf1D.fits")],
                    "x_phy": [x_ast],
                    "y_phy": [y_ast],
                    "ra": [0],
                    "dec": [0],
                    "ZeroPoint": [zp],
                    "FWHM": [fwhm_final],
                }
            )
            rock_coordinates = pd.concat([rock_coordinates, new_row], ignore_index=True)
        #coord_x, coord_y = get_ast_coords(img_data_padded, x_ast, y_ast, 30)
        #c = get_ra_dec(img_wcs_header, coord_x, coord_y)
        #MPC_report_line = get_MPC_report_line(img_wcs_header, asteroid, c, mag)

csv_path = os.path.join(curpath, f"rock_coords-{asteroid}.csv")

rock_coordinates.to_csv(csv_path, index=False)
print(f"Saved coordinates to: {csv_path}")


def run_astroid_utility(ast, cam, path):
    utility_script = os.path.join(abs_curpath, "astroid_coordinates_utility.py")
    subprocess.run(
        [
            sys.executable,
            utility_script,
            "--ast",
            ast,
            "--cam",
            cam,
            "--dir_path",
            path,
        ],
        check=True,
    )


run_astroid_utility(asteroid, camera, curpath)
