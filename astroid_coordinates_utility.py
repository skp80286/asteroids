#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 1 13:23:18 2023
@author: Harsh Choudhary
"""
import pandas as pd
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
import image_registration
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ast", type = str, help="asteroid, name of the target to be processed for astrometry")
parser.add_argument("--cam", default = "andor", type = str, help = "camera (andor/apogee), camera with which data is acquired, default is andor")
parser.add_argument("--dir_path", default=os.path.abspath('.'), help="Directory to save outputs")
args = parser.parse_args()
asteroid = args.ast
camera = args.cam
curpath = args.dir_path 
# Add to argparse in the utility script:
 # Get the absolute path of the current directory
print(curpath)
filename = f"rock_coords-{asteroid}.csv"  # Construct the filename
file_path = os.path.join(curpath, filename)  # Join the path and filename

file_df = pd.read_csv(file_path)  # Read the CSV file
# coordPath = os.path.abspath('edge_detection\canny_edge_detector')
sciPath = os.path.abspath("..\Growth_Data")


# file_df = pd.read_csv("rock_coords-" + asteroid + ".csv")
# file_df = pd.read_csv(args.csv)

# psf_path = os.path.join(coordPath, 'psf2D_fin.fits')

def get_MPC_report(img_header, target, coord_ra, coord_dec, mag):
    """
    Function to get the MPC Report ile Line for an image.

    """
    if mag > 1e-6:
        source_mag_str = f'{mag[0]:02.2f}r'
    else:
        source_mag_str = ' '*6

    coord_ra_arr = coord_ra.split(" ")
    coord_dec_arr = coord_dec.split(" ")
    source_mag = mag
    exp_time = float(img_header["EXPTIME"])  # seconds
    print("Exposure time = " + str(exp_time))
    time = img_header["DATE-OBS"].split("T")[1].split(":")
    time_format = (
        int(time[0])
        + (int(time[1]) / 60)
        + ((float(time[2]) + (exp_time * 0.5)) / 3600)
    ) / 24
    date = img_header["DATE-OBS"].split("T")[0].split("-")
    UT_date = int(date[2]) + time_format
    date_MPC_format = date[0] + " " + date[1] + " " + str(round(UT_date, 6))
    print("MPC Report Line:")
    MPC_report_line = (
        "     "
        + target
        + "  C"
        + date_MPC_format
        + f'{int(coord_ra_arr[0]):02}'
        + " "
        + f'{int(coord_ra_arr[1]):02}'
        + " "
        + f'{float(coord_ra_arr[2]):06.3f}'
        + f'{int(coord_dec_arr[0]):+02}'
        + " "
        + f'{np.abs(int(coord_dec_arr[1])):02}'
        + " "
        + f'{np.abs(round(float(coord_dec_arr[2]), 1)):05.2f}'
        + " "*9
        + source_mag_str
        + " "*6 + "N51"
    )
    print(MPC_report_line)
    return MPC_report_line


#        MPC_report_file = open("MPC_report_file-"+target+".txt", 'a')
#        MPC_report_file.write(MPC_report_line + '\n')
#        MPC_report_file.close()


def psf_fit(data_cutout, psf_array):
    print(psf_array.shape)
    print(data_cutout.shape)
    # data_cutout = data_cutout[10: 50, 10: 50]
    # psf_array = psf_array[10: 50, 10: 50]
    psf_array = psf_array / np.sum(psf_array)
    xoff, yoff, exoff, eyoff = image_registration.chi2_shifts.chi2_shift(
        psf_array, data_cutout, 10, boundary="wrap", nfitted=2, upsample_factor="auto"
    )
    print(
        "Here are the x and y off values {} {} {} {}".format(xoff, yoff, exoff, eyoff)
    )
    # if abs(xoff) > 5.0 or abs(yoff) > 5.0:
    # xoff = 0.0
    # yoff = 0.0
    # phot_type = 'Forced_PSF'
    # else:
    # phot_type = 'PSF'
    resize_sci = image_registration.fft_tools.shift2d(data_cutout, -xoff, -yoff)
    resize_psf = psf_array
    flux = np.sum(resize_sci * resize_psf) / np.sum(resize_psf * resize_psf)
    return flux, xoff, yoff


# def do_psf_photometry(x, y, data, background, psfModelData):

    science_data = data
    psfModelData = psfModelData / np.sum(psfModelData)
    print(np.shape(psfModelData))
    print(np.sum(psfModelData))
    psf_data = psfModelData
    cutout = Cutout2D(science_data, position=(x, y), size=(29, 29))
    Is = cutout.data
    print(np.shape(Is))
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    try:
        bkg = Background2D(
            Is,
            np.shape(psf_data),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )
        print(np.median(bkg.background))
        background = np.median(bkg.background)
        print("Background estimated using bkg_estimator")
    except:
        print("Background estimated using sextractor ")
        background = background
    sigma_flux = np.sqrt(np.sum(Is * psf_data**2)) / np.sum(psf_data * psf_data)
    Is = Is - background
    flux, xoff, yoff = psf_fit(Is, psf_data)
    mag = -2.5 * np.log10(flux)
    psf_mag = mag
    mag_err = 2.5 * np.log10(1 + sigma_flux / flux)
    psf_mag_err = mag_err
    print(mag_err, psf_mag_err)
    print("PSF-fit magnitude of target is %.2f +/- %.2f" % (psf_mag, psf_mag_err))
    return (
        psf_mag,
        psf_mag_err,
        PSF_type,
        np.round(mag, 3),
        np.round(mag_err, 3),
        xoff,
        yoff,
    )

script_dir = os.path.dirname(os.path.abspath(__file__))

print("This is the correct script path:", script_dir)

# Construct the correct path to the file inside astrometry_req
ast_req = os.path.join(script_dir, "astrometry_req")
ast_conf = pd.read_csv(os.path.join(ast_req, "astrometry_conf.csv"))
pix_pad = np.asarray(ast_conf[ast_conf["CAMERA"] == camera]["PIX_PADDED"])[0]


for i in range(len(file_df)):
    name = file_df.iloc[i]["ImageName"]
    y_pix = file_df.iloc[i]["y_phy"]
    print(y_pix)
    x_pix = file_df.iloc[i]["x_phy"]
    print(x_pix)
    psfPath = file_df.iloc[i]["psfPath"]
    zeroPoint = file_df.iloc[i]["ZeroPoint"]
    reportName = file_df.iloc[i]['ReportImg']

    psf_data, psf_header = fits.getdata(psfPath, header=True)
    psf_cutout = psf_data[
        (psf_data.shape[0] // 2) - 10 : (psf_data.shape[0] // 2) + 10,
        (psf_data.shape[1] // 2) - 10 : (psf_data.shape[1] // 2) + 10,
    ]
    fits.writeto("psfCutout.fits", psf_cutout, psf_header, overwrite=True)
    sciPath = name
    data, header = fits.getdata(sciPath, header=True)
    # y_pix = data.shape[0]-y_pix
    print("y pixel:", y_pix)
    print("x pixel:", x_pix)
    fits.writeto(
        str(i) + "rockCutout.fits",
        data[y_pix - 200 : y_pix + 200, x_pix - 200 : x_pix + 200],
        header,
        overwrite=True,
    )
    flux, x_off, y_off = psf_fit(
        data[y_pix - 10 : y_pix + 10, x_pix - 10 : x_pix + 10], psf_cutout
    )
    # psf_mag, psf_mag_err, PSF_type, mag_round_3, mag_err_round_3, x_off, y_off = do_psf_photometry(x_pix, y_pix, data[y_pix-10:y_pix+10, x_pix-10:x_pix+10], 0, psf_cutout)
    # print("Look at this PSF:", psf_mag)
    wcs = WCS(header)
    ra_, dec_ = wcs.all_pix2world([x_pix + x_off+pix_pad], [y_pix + y_off+pix_pad], 0)
    c = SkyCoord(ra=ra_ * u.degree, dec=dec_ * u.degree, frame="icrs")
    # ra_ = x_pix+x_off
    # dec_ = y_pix+y_off
    coords_list = c.to_string("hmsdms")
    coords = coords_list[0].split(" ")
    coords[0] = coords[0].replace("h", " ").replace("m", " ").replace("s", "")
    coords[1] = (
        coords[1].replace("d", " ").replace("m", " ").replace("s", "").replace("+", "")
    )
    # cords_arr = coords[0].split(" ")
    # print(cords_arr[2])
    file_df.loc[i, "ra"] = str(coords[0])
    file_df.loc[i, "dec"] = str(coords[1])
    
    # ra_str = f"{h:02d} {m:02d} {s:05.2f}"
    # dec_str = f"{abs(d):02d} {m:02d} {s:05.2f}"
    # file_df.loc[i, "ra"] = ra_str
    # file_df.loc[i, "dec"] = dec_str
    
    img_wcs_header = fits.open(reportName.replace(".fits", ".new"))[0].header
    MPC_report_line = get_MPC_report(img_wcs_header, asteroid, coords[0], coords[1], 0)
    MPC_report_file = open(os.path.join(curpath, "MPC_report_file-"+asteroid+".txt"), 'a')
    MPC_report_file.write(MPC_report_line + '\n')
    MPC_report_file.close()
    print(file_df.head(5))

# print(file_df.head(5))
file_df.to_csv("rock_coords-" + asteroid + ".csv", index=False)
