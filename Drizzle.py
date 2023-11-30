import pickle
import random
import numpy as np
from drizzlepac import astrodrizzle
from drizzlepac import tweakreg
from tqdm import tqdm
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
from astropy.wcs import WCS

#
# Uses tweakreg to align images and astrodrizzle to subtract images
#


# Function to get the right ascension and declination from the image headers
def get_radec(filenames):
    ra, dec = [], []

    for file in filenames:
        with fits.open(file) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            ra.append(wcs.wcs.crval[0])
            dec.append(wcs.wcs.crval[1])

    return ra, dec

# Function to find the image closest to the mean right ascension and declination
def find_closest_image(filenames, mean_ra, mean_dec):
    closest_image = None
    min_distance = float('inf')

    for file in filenames:
        with fits.open(file) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            distance = np.sqrt((wcs.wcs.crval[0] - mean_ra)**2 + (wcs.wcs.crval[1] - mean_dec)**2)
            if distance < min_distance:
                min_distance = distance
                closest_image = file

    return closest_image



# Function to simulate supernovae and save the modified image
def simulate_and_save_supernovae(image_filename, output_filename, psf_interpolator, supernovae_coords, flux):
    with fits.open(image_filename) as hdul:
        image_data = hdul[1].data
        wcs = WCS(hdul[1].header)

        for i, (ra, dec) in enumerate(supernovae_coords):
            x_center, y_center = wcs.all_world2pix(ra, dec, 0)

            if 0 <= x_center < image_data.shape[1] and 0 <= y_center < image_data.shape[0]:
                # Create a meshgrid for the PSF interpolation
                x = np.arange(image_data.shape[1])
                y = np.arange(image_data.shape[0])
                X, Y = np.meshgrid(x, y)

                # Adjust the PSF position to be centered at (x_center, y_center)
                supernova_psf = psf_interpolator(X - x_center, Y - y_center, grid=False) * flux[i]

                # Add the PSF to the image data
                image_data += supernova_psf

        hdul[0].header.add_history('Simulated supernovae added.')
        hdul.writeto(output_filename, overwrite=True)



def generate_random_coords(num_coords, wcs_info):
    ra_dec_coords = []
    for _ in range(num_coords):
        # Generate random pixel coordinates
        random_x = random.uniform(0, wcs_info.pixel_shape[0])
        random_y = random.uniform(0, wcs_info.pixel_shape[1])
        # Convert pixel coordinates to RA/Dec
        ra, dec = wcs_info.all_pix2world(random_x, random_y, 0)
        ra_dec_coords.append((ra, dec))
    return ra_dec_coords



# Load the PSF standard data for WFC3/IR
with fits.open("PSFSTD_WFC3IR_F125W.fits") as f:
    psf_data = f[0].data

# Prepare the subsampled grid and the interpolator
xy_subsampled = np.arange(101, dtype=np.float64) * 0.25
xy_subsampled -= np.median(xy_subsampled)
psf_interpolator = RectBivariateSpline(xy_subsampled, xy_subsampled, psf_data[4], kx=3, ky=3)

# Load the visit data
with open("data.pkl", "rb") as file:
    visits, visits_that_match = pickle.load(file)

# Define the amount of supernovae to be implanted
num_supernovae = 100

# Generates magnitude and flux values
mag = np.random.random(size=num_supernovae) * 2 + 24


# Write supernova mag and coords into file specific for each visit
#
#

# Create an array of the same flux value, repeated for each supernova
flux = np.full(num_supernovae, 10**(-0.4 * (mag - 26.232)))


# Main loop to process each visit
for match in tqdm(visits_that_match, desc="Processing matches", unit="match"):
    # Original image filenames
    input_flcs_orig = [f"{image}" for image in visits[match[0]].union(visits[match[1]])]


    # Process the original images with tweakreg and astrodrizzle
    # Get the mean RA and Dec for the reference image
    ra, dec = get_radec(input_flcs_orig)
    mean_ra, mean_dec = np.mean(ra), np.mean(dec)

    # Find the closest image to use as the reference
    refimage_orig = find_closest_image(input_flcs_orig, mean_ra, mean_dec)

    tweakreg.TweakReg(
        input_flcs_orig, 
        updatewcs=True,
        interactive=False,
        updatehdr=True,
        residplot='No plot',
        reusename=True,
        clean=True)

    # Generate random supernova coordinates for this set of images
    with fits.open(input_flcs_orig[0]) as hdul:
        wcs_info = WCS(hdul[1].header)
        supernovae_coords = generate_random_coords(num_supernovae, wcs_info)


    # Simulate supernovae for each original image and save to new files
    input_flcs_sn = []
    for flt_image in input_flcs_orig:
        sn_image = flt_image.replace('.fits', '_s.fits')
        simulate_and_save_supernovae(flt_image, sn_image, psf_interpolator, supernovae_coords, flux)
        input_flcs_sn.append(sn_image)

    astrodrizzle.AstroDrizzle(
        input_flcs_orig, 
        output=f"visit_{match[0]}_{match[1]}_orig", 
        preserve=False,
        driz_sep_bits='64, 512, 8192',
        driz_cr_corr=True,
        final_bits='64, 512, 8192',
        clean=False,
        configobj=None,
        build=True,
        final_refimage=refimage_orig)  # Use the reference image for the original set

    # Process the supernovae-added images with tweakreg and astrodrizzle
    # Since we have already calculated the mean RA and Dec, we can reuse those values
    refimage_sn = find_closest_image(input_flcs_sn, mean_ra, mean_dec)
    
    tweakreg.TweakReg(
        input_flcs_sn,
        updatewcs=True,
        interactive=False,
        updatehdr=True,
        residplot='No plot',
        reusename=True,
        clean=True
    )

    astrodrizzle.AstroDrizzle(
        input_flcs_sn,
        output=f"visit_{match[0]}_{match[1]}_sn",
        preserve=False,
        driz_sep_bits='64, 512, 8192',
        driz_cr_corr=True,
        final_bits='64, 512, 8192',
        clean=False,
        configobj=None,
        build=True,
        final_refimage=refimage_sn  # Use the reference image for the supernovae-added set
    )


