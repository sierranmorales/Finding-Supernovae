import astropy.io.fits as fits
import astropy.wcs as wcs
import numpy as np
import os
import glob
import tqdm

# new match
#
# Takes 2 images
# Finds how many grid points from image 1 land in image 2:
# 	Run pix2world on image 1 and get ra dec (right ascension and declination)
# 	Run world to pix, to get from ra dec to pixels
# 	Figure out how many pixels overlap
#   Write to file if they ovelap sufficiently

f = open('OverlapLog.txt', 'w')

x = np.linspace(1, 1014, 50)
y = np.linspace(1, 1014, 50)
xv, yv = np.meshgrid(x, y)

xv = xv.flatten()
yv = yv.flatten()


files = glob.glob('/Users/sierra/Documents/Repo1/mastDownload/HST/*/*_flt.fits')


for i in tqdm.trange(len(files)):
  hdulist = fits.open(files[i])
  w = wcs.WCS(hdulist[('sci',1)].header)
  datei = hdulist[0].header['expstart']
  filteri = hdulist[0].header['filter']
  hdulist.close()

  radec = w.all_pix2world(np.transpose([xv, yv]), 1)

  for j in range(i+1, len(files)):
    hdulist = fits.open(files[j])
    w = wcs.WCS(hdulist[('sci',1)].header)
    hdulist.close()
    datej = hdulist[0].header['expstart']
    filterj = hdulist[0].header['filter']

    pix = w.all_world2pix(radec, 1)
    print(pix.shape)

    overlap = 0

    for k in pix:
      if np.all(k >= 1) and np.all(k <= 1014):
        overlap += 1

    if overlap > 100:
        f.write(" ".join([files[i], str(datei), str(filteri), files[j], str(datej), str(filterj),'\n']))
f.close()






  	
  	
