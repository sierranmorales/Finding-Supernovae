import glob
from drizzlepac import astrodrizzle 
from drizzlepac import tweakreg

#
#   Uses astrodrizzle to subtract images
#


input_flcs01 = open('/Users/sierra/Documents/Repo1/Match 2/OverlapLog.txt', 'r')
lines = input_flcs01.readlines()

print(lines)


for i in range(len(lines)):
    splitLine = lines[i].split()
    print(splitLine)


inputList = input_flcs01 + input_flcs02


tweakreg.TweakReg(inputList,
    updatewcs = True,
    interactive = False,
    updatehdr = True)

astrodrizzle.AstroDrizzle(input_flcs01,
    output='01',
    preserve=False,
    driz_sep_bits='64, 512, 8192',
    driz_cr_corr=True,
    final_bits='64, 512, 8192',
    clean=False,
    configobj=None,
    build=True,
    )

astrodrizzle.AstroDrizzle(input_flcs02,
    output='02',
    preserve=False,
    driz_sep_bits='64, 512, 8192',
    driz_cr_corr=True,
    final_bits='64, 512, 8192',
    clean=False,
    configobj=None,
    build=True,
    #
    final_refimage = '01_drz.fits'
    #
    )