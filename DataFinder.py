import numpy as np
import csv
import tqdm

#
# Reads in CSV file containing data on all images in the HST (Hubble Space Telescope) database fitting our filters
# Checks each image's data against each other's to check if compatible.
# 

# Checks if positioning is compatible
def get_ang(ra1, dec1, ra2, dec2):
  r1 = np.array([np.cos(ra1*np.pi/180.)*np.cos(dec1*np.pi/180.), np.sin(ra1*np.pi/180.)*np.cos(dec1*np.pi/180.), np.sin(dec1*np.pi/180.)], dtype=np.float64)
  r2 = np.array([np.cos(ra2*np.pi/180.)*np.cos(dec2*np.pi/180.), np.sin(ra2*np.pi/180.)*np.cos(dec2*np.pi/180.), np.sin(dec2*np.pi/180.)], dtype=np.float64)
  ang = np.arccos(np.dot(r1, r2))*180./np.pi
  return ang

  
f = open("CSVFile2.csv", 'r')
lines = f.read().split('\n')
print(lines)
f.close()


for i in tqdm.trange(1, len(lines)-1):
  for j in range(i+1, len(lines)-1):

    name1 = lines[i].split(",")[0]
    name2 = lines[j].split(",")[0]
    ra1 = float(lines[i].split(",")[1])
    ra2 = float(lines[j].split(",")[1]) 
    dec1 = float(lines[i].split(",")[2]) 
    dec2 = float(lines[j].split(",")[2]) 
    filter1 = lines[i].split(",")[6]
    filter2 = lines[j].split(",")[6]
    posAngle1 = float(lines[i].split(",")[7])
    posAngle2 = float(lines[j].split(",")[7])

    # If the filters match
    if filter1 == filter2:
      if posAngle1 >= posAngle2 + 1 or posAngle1 <= posAngle2 - 1:
        # If declination is close enough
        if dec1 and dec2 < 0.05: 
          ang = get_ang(ra1, dec1, ra2, dec2)
          # If right ascension is close enough
          if ang < 0.05:
            # Then write this pair to file
            with open('MatchLog.txt', 'a') as f2:
              f2.write(" ".join([name1, str(ra1), str(dec1), name2, str(ra2), str(dec2), '\n']))
            f2.close()

            print('match\n')

