import numpy as np

#
#   Searches OverlapLog.txt for images that are suitable to be Drizzled together.
#


# File containing all images that have sufficient overlap
file = open('/Users/sierra/Documents/Repo1/Match 2/OverlapLog.txt', 'r')
lines = file.readlines()

visits = []

for k in range(len(lines)):
    found_visit = 0;

    # Extract various data components from source file
    filt1 = lines[k].split(" ")[2]
    filt2 = lines[k].split(" ")[5]
    date1 = float(lines[k].split(" ")[1])
    date2 = float(lines[k].split(" ")[4])
    image1 = lines[k].split(" ")[0]
    image2 = lines[k].split(" ")[3]

    splitLine = lines[k].split()
    print(splitLine)

    # If same filter and different dates
    if filt1 == filt2 and np.abs(date1 - date2) < 10:
    # Now we know that i and j should be paired
        for m in range(len(visits)):
            # If has been visited
            if image1 in visits[m] or image2 in visits[m]:
                visits[m].add(image1)
                visits[m].add(image2)
                found_visit = 1
        if found_visit == 0:
        # We didn't find a place to put image1 and image2
            visits.append(set([image1]))


# Set that stores all matching visits
visits_that_match = set()

for m in range(len(visits)):
    for n in range(m + 1, len(visits)):
        for line in lines:
            image1 = line.split(" ")[0]
            image2 = line.split(" ")[3]
            if (image1 in visits[m] and image2 in visits[n]) or (image1 in visits[n] and image2 in visits[m]):
                visits_that_match.add((m, n))
                
print(visits_that_match)














#################