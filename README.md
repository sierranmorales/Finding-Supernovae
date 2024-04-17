# 11102-NOVA
Searching the Hubble Space Telescope Archive for Supernovae



This project aims to analyze all images in the Hubble Space Telescope archive and, using a convelutional neural network, identify supernovae.

Steps to run project:
- Download CSV file from HST archive containing all files fitting your parameters.
- Run DataFinder.py to compile a MatchLog.txt file containing all compatible images (different time, same location, same filter)
- Run DownloadData.py to download all images from the .txt file.
- Run DrizzleMatch.py to determine which images have sufficient overlap and write to OverlapLog.txt
- Run Visits.py to search OverlapLog.txt for images that are suitable to be Drizzled together and compile them in a data.pkl file
- Run Drizzle.py to align and subtract the images using TweakReg and AstroDrizzle
- Run NeuralNet.py to finalize the data and train the model

