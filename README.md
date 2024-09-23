# 11102-NOVA
Searching the Hubble Space Telescope Archive for Supernovae
Funded by UROP Fall 2023


This project aims to analyze images from the Hubble Space Telescope archive and, using a convelutional neural network, identify supernovae. This project creates and utilizes simulated data to train the model training.

Steps to run project:
- Download CSV file from HST archive containing all files fitting your parameters.
- Run DataFinder.py to compile a MatchLog.txt file containing all compatible images (different time, same location, same filter).
- Run DownloadData.py to download all images from the .txt file.
- Run DrizzleMatch.py to determine which images have sufficient overlap and write to OverlapLog.txt
- Run Visits.py to search OverlapLog.txt for images that are suitable to be Drizzled together and compile them in a data.pkl file.
- Run Drizzle5.py to align and subtract the images using TweakReg and AstroDrizzle. This step also adds simulated supernovae into the training data.
- Run NeuralNet.py to finalize the data and train the model.
