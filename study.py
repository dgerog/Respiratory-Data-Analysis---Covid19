from theStudy import *

# in file
FILE_INPUT = ['./data/mass_spectra_2020_11_11.xlsx','./data/mass_spectra_2020_09_10.xlsx','./data/mass_spectra_2020_09_11.xlsx','./data/mass_spectra_2020_11_11.xlsx',]
STORAGE_DIR = 'results/'

# which columns to use in order to extract the input vectors
COLS_TO_USE = "B:BK"

# which sheet to use
SHEETS_TO_USE = 1

# 
# Start parsing data 
#
study = theStudy()
for i in range(0, len(FILE_INPUT)):
    study.readTable(FILE_INPUT[i], COLS_TO_USE, SHEETS_TO_USE)
    groupNumb = study.kneeThresholding(_kernelCov=None, _oFName=STORAGE_DIR + 'knee-graph-' + str(i) + '.png')