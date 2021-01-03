#
# COVID Data Analysis
#
# Classification of respiratory data trying to predict:
#   i) PATIENT vs. NOT PATIENT
#  ii) ACTIVE vs. NOT ACTIVE
#
# Classification methods tested:
#   i) SVM
#  ii) GMM
# iii) DECISSION TREES

#import basic libs
import matplotlib.pyplot as plt

# import this study
from theStudy import *

# input files
FILE_INPUT = [
'./data/mass_spectra_2020_08_26.xlsx',
'./data/mass_spectra_2020_08_27.xlsx',
'./data/mass_spectra_2020_09_09.xlsx',
'./data/mass_spectra_2020_09_10.xlsx',
'./data/mass_spectra_2020_09_11.xlsx',
'./data/mass_spectra_2020_09_15.xlsx',
'./data/mass_spectra_2020_11_09.xlsx',
'./data/mass_spectra_2020_11_10.xlsx',
'./data/mass_spectra_2020_08_29.xlsx',
'./data/mass_spectra_2020_08_30.xlsx',
'./data/mass_spectra_2020_08_31.xlsx',
'./data/mass_spectra_2020_09_01.xlsx',
'./data/mass_spectra_2020_09_02.xlsx',
'./data/mass_spectra_2020_09_04.xlsx',
'./data/mass_spectra_2020_09_05.xlsx',
'./data/mass_spectra_2020_09_06.xlsx',
'./data/mass_spectra_2020_09_07.xlsx',
'./data/mass_spectra_2020_09_08.xlsx',
'./data/mass_spectra_2020_09_12.xlsx',
'./data/mass_spectra_2020_09_13.xlsx',
'./data/mass_spectra_2020_09_14.xlsx',
'./data/mass_spectra_2020_11_11.xlsx',
'./data/mass_spectra_2020_11_09.xlsx',
'./data/mass_spectra_2020_11_10.xlsx',
]

# which columns to use in order to extract the input vectors
COLS_TO_USE = [
    "B:AB",
    "B:AK",
    "B:BJ",
    "B:CO",
    "B:BK",
    "B:K",
    "B:BE",
    "B:BN",
    "B:N",
    "B:H",
    "B:BF",
    "B:BB",
    "B:BA",
    "B:CE",
    "B:V",
    "B:T",
    "B:N",
    "B:CV",
    "B:T",
    "B:R",
    "B:DW",
    "B:BK",
    "B:BE",
    "B:BN"
]

# which sheet to use
SHEETS_TO_USE = 1

# where to store data
STORAGE_DIR = 'results/'

# how many age groups to split - define the group's extreme points
AGE_GROUPS = [0,20,40,65,100]

# percentage of points to use for training (rest is for testing)
TRAIN_PCT = .90

# iterations of the experiments (decrease the chance of getting results becaise of randomness)
ITERS = 20 

# how to handle imbalanced classes
DO_STABILIZE = True

#
# ALL SET -> We can start the experiments
#

# read the data
study = theStudy()
for i in range(0, len(FILE_INPUT)):
    study.readTable(
        # where and what to read
        _path=FILE_INPUT[i], _colsToRead=COLS_TO_USE[i], _sheetToRead=SHEETS_TO_USE, 
        # join all files in one dataset
        _doAppend=True, 
        # extra processing tasks
        _doFilterData=True, _doNormalize=True
    )