from theStudy import *

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# in file
FILE_INPUT = [
'./data/mass_spectra_2020_08_26.xlsx',
'./data/mass_spectra_2020_08_27.xlsx',
'./data/mass_spectra_2020_09_09.xlsx',
'./data/mass_spectra_2020_09_10.xlsx',
'./data/mass_spectra_2020_09_11.xlsx',
'./data/mass_spectra_2020_09_15.xlsx',
'./data/mass_spectra_2020_11_09.xlsx',
'./data/mass_spectra_2020_11_10.xlsx',
]
STORAGE_DIR = 'results/'

# which columns to use in order to extract the input vectors
COLS_TO_USE = [
    "B:AB",
    "B:AK",
    "B:BJ",
    "B:CO",
    "B:BK",
    "B:K",
    "B:BE",
    "B:BN"
]

# which sheet to use
SHEETS_TO_USE = 1

# how many age groups to split
AGE_GROUPS_TO_SPLIT = 4

ITERS = 10

study = theStudy()

# 
# read the data
#
for i in range(0, len(FILE_INPUT)):
    study.readTable(_path=FILE_INPUT[i], _colsToRead=COLS_TO_USE[i], _sheetToRead=SHEETS_TO_USE, _doAppend=True, _doFilterData=True)

#
# Start the experiments
#

dataX = study.flattenData(_appendThis='age')

# Split age groups
ages = study.F[study.AGE_LINE,:].astype(int);
(ageH, ageB) = np.histogram(ages, bins=AGE_GROUPS_TO_SPLIT)
for ageGroup in range(0, len(ageB)-1):
ageInd = np.where((ages>=ageB[ageGroup]) * (ages<ageB[ageGroup+1]))
ageInd = ageInd[0]
(R, P, F1) = (0, 0, 0)
for iter in range(0,ITERS):
    (tr, te) = study.prepareCrossValidation(_trainPct=.7, _allInd=ageInd)
    # assign labels
    indPtr = np.where(study.isActive[tr] == True)  # patients' index
    indPtr = indPtr[0]
    Labels = np.zeros((len(tr)))
    Labels[indPtr] = 1
    #
    # Method 2: RBF
    #
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(np.transpose(dataX[:,tr]), Labels)
    # predict & analyze
    Z = gpc.predict(np.transpose(dataX[:,te]))
    (Pi, Ri, F1i) = study.classificationAnalysis(Z.astype(bool), te)

        P = P + Pi
        R = R + Ri
        F1 = F1 + F1i

    print ('|-----------------------------------------|')
    print ('|              RBF Method                 |')
    print ('|-----------------------------------------|')
    print ('| AGE GROUP: %.1f - %.1f                  |' % (ageB[ageGroup], ageB[ageGroup+1]))
    print ('|-----------------------------------------|')
    print ('|      Precision | Recall | F1 Measure    |')
    print ('|        %.2f    |  %.2f  |     %.2f      |' % (P/ITERS, R/ITERS, F1/ITERS))
    print ('|-----------------------------------------|')