from theStudy import *

from sklearn import mixture

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
    "B:BK"
]

# which sheet to use
SHEETS_TO_USE = 1

# how many age groups to split
AGE_GROUPS = [0,20,30,40,50,60,100]

# percentage of points to use for training (rest is for testing)
TRAIN_PCT = .95

ITERS = 10

K = 10

study = theStudy()

# 
# read the data
#
for i in range(0, len(FILE_INPUT)):
    study.readTable(_path=FILE_INPUT[i], _colsToRead=COLS_TO_USE[i], _sheetToRead=SHEETS_TO_USE, 
    _doAppend=True, _doFilterData=True)

#
# Start the experiments
#
print ('|---------------------------------------------------------------------------------------|')
print ('|                                       GMM Method                                      |')
print ('|---------------------------------------------------------------------------------------|')
print ('|   Age Min | Age Max |  # of Records | K  | Precision | Recall | F1 Measure | Accuracy |')
print ('|---------------------------------------------------------------------------------------|')

dataX = study.flattenData(_appendThis=None)

# Split age groups
ages = study.F[study.AGE_LINE,:].astype(int);
(ageH, ageB) = np.histogram(ages, bins=AGE_GROUPS)
for ageGroup in range(0, len(ageB)-1):
    ageInd = np.where((ages>=ageB[ageGroup]) * (ages<ageB[ageGroup+1]))
    ageInd = ageInd[0]
    (R, P, F1, A) = (0, 0, 1, 0)
    for iter in range(0,ITERS):
        (tr, te) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=ageInd)
        # get two groups (Patients & Healthy) - TRAINING
        indPtr = np.where(study.isActive[tr] == True)  # patients' index
        indPtr = indPtr[0]
        indHtr = np.where(study.isPatient[tr] == False) # healthy index
        indHtr = indHtr[0]
        
        #
        # Method 1: GMM
        #

        # Learn patients patterns
        gmmP = mixture.GaussianMixture(n_components=K)
        gmmP.fit(np.transpose(dataX[:,indPtr]))

        # Learn healthy patterns
        gmmH = mixture.GaussianMixture(n_components=K)
        gmmH.fit(np.transpose(dataX[:,indHtr]))

        # analysis on healthy patterns
        p1 = gmmP.score_samples(np.transpose(dataX[:,te])) # test against patients' model
        p2 = gmmH.score_samples(np.transpose(dataX[:,te])) # test against healthy model
        Z = p1 > p2 # get the computed label (if True -> Patient)

        (Pi, Ri, F1i, Ai) = study.classificationAnalysis(Z, te)

        if F1i < F1:
            P  = Pi
            R  = Ri
            F1 = F1i
            A  = Ai
    
    print ('|   %7.1f | %7.1f | %13d | %2d | %9.2f | %6.2f | %10.2f | %8.2f |' % (ageB[ageGroup], ageB[ageGroup+1], len(ageInd), K, P, R, F1, A))
print ('|---------------------------------------------------------------------------------------|')