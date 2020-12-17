from theStudy import *

from sklearn.neighbors import KNeighborsClassifier

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
AGE_GROUPS = [0,20,40,65,100]

# percentage of points to use for training (rest is for testing)
TRAIN_PCT = .70

ITERS = 100
NEIGHBORS = [1,2,3,4,5]

study = theStudy()

# 
# read the data
#
for i in range(0, len(FILE_INPUT)):
    study.readTable(_path=FILE_INPUT[i], _colsToRead=COLS_TO_USE[i], _sheetToRead=SHEETS_TO_USE, 
    _doAppend=True, 
    _doFilterData=False, _doNormalize=True)

#
# Start the experiments
#
print ('|----------------------------------------------------------------------------------|')
print ('|                         kNN Method (Weighted Minkowski)                          |')
print ('|----------------------------------------------------------------------------------|')
print ('|   Age Min | Age Max |  # of Records | Precision | Recall | F1 Measure | Accuracy |')
print ('|----------------------------------------------------------------------------------|')

dataX = study.flattenData(_appendThis=None)

# Split age groups
ages = study.F[study.AGE_LINE,:].astype(int)
for ageGroup in range(0, len(AGE_GROUPS)-1):
    # find records in this age group
    ageInd = np.where((ages>=AGE_GROUPS[ageGroup]) * (ages<AGE_GROUPS[ageGroup+1]))
    ageInd = ageInd[0]

    # split in trainig and validation
    (trALL, teALL) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=ageInd)

    # start training (exhaustive search)
    bestF1 = -1
    for n in NEIGHBORS:
        if n >= .05*len(ageInd):
            continue
        
        for iter in range(0,ITERS):
            # split initial training in training and testing
            (tr, te) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=trALL)

            # get weights
            [v, e] = np.linalg.eig(np.cov(dataX[:,tr]))
            v = np.abs(v)
            classifier = KNeighborsClassifier(n_neighbors=n, 
                metric='wminkowski',
                p=1,
                metric_params= {
                    "w": v/(np.sum(v) + .0000000000000000001)
                }
            )

            # assign labels
            indPtr = np.where(study.isActive[tr] == True)  # patients' index
            indPtr = indPtr[0]
            Labels = np.zeros((len(tr)))
            Labels[indPtr] = 1

            #
            # Method 5: kNN
            #
            classifier.fit(np.transpose(dataX[:,tr]), Labels)

            # predict & analyze
            Z = classifier.predict(np.transpose(dataX[:,te]))
            (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), te)
            
            if F1 > bestF1:
                bestClassifier = classifier
    
    # validate model
    Z = bestClassifier.predict(np.transpose(dataX[:,teALL]))
    (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), teALL)
    
    print ('|   %7.1f | %7.1f | %13d | %9.2f | %6.2f | %10.2f | %8.2f |' % (AGE_GROUPS[ageGroup], AGE_GROUPS[ageGroup+1], len(ageInd), P, R, F1, A))
print ('|----------------------------------------------------------------------------------|')