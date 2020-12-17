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
AGE_GROUPS = [0,20,40,65,100]

# percentage of points to use for training (rest is for testing)
TRAIN_PCT = .70

ITERS = 100
GROUPS_K = [5,7,10,12]

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
print ('|                                 PDF Fitting                                      |')
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
    for n in GROUPS_K:
        if n >= .05*len(ageInd):
            continue

        gmmP = mixture.GaussianMixture(n_components=2)
        gmmH = mixture.GaussianMixture(n_components=n)
        for iter in range(0,ITERS):
            # split initial training in training and testing
            (tr, te) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=trALL)

            # get two groups (Patients & Healthy) - TRAINING
            indPtr = np.where(study.isActive[tr] == True)  # patients' index
            indPtr = indPtr[0]
            indHtr = np.where(study.isPatient[tr] == False) # healthy index
            indHtr = indHtr[0]
        
            #
            # Method 1: GMM (fit pdf)
            #

            # Learn patients patterns
            gmmP.fit(np.transpose(dataX[:,indPtr]))

            # Learn healthy patterns
            gmmH.fit(np.transpose(dataX[:,indHtr]))

            # predict & analyze
            p1 = gmmP.score_samples(np.transpose(dataX[:,te])) # test against patients' model
            p2 = gmmH.score_samples(np.transpose(dataX[:,te])) # test against healthy model
            Z = p1 > p2 # get the computed label (if True -> Patient)
            
            (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), te)
            if F1 > bestF1:
                bestF1  = F1
                bestN   = n
                bestInd = tr
    
    # get the best trained model
    indPtr = np.where(study.isActive[bestInd] == True)  # patients' index
    indPtr = indPtr[0]
    indHtr = np.where(study.isPatient[bestInd] == False) # healthy index
    indHtr = indHtr[0]

    gmmP = mixture.GaussianMixture(n_components=2)
    gmmH = mixture.GaussianMixture(n_components=bestN)
    # Learn patients patterns
    gmmP.fit(np.transpose(dataX[:,indPtr]))

    # Learn healthy patterns
    gmmH.fit(np.transpose(dataX[:,indHtr]))

    # validate model
    p1 = gmmP.score_samples(np.transpose(dataX[:,teALL])) # test against patients' model
    p2 = gmmH.score_samples(np.transpose(dataX[:,teALL])) # test against healthy model
    Z = p1 > p2 # get the computed label (if True -> Patient)
    (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), teALL)
    
    print ('|   %7.1f | %7.1f | %13d | %9.2f | %6.2f | %10.2f | %8.2f |' % (AGE_GROUPS[ageGroup], AGE_GROUPS[ageGroup+1], len(ageInd), P, R, F1, A))
print ('|----------------------------------------------------------------------------------|')