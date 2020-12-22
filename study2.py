from theStudy import *

from sklearn import svm

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

# how many age groups to split
AGE_GROUPS = [0,20,40,65,100]

# percentage of points to use for training (rest is for testing)
TRAIN_PCT = .70

ITERS = 20
NEIGHBORS = [1,2,3,4,5]

study = theStudy()

# 
# read the data
#
for i in range(0, len(FILE_INPUT)):
    study.readTable(_path=FILE_INPUT[i], _colsToRead=COLS_TO_USE[i], _sheetToRead=SHEETS_TO_USE, 
    _doAppend=True, 
    _doFilterData=True, _doNormalize=True)

#
# Start the experiments
#
print ('|----------------------------------------------------------------------------------|')
print ('|                                       SVM                                        |')
print ('|----------------------------------------------------------------------------------|')
print ('|   Age Min | Age Max |  # of Records | Precision | Recall | F1 Measure | Accuracy |')
print ('|----------------------------------------------------------------------------------|')

# initialize classifier
classifier = svm.SVC()

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
    for iter in range(0,ITERS):
        # split initial training in training and testing
        (tr, te) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=trALL)

        # assign labels
        indPtr = np.where(study.isActive[tr] == True)  # patients' index
        indPtr = indPtr[0]
        Labels = np.zeros((len(tr)))
        Labels[indPtr] = 1

        #
        # Method 3: kNN
        #
        classifier.fit(np.transpose(dataX[:,tr]), Labels)

        # predict & analyze
        Z = classifier.predict(np.transpose(dataX[:,te]))
        (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), te)
        
        if F1 > bestF1:
            bestF1  = F1
            bestInd = tr

            if ageGroup == len(AGE_GROUPS)-2:
                #keep the model for the oldest age group
                oldestInd = tr
    
    # get the best trained model
    indPtr = np.where(study.isActive[bestInd] == True)  # patients' index
    indPtr = indPtr[0]
    Labels = np.zeros((len(bestInd)))
    Labels[indPtr] = 1
    classifier.fit(np.transpose(dataX[:,bestInd]), Labels)

    # validate model
    Z = classifier.predict(np.transpose(dataX[:,teALL]))
    (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), teALL)
    
    print ('|   %7.1f | %7.1f | %13d | %9.2f | %6.2f | %10.2f | %8.2f |' % (AGE_GROUPS[ageGroup], AGE_GROUPS[ageGroup+1], len(ageInd), P, R, F1, A))
print ('|----------------------------------------------------------------------------------|')


#
# Check the model fit on oldest group with other age groups
#
indPtr = np.where(study.isActive[oldestInd] == True)  # patients' index
indPtr = indPtr[0]
Labels = np.zeros((len(oldestInd)))
Labels[indPtr] = 1
classifier.fit(np.transpose(dataX[:,oldestInd]), Labels)
for ageGroup in range(0, len(AGE_GROUPS)-1):
    # find records in this age group
    ageInd = np.where((ages>=AGE_GROUPS[ageGroup]) * (ages<AGE_GROUPS[ageGroup+1]))
    ageInd = ageInd[0]

    # validate model
    Z = classifier.predict(np.transpose(dataX[:,ageInd]))
    (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), ageInd)
    print ('|   %7.1f | %7.1f | %13d | %9.2f | %6.2f | %10.2f | %8.2f |' % (AGE_GROUPS[ageGroup], AGE_GROUPS[ageGroup+1], len(ageInd), P, R, F1, A))
print ('|----------------------------------------------------------------------------------|')