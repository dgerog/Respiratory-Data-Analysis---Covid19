from theStudy import *

from sklearn.decomposition import KernelPCA
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
AGE_GROUPS_TO_SPLIT = 5

# percentage of points to use for training (rest is for testing)
TRAIN_PCT = .8

ITERS = 20
NEIGHBORS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

study = theStudy()

# 
# read the data
#
for i in range(0, len(FILE_INPUT)):
    study.readTable(_path=FILE_INPUT[i], _colsToRead=COLS_TO_USE[i], _sheetToRead=SHEETS_TO_USE, _doAppend=True, _doFilterData=True)

#
# Start the experiments
#
print ('|---------------------------------------------------------------------------------------|')
print ('|                          kNN (with PCA) Method - Manhattan                            |')
print ('|---------------------------------------------------------------------------------------|')
print ('|   Age Min | Age Max |  # of Records | K  | Precision | Recall | F1 Measure | Accuracy |')
print ('|---------------------------------------------------------------------------------------|')

dataX = study.flattenData(_appendThis='age')

transformer = KernelPCA(n_components=10, kernel='linear', degree=5, gamma=.1)

# Split age groups
ages = study.F[study.AGE_LINE,:].astype(int)
(ageH, ageB) = np.histogram(ages, bins=AGE_GROUPS_TO_SPLIT)
for ageGroup in range(0, len(ageB)-1):
    ageInd = np.where((ages>=ageB[ageGroup]) * (ages<ageB[ageGroup+1]))
    ageInd = ageInd[0]
    bestF1 = -1
    for n in NEIGHBORS:
        (R, P, F1, A) = (0, 0, 1, 0)
        classifier = KNeighborsClassifier(n_neighbors=n)
        for iter in range(0,ITERS):
            (tr, te) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=ageInd)
            # assign labels
            indPtr = np.where(study.isActive[tr] == True)  # patients' index
            indPtr = indPtr[0]
            Labels = np.zeros((len(tr)))
            Labels[indPtr] = 1
            #
            # Method 4: PCA + kNN
            #
            X_transformed = transformer.fit_transform(np.transpose(dataX[:,tr]))
            classifier.fit(X_transformed, Labels)

            # predict & analyze
            X_transformed = transformer.fit_transform(np.transpose(dataX[:,te]))
            Z = classifier.predict(X_transformed)
            (Pi, Ri, F1i, Ai) = study.classificationAnalysis(Z.astype(bool), te)

            if F1i < F1:
                P  = Pi
                R  = Ri
                F1 = F1i
                A  = Ai
        
        if F1 > bestF1:
            bestF1 = F1
            bestR  = R
            bestP  = P
            bestA  = A
            K      = n
    
    print ('|   %7.1f | %7.1f | %13d | %2d | %9.2f | %6.2f | %10.2f | %8.2f |' % (ageB[ageGroup], ageB[ageGroup+1], len(ageInd), K, bestP, bestR, bestF1, bestA))
print ('|---------------------------------------------------------------------------------------|')