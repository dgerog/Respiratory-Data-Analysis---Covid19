from theStudy import *

from sklearn import mixture

# in file
FILE_INPUT = ['./data/mass_spectra_2020_11_11.xlsx',
'./data/mass_spectra_2020_09_10.xlsx',
'./data/mass_spectra_2020_09_11.xlsx',
'./data/mass_spectra_2020_09_09.xlsx',
]
STORAGE_DIR = 'results/'

# which columns to use in order to extract the input vectors
COLS_TO_USE = [
    "B:BK",
    "B:CO",
    "B:BK",
    "B:BJ",
]

# which sheet to use
SHEETS_TO_USE = 1

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
P = 0
R = 0
F1 = 0
for iter in range(0, ITERS):
    (tr, te) = study.prepareCrossValidation()
    
    # get two groups (Patients & Healthy)
    indPtr = np.where(study.isPatient[tr] == True)  # patients' index
    indPtr = indPtr[0]
    indHtr = np.where(study.isPatient[tr] == False) # healthy index
    indHtr = indHtr[0]
    
    # estimate number of clusters (training)
    kP = study.kneeThresholding(_X=study.X[indPtr], _oFName=STORAGE_DIR + 'G_TR_Patients-' + str(i) + '.eps')
    kH = study.kneeThresholding(_X=study.X[indHtr], _oFName=STORAGE_DIR + 'G_TR_Healthy-' + str(i) + '.eps')

    #
    # Method 1: GMM
    #

    # Learn patients patterns
    gmmP = mixture.GaussianMixture(n_components=kP)
    gmmP.fit(np.transpose(study.X[:,indPtr]))

    # Learn healthy patterns
    gmmH = mixture.GaussianMixture(n_components=kH)
    gmmH.fit(np.transpose(study.X[:,indHtr]))
    
    # analysis on healthy patterns
    p1 = gmmP.score_samples(np.transpose(study.X[:,te])) # test against patients' model
    p2 = gmmH.score_samples(np.transpose(study.X[:,te])) # test against healthy model
    Z = p1 > p2 # get the computed label (if True -> Patient)

    (P_i, R_i, F1_i) = study.classificationAnalysis(Z, te)
    
    P = P + P_i
    R = R + R_i
    F1 = F1 + F1_i

print ('|-----------------------------------------|')
print ('|              GMM Method                 |')
print ('|-----------------------------------------|')
print ('|      Precision | Recall | F1 Measure    |')
print ('|        %.2f    |  %.2f  |     %.2f      |' % (P/ITERS, R/ITERS, F1/ITERS))
print ('|-----------------------------------------|')