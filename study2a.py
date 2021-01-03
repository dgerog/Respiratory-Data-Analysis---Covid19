#
# METHOD: GMM
#
# Data to use:
#  i) ALL (PATIENT & HEALTHY)
# Classes:
#  i) PATIENT: Active
# ii) HEALTHY: Is Not Active
#
# Problem trying to solve: PATIENT vs. NOT PATIENT
#

# import basic configuration
from common import *

AGE_GROUPS = [0,60]

# import custom experiment configuration
from sklearn import mixture

print ('|----------------------------------------------------------------------------------------------------------|')
print ('|                                                    GMM                                                   |')
print ('|                                                                                                          |')
print ('|  CASE 1 => PATIENT: Is Active  --  HEALTHY: Not Patient                                                  |')
print ('|  CASE 2 => PATIENT: Is Patient --  HEALTHY: Not Patient                                                  |')
print ('|----------------------------------------------------------------------------------------------------------|')
print ('|  CASE  |   Age Min | Age Max | # of Records | # of Patients | Precision | Recall | F1 Measure | Accuracy |')
print ('|----------------------------------------------------------------------------------------------------------|')

# Split age groups
dataX = study.flattenData(_appendThis=None)
ages = study.F[study.AGE_LINE,:].astype(int)
for ageGroup in range(0, len(AGE_GROUPS)-1):
    # find records in this age group
    ageInd = np.where((ages>=AGE_GROUPS[ageGroup]) * (ages<AGE_GROUPS[ageGroup+1]))
    ageInd = ageInd[0]
    # stabilize
    if DO_STABILIZE:
        indP = np.where(study.isPatient[ageInd] == True)
        indH = np.where(study.isPatient[ageInd] == False) 
        ageInd = study.stabilizeSet(_indP=ageInd[indP[0]], _indH=ageInd[indH[0]])
    #  run iterations to eliminate randomness
    for iter1 in range(0,ITERS):
        # split in trainig and validation
        (trALL, teALL) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=ageInd)
        # start training (exhaustive search)
        bestF1 = -1
        for iter2 in range(0,ITERS):
            # split initial training in training and testing
            (tr, te) = study.prepareCrossValidation(_trainPct=TRAIN_PCT, _allInd=trALL)
            # train
            indH = tr[np.where(study.isActive[tr] == False)]
            indP = tr[np.where(study.isActive[tr] == True)]
            classifier = mixture.GaussianMixture(
                n_components=2, 
                means_init=[
                        np.transpose(np.mean(dataX[:,indH[0]], axis=1)),
                        np.transpose(np.mean(dataX[:,indP[0]], axis=1)),
                    ]
            )
            classifier.fit(np.transpose(dataX[:,tr]))
            # predict & analyze
            Z = classifier.predict(np.transpose(dataX[:,te]))
            (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), study.isActive[te])
            if F1 > bestF1:
                bestF1  = F1
                bestC = classifier
    # validate model
    Z = bestC.predict(np.transpose(dataX[:,teALL]))
    (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), study.isActive[teALL])
    print ('| CASE 1 |   %7.1f | %7.1f | %12d | %13d | %9.2f | %6.2f | %10.2f | %8.2f |' % (AGE_GROUPS[ageGroup], AGE_GROUPS[ageGroup+1], len(ageInd), np.sum(study.isActive[ageInd] == 1), P, R, F1, A))
    (P, R, F1, A) = study.classificationAnalysis(Z.astype(bool), study.isPatient[teALL])
    print ('| CASE 2 |           |         | %12d | %13d | %9.2f | %6.2f | %10.2f | %8.2f |' % (len(ageInd), np.sum(study.isPatient[ageInd] == 1), P, R, F1, A))
    print ('|__________________________________________________________________________________________________________|')