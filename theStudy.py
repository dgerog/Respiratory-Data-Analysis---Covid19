import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
import math

class theStudy:
    """
        theStudy class introduces all the data parsing concepts presented in the paper
        
        Class parameters:
            self.X: Actual respiratory data
            self.F: Augemnted featires
            
        Supported methods (public):
            readTable: Get the columns of an excel/csv file
            kneeThresholding: Knee thresholding on the covariance matrix - Estimate number of clusters
            prepareCrossValidation: Split the training set in two subsets (random sampling) for cross validation
            classificationAnalysis: Analyse the clasification results (Precission/Recall - F1 Measure)
    """
    
    def __init__(self):
        #initialize properties
        self.X = []
        self.F = []
        self.isPatient = []
        self.isActive = []
        self.isNonActive = []
        self.patientRecs = []
        self.healthyRecs = []
    #
    # Public
    #

    """
        Method: kneeThresholding
    """    
    def kneeThresholding(self, _kernelCov=None, _X=None, _oFName=None):
        """
            Perform knee/elbow thresholding.
            
            To determine the number of clusters:
                1. Order the input.
                2. Plot the input (x-axis: input index, y-axis: input)
                3. Compute the line crosses the points marked by the first and last
                   input of the previous plot.
                4. Compute the distances of all the points of the previous plot
                   to the line computed in step 3.
                5. Detect the point with the largest distance (knee detection).
                6. The index of the point coresponds to expected threshold.

            _kernelCov: A string with the preprocessing on the data. None for no preprocessing.
            _oFName: The full path to save the plots. None for no ploting
                           
            RETURN: Number of clusters (estimated)
        """
        #
        # Part 1. Estimate the threshold
        #

        # where to apply? If none -> use all records
        if _X is None:
            _X = self.X
        
        # compute covariance matrix
        if (not _oFName is None):
            #kernel PCA - To DO
            S = np.cov(_X)
        else:
            #linear (typical cov matrix)
            S = np.cov(_X)

        # compute eigenvalues
        [v, e] = np.linalg.eig(S)

        # sort eigenvalues - keep magnitude (in case of complex eigenvalues)
        _Y = np.sort(np.abs(v))[::-1]

        # diagonal line (equation)
        _X = range(0,_Y.shape[0])
        P1 = np.array([0         , _Y[0]])
        P2 = np.array([_X[_Y.shape[0]-1], _Y[_Y.shape[0]-1]])
        
        l = (P2[1] - P1[1])/(P2[0] - P1[0])
        b = P2[1] - l*P2[0]
        
        # find distances (Euclidean)
        d = []
        Q = np.sqrt(l*l + 1)
        for i in range(0,_Y.shape[0]):
            d.append(np.abs(l*i - _Y[i] + b)/Q)
        
        # find max
        x_max = np.argmax(d)
        P0 = [_X[x_max], _Y[x_max]] # this is the point with max distance
        
        #
        # Part 2. Plot
        #
        
        if (not _oFName is None):
            #plot and save
            plt.plot(_X, _Y, 'b-', linewidth=1, markersize=5)
            plt.plot([P1[0], P2[0]], [P1[1], P2[1]], 'r-', linewidth=3, markersize=6)
            plt.plot([P0[0], P1[0]], [P0[1], P1[1]], 'gh-', linewidth=3, markersize=6)
            plt.plot([P0[0], P2[0]], [P0[1], P2[1]], 'gh-', linewidth=3, markersize=6)            
            plt.plot(P0[0], P0[1], 'ks', linewidth=3, markersize=6)
            plt.title('K = ' + str(P0[0]))
            plt.savefig(_oFName, bbox_inches='tight', pad_inches=1)
            plt.close()
        
        return(P0[0])

    """
        Method: readTable
    """ 
    def readTable(self, _path,  _colsToRead, _sheetToRead=0, _doAppend=False, _doFilterData=False):
        """
            Read an excel/csv file to extract the table data.
            Patients' records are stored vertically (Columns)
            
            _path (str): FULL path of the excel file to read - Read permission is assumed
            _colsToRead (str): Columns to read, eg. D:AF meaning read cols from D to AF. None to read all.
                               IMPORTANT: if _headersRow is True then make sure the headers row is also include.
            _sheetToRead (int): Which sheet to read (1 for the first sheet, etc.)
            _doAppend: Instruct if the new records will be added to the end of the existing records.
                        FALSE: Initialize with the data - Drop old records (default).
                        TRUE: Append to existing records.
            _doFilterData: Wheather to remove max value (True/False)
        """
        # Read data
        T = pd.read_excel(io=_path, header=None, usecols=_colsToRead, sheet_name=_sheetToRead, dtype=str).values

        #
        # Get extra features
        #
        # _Lines start from index 0_
        #
        # Line 1: STATUS
        # Line 2: PCR
        # Line 3: PCR HOSPITALIZED
        # Line 4: RAPID TEST
        # Line 5: SMOKING
        # Line 6: AGE (YEARS)
        F = T[0:6, :]

        #
        # Read the respiratory data
        #
        X = T[7:,:].astype(float)

        if _doFilterData:
            # upper bound filter
            for i in range(0,X.shape[1]):
                th = np.max(X[:,i])
                X[X[:,i]>=th,i] = np.min(X[:,i])

        #
        # Assign patient labels
        #
        isPatient = (F[0,:] == 'P').astype(bool)

        #isActive = (isPatient == True) and (F[2,:].astype(bool) or F[3,:].astype(bool))
        #isNonActive = (isPatient == True) and not isActive

        if (_doAppend):
            self.X = np.concatenate((self.X, X), axis=1) if len(self.X) else X
            self.F = np.concatenate((self.F, F), axis=1) if len(self.F) else F
            self.isPatient = np.concatenate((self.isPatient, isPatient)) if len(self.isPatient) else isPatient
            #self.isActive = np.concatenate((self.isActive, isActive)) if len(self.isActive) else isActive
            #self.isNonActive = np.concatenate((self.isNonActive, isNonActive)) if len(self.isNonActive) else isNonActive
        else:
            self.X = X
            self.F = F
            self.isPatient = isPatient
            #self.isActive = isActive
            #self.isNonActive = isNonActive

    """
        Method: prepareCrossValidation
    """ 
    def prepareCrossValidation(self, _trainPct=.9):
        """
            Suffle the training set and prepare two sets: Training & Testing
            
            _trainPct: (% in (0,1)) - percentage of data to use for the training set
                        Default is 10-fold cross validation

            RETURN: (trainInd, testInd): The indices of self.X to contain the training and testing observations respectively.
        """

        if _trainPct <=0 or _trainPct >=1:
            _trainPct = .9 # default is 10-fold cross validation
        
        allInd = list(range(0,self.X.shape[1]))
        random.shuffle(allInd)

        breakInd = math.floor(_trainPct*self.X.shape[1])
        trainInd = allInd[0:breakInd]
        testInd  = allInd[breakInd:]

        return(trainInd, testInd)
        
    """
        Method: classificationAnalysis
    """ 
    def classificationAnalysis(self, _Y, _Ind):
        """
            Analyze the classification results.
            Compute:
                Precission
                Recall
                F1 Measure
            
            _Y: The computed labels - Analyze this classification result (True/False).
            _Ind: The index of the observation => Y[i] is tested against self.X[_Ind[i]]

            RETURN: (P,R,F1): Precission, Recall, F1 Measure
        """

        TP = 0 # True Positive
        TN = 0 # True Negative
        FP = 0 # False Positive
        FN = 0 # False Negative

        for i in range(0,_Y.shape[0]):
            if self.isPatient[_Ind[i]] and _Y[i]:
                TP = TP + 1
            elif self.isPatient[_Ind[i]] and not _Y[i]:
                FP = FP + 1
            elif not self.isPatient[_Ind[i]] and _Y[i]:
                FN = FN + 1
            elif not self.isPatient[_Ind[i]] and not _Y[i]:
                TN + TN + 1

        R = TP/(TP+FN)
        P = TP/(TP+FP)
        F1 = 2*P*R/((P+R) + .0000000000000000001)

        return (P,R,F1)