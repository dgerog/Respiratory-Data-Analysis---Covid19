# Respiratory Data Analysis - Covid19

The purpose of this study is to analyze respiratory data and distinguish between infected and non infected observations.

This repo includes the puthon code used to study the data and produce the prediction model. The training and testing data is **not** included. Please contact me for further details about accessing the dataset.

The main analysis is executed in the _theStudy.py_ file, where class `theStudy` is implemented with the following properties and methods:

* **Class parameters:**
    * **X**: Actual respiratory data
    * **F**: Augmented features (Labels, Age, Smoking/Non Smoking, etc.)
            
* **Supported methods (public):*
    * *readTable:** Get the columns of an excel/csv file (data import)
    * **writeTable:** Save the read records in a csv file (data export)
    * **kneeThresholding:** Knee thresholding on the covariance matrix - Estimate number of clusters
    * **prepareCrossValidation:** Split the training set in two subsets (random sampling) for cross validation
    * **classificationAnalysis:** Analyse the clasification results (Precission/Recall - F1 Measure)
    * **trimData:** Trim the specified indices
    * **stabilizeSet:** Handle unbalanced datasets -> Make all classes have almost equal number of observations

## Methods Tested

Several clustering methodologies were stidied, as described in the following files. Two scenarios were tested:
1. All records were used to cluster Petients vs. Non Patients (CASE 1)
2. ONLY patient's records were used to cluster Active vs. Non Active (CASE 2)

Obesrvations were split in age groups and for each group a dedicated mpdel was trained with a cross validation testing approach.
The age groups are as follows: [0,20) - [20,40), [40,65), [65,100). Age groups can be configured by setting the `AGE_GROUPS` variable in _common.py_ file.

Files:
* _common.py_: Experiment configuration and constants.
* _study1a.py_ (case 1 above), _study1b.py_ (case 2 above): SVM classifier.
* _study2a.py_ (case 1 above), _study2b.py_ (case 2 above): GMM classifier with 2 components (patient, non patient).
* _study3a.py_ (case 1 above), _study3b.py_ (case 2 above): Decision Trees.

## Handling inbalanced classes

In order to deal with the problem if imbanaced classes (especially for younger groups), we trimmed the larger classes, by randomly selecting a secified number of observations. We noticed that in order to get better results, a ratio 2 patienst for 1 non patient should be introduced in our observations.