# The intrinsic geometry of reading
This repository contains the code and notebooks required to run the analysis and generate the figures presented in our manuscript "The intrinsic geometry of reading" 

To run this repository and recreate in analysis there are several files which we do not have the rights to distribute but they are freely available. 
These files are 
1. The Human Connectome Project (HCP) behvaioral data which can be downlaoded [here.](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release)
2. The HCP has a separate set of restricted variables which you can be read more about how to access [here.](https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage)
    - The restricted files are necessary as they were used to ensure that splits in cross validatoin folds respected faily structure. 
    - Furthermore, confounds including age, height, weight, and blood pressure are found in this dataset

To recreate the analysis there are several other dependencies which must be installed. 
This repository uses primarily Python but calls some functions through PALM in MATLAB. 
Therefore you need to add these to your matlab path in all notebooks
- [PALM](https://github.com/andersonwinkler/PALM)
- [PermCCA](https://github.com/andersonwinkler/PermCCA) *note this dependency will be deprecated in the future

Finally, you will need download the associated data which is hosted on [Zenodo](https://zenodo.org/) <!---(update link when files are uploaded)-->
Upon downloading this data repository, set its path to the **data_dir** variable in each notebook. 

## Notebook 1
Notebook 1 will recreate our cross validation scheme such that you can recreate the same training and testing folds on your local machine. 
If you choose not to run this, these folds are already saved in this working directory in the *cross_validation_folds.json* file 

## Notebook 2
This notebook will replicate the feature selection. The notebook runs an example of the correlation and p-value calculation from each CV fold however to actually run this code we recommend running the parallelized scripts in the *nb_2_cluster_script* directory. However if you wish to skip this example and simply run the feature selection step after calculating the initial correlation and p-values between brain data and reading, then you will find them in the data directory in the folders *distCPMfeaturesReading* and *funcCPMfeaturesReading* respectively where they can be loaded as *prebaked* values

## Notebook 3 
In notebook 3 the main analysis of the manuscript is performed. Here you will find the code to run all analysis of the functional connectivity and cortical distance-based models. Notably, the permutations have been generated separately using the subjects included in the analysis. For more informatoin [see](https://brainder.org/2015/12/07/permutation-tests-in-the-human-connectome-project/). The guide for generating these permutations can be found [here](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)ExchangeabilityBlocks.html#EBs_for_data_of_the_Human_Connectome_Project). If you do not wish to generate them fro scratch then you can simply use the precalculated permutatoin blocks in the directory *fold_permutations*. 

All other prerequesite files can be found either in this directory i.e. the feature json files, or through the data directory. Precalculated results of model performance are also in th working directory in the *distance_CV_results.csv* as well as the *network_centered_CV_results* and the *internetwork_models_CV_results.csv* files.

## Notebook 4 
This notebook will visualize the features both in the feature space via circle plots and heatmaps, as well as on the cortical surface. 
If you run this notebook in it's entirely than do know that the gifti output files will require the palette to be changed and the cortex structure set in the metadata which can be done through the connectome workbench commands wb_command -set-structure and wb_command -metric-palette. 
Otherwise, simply open the figures.scene file in connectome workbench and the surface components of all visualizations are readily available.



