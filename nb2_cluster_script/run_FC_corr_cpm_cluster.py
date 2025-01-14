import json
import numpy as np
import pandas as pd
import sys

import pandas as pd 
import numpy as np 
import datetime
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import CCAtools
import seaborn as sns

import matlab.engine as mlab_eng
import matlab


path2PermCCA = "PermCCA/"  ### https://github.com/andersonwinkler/PermCCA
### note environment requirement being phased out ^^
palmPath = "palm-alpha119_2/"  ### path to palm
### see https://github.com/andersonwinkler/PALM
eng=mlab_eng.start_matlab()
eng.addpath(path2PermCCA)
eng.addpath(palmPath)

from CCAtools.preprocessing import prep_confounds,cube_root,gauss_SM,zscore,normal_eqn_python



##### preprocessing functions we'll add to a package later 
from CCAtools.preprocessing import zscore
def prep_confounds_local(confs,eng):
    """ set the confounds up with gaussianization and normalization as done by smith et al 2015."""
    assert ('palm' in eng.path())==True,'add PermCCA to your matlab path'
    mat_data=matlab.double(confs.values.tolist()) ### they actually included the binary acquisition data in the gaussianization
    print('gaussianizing')
    gaussed=np.asarray(eng.palm_inormal(mat_data))
    squared=gaussed[:,1:]**2   
    ready_confs=np.hstack([gaussed,squared])
    ready_confs=zscore(np.hstack([gaussed,squared]),ax=0)

    return ready_confs

def set_confounds(data,mlab_eng):
    """takes in a full data set of all HCP variables and extracts and preprocesses confounds to be regressed"""
    eng=mlab_eng
    confounds=data
    gend=LabelEncoder().fit_transform(confounds['Gender'])
    acq=LabelEncoder().fit_transform(confounds['Acquisition'])
    acq[acq<2]=0
    acq[acq>0]=1  
    df=confounds.copy()
    df['Acquisition']=acq
    df['Gender']=gend
    df['FS_IntraCranial_Vol']=data['FS_IntraCranial_Vol'].map(cube_root)
    df['FS_BrainSeg_Vol']=data['FS_BrainSeg_Vol'].map(cube_root)
    df['Larea']=data['Larea'].map(np.sqrt)
    df['Rarea']=data['Rarea'].map(np.sqrt)
    confounds=prep_confounds_local(df,eng)
    return confounds

def preprocess_SM(data,confs,mlab_eng):
    """preprocess the subject measures. Guassianize and remove confounds."""
    eng=mlab_eng
    assert ('palm' in eng.path())==True,'add PermCCA to your matlab path'
    data=data
    gaussed=gauss_SM(data,eng)
    residuals=normal_eqn_python(confs,gaussed)
    cleaned=zscore(residuals)
    cleaned=pd.DataFrame(cleaned,index=data.index,columns=data.columns)
    return cleaned

def preprocessDists(data,confounds):    
    NET=data.copy()
    dims=NET.shape
    ##### check for vertices with no variance i.e guaranteed masks 
    steady_masks=np.where(np.sum(NET)==0)[0]
    valididx=np.where(np.sum(NET)!=0)[0]
    
    if len(steady_masks)!=0:
        NET=NET.iloc[:,valididx]
        
#     amNET = np.abs(np.nanmean(NET, axis=0))
    NET1 = NET#/amNET
    NET1=NET1-np.mean(NET1,axis=0)
    NET1=NET1/np.nanstd(NET1.values.flatten())
    NET1=normal_eqn_python(confounds,NET1)
    NET1=pd.DataFrame(NET1,columns=NET.columns,index=data.index)
    
    if len(steady_masks)!=0:
        out=np.zeros(dims)
        out[:,valididx]=NET1.values
        NET1=pd.DataFrame(out,index=NET.index)
    
    return NET1


def clean_data(subjectList,all_confs,all_SM,all_dist,mlab=eng):
    ### remove confounds for a group of subjects
    ### always done independently to avoid leakage
    
    ## set the confounds 
    confs=all_confs.loc[subjectList]
    confs=set_confounds(confs,mlab)
    ## regress them from behavioral measures
    behavior=all_SM.loc[subjectList]
    behavior=preprocess_SM(behavior,confs,mlab)
    ## regress them from distance measures 
    distance=all_dist.loc[subjectList]
    distance=preprocessDists(distance,confs)
    
    return confs,behavior,distance

from scipy.stats import spearmanr
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pandas as pd

# Adjusted function to compute Spearman's correlation for a single column.
# Now, it directly takes the column data as an argument instead of the column index and entire DataFrame.
def corr_single_column(column_data, beh_data, behavior):
    r, p = spearmanr(column_data, beh_data[behavior])
    return r, p

# Parallel version of corrDist2BEH that's more memory-efficient
def corrDist2BEH_parallel(dist, beh_data, behavior):
    num_processes = multiprocessing.cpu_count()  # Utilize all available cores
    
    # Prepare column data for each task to reduce memory usage
    # This time, instead of passing the whole DataFrame and column index,
    # pass the data of each column directly.
    tasks = [(dist.iloc[:, i].values, beh_data, behavior) for i in range(dist.shape[1])]

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Execute the tasks in parallel and collect the results
        results = pool.starmap(corr_single_column, tasks)
    
    # Separate the results into correlation coefficients and p-values
    cr, p_vals = zip(*results)
    
    return np.asarray(cr), np.asarray(p_vals)



### set paths
restricted_hcp_path = ""  ## set path to your copy of hcp restricted data
hcp_behavior_data = ""  ### set path to your copy of hcp behvaioral data
data_dir = ""  ### set to directory for data used in/throughout analysis

#### load behavioral data
### this is HCP restricted data. for access see
### https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage
RestrictedData = pd.read_csv(restricted_hcp_path, index_col="Subject")
### load non-restricted hcp behavioral data
BehData = pd.read_csv(hcp_behavior_data, index_col="Subject")
### merge the dtaframes
fullData = BehData.merge(RestrictedData, on="Subject")
fullData.index = fullData.index.map(str)
fullData.index = fullData.index.map(str)
### load in square_root of total hemisphere surface area for each subject
Larea = pd.read_csv(f"{data_dir}/LareaFactors.csv")
Larea.rename(index={0: "Larea"}, inplace=True)
Rarea = pd.read_csv(f"{data_dir}/RareaFactors.csv")
Rarea.rename(index={0: "Rarea"}, inplace=True)
area = pd.concat([Larea, Rarea]).T
area.index.names = ["Subject"]
fullData = area.join(fullData, on="Subject", how="inner")

#### Load FC data
LSchaeferFC = pd.read_csv(f"{data_dir}/Schaefer200FConn.L.mat.csv")
LSchaeferFC.rename(columns={"Unnamed: 0": "Subject"}, inplace=True)
LSchaeferFC.set_index("Subject", inplace=True)
LSchaeferFC.columns = [f"L.{i}" for i in LSchaeferFC.columns]

RSchaeferFC = pd.read_csv(f"{data_dir}/Schaefer200FConn.R.mat.csv")
RSchaeferFC.rename(columns={"Unnamed: 0": "Subject"}, inplace=True)
RSchaeferFC.set_index("Subject", inplace=True)
RSchaeferFC.columns = [f"R.{i}" for i in RSchaeferFC.columns]
SchaeferFC = pd.concat([LSchaeferFC, RSchaeferFC], axis=1)
SchaeferFC.index = SchaeferFC.index.map(str)


### remove the two subjects that have missing data when generating permutations 
#### remove a few subjects that don't have all the data to generate the right permutations
SchaeferDist.drop(['168240','376247'],inplace=True)
fullData.drop(['168240','376247'],inplace=True)
Larea.T.drop(['168240','376247'],inplace=True)
Rarea.T.drop(['168240','376247'],inplace=True)
area.drop(['168240','376247'],inplace=True)

### load confound data 

conf_categories=fullData[['Acquisition','Gender','Age_in_Yrs','Height','Weight','BPSystolic','BPDiastolic',
                    'FS_IntraCranial_Vol','FS_BrainSeg_Vol','Larea','Rarea']]
# conf_categories.dropna(inplace=True)

SensoryTasks=['ReadEng_Unadj']
# subjects=list(fullData.index)
subjSM=fullData[SensoryTasks]


full_subjList=subjList=[str(i[0]) for i in pd.read_csv('SubjectInclusionList.csv',index_col=0).values]

conf_categories=conf_categories.loc[full_subjList]
subjSM=subjSM.loc[full_subjList]

complete_data=fullData.loc[full_subjList]





# Check if the command-line argument is provided
if len(sys.argv) != 2:
    print("Usage: python script_name.py fold_index")
    sys.exit(1)

fold_index = sys.argv[1]

# Open JSON file
with open('cross_validation_folds.json', 'r') as f:
    data = json.load(f)

# Extract the appropriate dataset based on command-line argument
fold_json = list(data.keys())[int(fold_index)]
subjects = data[fold_json]['training']


### clean data for feature selection via spearman correlation 
_,beh,dist=clean_data(subjects,conf_categories,subjSM,SchaeferDist)
    
### run the spearman correlation on this fold of the training data
### save out the correlation and p-values for each fold 
corrs=np.zeros([1,39800])
p_vals=np.zeros([1,39800])
for idx,task in enumerate(subjSM.columns):
    rho,p=corrDist2BEH_parallel(dist,beh,task)
    corrs[idx,:]=rho
    p_vals[idx,:]=p
corrs=pd.DataFrame(corrs)
corrs.columns=dist.columns
corrs.index=beh.columns
p_vals=pd.DataFrame(p_vals)
p_vals.columns=dist.columns
p_vals.index=beh.columns

out_int=fold_json.split('d')[1]
corrs.to_csv(f'{data_dir}/funcCPMfeaturesReading/fold{out_int.zfill(3)}_corrs.csv')
p_vals.to_csv(f'{data_dir}/funcCPMfeaturesReading/fold{out_int.zfill(3)}_pvals.csv')
