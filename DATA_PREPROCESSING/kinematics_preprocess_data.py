import pickle
import numpy as np
import pandas as pd
import utilities

RANDOM_STATE = 42
MAIN_DATA_FOLDER = "D:/.../MAIN_CODE/DATA/"
KINEMATICS_RAW_DATA = MAIN_DATA_FOLDER+"KINEMATICS_RAW_DATA/"

#################################################################################################################
#
# load the data
#
#################################################################################################################

labels = pd.read_csv("DATA/avg_legibility_score.csv")
labels=labels[["subj","rep"]]
labels["sub_id"]=labels["subj"]

########################################################################################################################
#
# get train and test sets split into windows
#
########################################################################################################################
(X, Y, subjInfo) = utilities.load_multilabel_kinematics_data(KINEMATICS_RAW_DATA, labels, False)
subjInfo = np.array(subjInfo)
Y = np.array(Y)

with open('DATA/subj_labels.pickle','wb') as handle:
    pickle.dump((X, Y, subjInfo),handle,protocol=4)
