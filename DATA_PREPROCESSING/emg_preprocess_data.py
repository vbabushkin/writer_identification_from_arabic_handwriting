# This is the main code for EMG data preprocessing
import pickle
import numpy as np
import pandas as pd
import path_variables
import utilities
#################################################################################################################
#
# parameters
#
#################################################################################################################
winSize = 1024
output_fs = 128 #Hz
#################################################################################################################
#
# load the data
#
#################################################################################################################
# load experts labels data
dfExpertLabels = pd.read_csv(path_variables.EXPERT_LABELS_PATH)
# read EMG data
(X, Y, mainSubjInfo, subjInfo) = utilities.read_emg(dfExpertLabels)

# X original EMG series
# Y EMG labels
# main subject info -- file that contains information about sibject paragraph and legibility score
# output_fs -- final sampling rate
# segment processed EMG data into lines (here the line index is added to the end of the mainSubjInfo):
(seg_X, seg_Y, mainSubjInfo, subjInfo) = utilities.segment_emg(gradThreshold = 1.2, saveToFile = True)

# load the segmented data
with open(path_variables.NEW_EMG_SLICED_DATA + '/emg_all_lines_labels.pickle','rb') as handle:
    (seg_X, seg_Y, mainSubjInfo, subjInfo)  = pickle.load(handle)
# downsample EMG data:
(new_X,new_Y,new_mainSubjInfo)=utilities.downsample_emg(seg_X,seg_Y,mainSubjInfo, output_fs)

#################################################################################################################
#
# Cleaning the data
#
#################################################################################################################

X = new_X
Y = new_Y
# find empty frames and also dots and short stroks (not words)
emptyFramesIndices = []
allFramesShapes = []
for k in range(len(X)):
    allFramesShapes.append(X[k].shape[0])
    if X[k].shape[0] < 100:
        emptyFramesIndices.append(k)

allFramesShapes = np.array(allFramesShapes)
sortedAllFramesShapes = np.sort(allFramesShapes)

idxToRemove =[]
for recLength in sortedAllFramesShapes:
    if recLength<winSize:
        j = np.where(allFramesShapes==recLength)[0][0]
        print(recLength,",",j)
        idxToRemove.append(j)

for i in sorted(idxToRemove, reverse=True):
    del X[i]
    del Y[i]
    del mainSubjInfo[i]
#################################################################################################################
#
# Combine all lines into 1 paragraph:
#
#################################################################################################################
mainSubjInfo = np.array(mainSubjInfo)
subjInfo = np.array(subjInfo)
Y = mainSubjInfo[:,0]
X_main = []
Y_main = []
main_subj_info =  []
# combine into 300 samples
for subj in range(1,51):
    paragraphs = np.unique(mainSubjInfo[np.where(mainSubjInfo[:,0]==subj)[0],1])
    # for each paragraph find number of lines
    for par in paragraphs:
        lines = np.unique(mainSubjInfo[np.where((mainSubjInfo[:,0]==subj) & (mainSubjInfo[:,1]==par))[0],2])
        line_idx = np.where((mainSubjInfo[:,0]==subj) & (mainSubjInfo[:,1]==par))[0]
        X_tmp = X[line_idx[0]]

        shapes_array = [X[line_idx[0]].shape[0]]
        for l_idx in line_idx[1:]:
            X_tmp = np.vstack((X_tmp,X[l_idx]))
            shapes_array.append(X[l_idx].shape[0])
        X_main.append(X_tmp)
        Y_main.append(subj)
        main_subj_info.append([subj,par,subj])
        del(X_tmp)

main_subj_info = np.array(main_subj_info)
#################################################################################################################
#
# Check sizes and save the clean data for use in the models
#
#################################################################################################################
sizes_array = []
for i in range(len(X_main)):
  sizes_array.append(X_main[i].shape[0])
sizes_array = np.array(sizes_array)

print(np.min(sizes_array))
print(np.max(sizes_array))
Y_main = np.array(Y_main)
with open('DATA/emg_clean50.pickle','wb') as handle:
    pickle.dump((X_main, Y_main, main_subj_info), handle)