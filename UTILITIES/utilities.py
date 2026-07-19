import csv
import glob
import json
import os
import pickle
import warnings
from sys import platform
import datetime
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from PIL import Image
from matplotlib.ticker import MaxNLocator
from scipy import signal
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix

import kinematic_features_names
import path_variables

warnings.filterwarnings('ignore')
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
plt.style.use('default')

#################################################################################################################
#
# utilities
#
#################################################################################################################
# for splitting into sliding windows of length L with overlap ov
# a - array,
# L -length of the window
# ov - overlap
def get_strides(a, L, ov):
    if a.shape[0]<L:
        out = np.pad(a, ((0, L-a.shape[0]),(0,0)), 'constant', constant_values=0)
        out = np.expand_dims(out, 0)
    else:
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        tmpA = np.zeros((L, a.shape[1]))
        tmpA[:L - (i + 2 * L - ov - a.shape[0]), :] = a[i + L - ov - 1:a.shape[0] - 1, :]
        out.append(tmpA)
    return np.array(out)

# filesArray -- array of the subject files with the kinematics data
# labelDf -- dataframe where all the labels are stored
# labelName -- the name of the label we want to use for classification
# applyFilter -- boolean value if we want to clean the data with median filter
# MED_FILTER_WIN -- the length of median filter
def load_kinematics_data(datapath, labelDf, labelName, applyFilter, MED_FILTER_WIN = 20):
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []
    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*.csv")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)

    X = []
    Y = []
    subjInfo = []
    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[2])
            repNum = int(currentFile.split('/')[-1].split('_')[3][-1])
        df = pd.read_csv(currentFile)
        colsToRemove = ['handId', 'sec', 'min', 'hour', 'lifetimeOfThisHandObject', 'confidence']
        df = df.drop(colsToRemove, axis=1)
        featuresNames = df.columns.to_numpy()
        tmpX = df[featuresNames].to_numpy().astype(np.float32)
        tmpX = tmpX[2:-2, :]

        tmpDf = labelDf.set_index(['subj', 'rep'])
        label = tmpDf.at[(subjNum, repNum), labelName].astype(int)

        Y.append(label)

        w = np.zeros(shape=tmpX.shape)
        # apply median filter
        if applyFilter:
            for j in range(tmpX.shape[1]):
                w[:, j] = median_filter(tmpX[:, j], MED_FILTER_WIN)
            X.append(w)
        else:
            X.append(tmpX)
        subjInfo.append([subjNum, repNum, label])
    return(X,Y, subjInfo)


def split_data_by_windows(dataSet,labelSet,subjInfo, winSize, overlap):
    tmp_X = []
    tmp_Y = []
    tmpSubjInfo = []
    for i in range(len(dataSet)):
        # sliding window
        a = get_strides(dataSet[i], winSize, overlap)
        subjNum = subjInfo[i][0]
        repNum = subjInfo[i][1]
        #legScore = subjInfo[i][3]
        label = labelSet[i]
        tmp_X.append(a)
        tmp_Y.append(np.repeat(label, a.shape[0]))
        tmpSubjInfo.append(np.repeat([[subjNum,repNum, label]], a.shape[0], axis = 0))

    for i in range(len(tmp_X)):
        if i == 0:
            X = tmp_X[i]
            Y = tmp_Y[i]
            subjInfo = tmpSubjInfo[i]
        else:
            X = np.vstack((X, tmp_X[i]))
            Y = np.hstack((Y, tmp_Y[i]))
            subjInfo = np.hstack((subjInfo, tmpSubjInfo[i]))
    return(X,Y,subjInfo)

def load_emg_data(datapath, labels):
    filesArray = glob.glob(datapath + "/*.csv")
    filesArray.sort()
    X = []
    Y = []
    subjInfo = []
    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[3].split('.')[0])
        elif platform == "win32":
            subjNum = int(currentFile.split('\\')[-1].split('_')[1])
            repNum = int(currentFile.split('\\')[-1].split('_')[3].split('.')[0])

        df = pd.read_csv(currentFile)
        df = df.drop("Timestamp", axis=1)
        featuresNames = df.columns.to_numpy()
        tmpX = df[featuresNames].to_numpy().astype(np.float32)
        tmpDf = labels.set_index(['subj', 'rep'])
        legibility_score = tmpDf.at[(subjNum, repNum), 'legibility_score'].astype(int)
        X.append(tmpX)
        Y.append(legibility_score)
        subjInfo.append([subjNum, repNum, legibility_score])
    return (X,Y,subjInfo)


# filesArray -- array of the subject files with the kinematics data
# labelDf -- dataframe where all the labels are stored
# labelName -- the name of the label we want to use for classification
# augmentation -- boolean value if we want to clean the data with median filter
def load_visual_data(datapath, labelDf, size = (250, 250)):
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []
    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*/*.png")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)

    X = []
    Y = []
    subjInfo = []
    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[4])
            repNum = int(currentFile.split('/')[-1].split('_')[5][-1])

        # find in labels
        legibility_score = np.array(labelDf["legibility_score"][(labelDf["subj"] == subjNum) & (labelDf["rep"] == repNum)])[0]
        Y.append(legibility_score)

        # preprocess the images
        # load the image
        image = Image.open(currentFile)
        imageBox = image.getbbox()
        cropped = image.crop(imageBox)
        resized = cropped.resize(size)
        # convert image to numpy array
        data = np.asarray(resized)
        # append to dataset
        X.append(data)
        subjInfo.append([subjNum, repNum, legibility_score])


    X = np.array(X)
    Y = np.array(Y)
    return(X,Y,subjInfo)

## plot distribution
def plot_y_test_train(y_train, y_test, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,4),sharey = True)
    labels_tr, counts_tr = np.unique(y_train, return_counts=True)
    labels_ts, counts_ts = np.unique(y_test, return_counts=True)
    bar1 = ax1.bar(labels_tr, counts_tr, align='center')
    ax1.bar_label(bar1)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(labels_tr, labels_tr)
    ax1.set_xlabel("legibility score, train")
    ax1.set_ylabel("count of samples")
    ax1.spines[['right', 'top']].set_visible(False)
    bar2 = ax2.bar(labels_ts, counts_ts, align='center')
    ax2.bar_label(bar2)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xticks(labels_ts, labels_ts)
    ax2.set_xlabel("legibility score, test")
    ax2.set_ylabel("count of samples")
    ax2.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)
    plt.close()

# split into train test for each fold
# currentTrainSubjInfo contains train indices for train set of current fold
# currentTestSubjInfo contains test indices for test set of current fold
# subjInfo contains main dataset indices before splitting
# X and Y are main dataset X and lables Y before splitting
def fold_train_test_windows(X,Y,currentTrainSubjInfo,currentTestSubjInfo, subjInfo, winSize, overlap, scale = True):
    # create training set
    for i in range(len(currentTrainSubjInfo)):
        currentTrainSubj = currentTrainSubjInfo[i, 0]
        currentTrainPar = currentTrainSubjInfo[i, 1]
        currentTrainIdx = np.where((subjInfo[:, 0] == currentTrainSubj) & (subjInfo[:, 1] == currentTrainPar))[0]
        if i == 0:
            foldTrainIdx = currentTrainIdx
        else:
            foldTrainIdx = np.hstack((foldTrainIdx, currentTrainIdx))

    pre_y_train = Y[foldTrainIdx]
    pre_X_train = [X[idx] for idx in foldTrainIdx]

    # create testing set
    for i in range(len(currentTestSubjInfo)):
        currentTestSubj = currentTestSubjInfo[i, 0]
        currentTestPar = currentTestSubjInfo[i, 1]
        currentTestIdx = np.where((subjInfo[:, 0] == currentTestSubj) & (subjInfo[:, 1] == currentTestPar))[
            0]
        if i == 0:
            foldTestIdx = currentTestIdx  # these are already testing indices that include lines
        else:
            foldTestIdx = np.hstack((foldTestIdx, currentTestIdx))

    pre_y_test = Y[foldTestIdx]
    pre_X_test = [X[idx] for idx in foldTestIdx]

    # now for each line split the data with the sliding window
    trainSet_X = []
    trainSet_Y = []
    testSet_X = []
    testSet_Y = []

    for i in range(len(pre_X_train)):
        tmpX = pre_X_train[i]
        # sliding window
        a = get_strides(tmpX, winSize, overlap)
        trainSet_X.append(a)
        trainSet_Y.append(np.repeat(pre_y_train[i], a.shape[0]))
    X_train = np.concatenate(trainSet_X, axis = 0)
    y_train = np.concatenate(trainSet_Y, axis = 0)
    print("Finished stacking X_train")

    for i in range(len(pre_X_test)):
        tmpX = pre_X_test[i]
        # sliding window
        a = get_strides(tmpX, winSize, overlap)
        testSet_X.append(a)
        testSet_Y.append(np.repeat(pre_y_test[i], a.shape[0]))
    X_test = np.concatenate(testSet_X, axis = 0)
    y_test = np.concatenate(testSet_Y, axis = 0)
    print("Finished stacking X_test")

    if scale:
        # scale
        s = sklearn.preprocessing.StandardScaler()
        for i in range(X_train.shape[0]):
            X_train[i] = s.fit_transform(X_train[i])

        for i in range(X_test.shape[0]):
            X_test[i] = s.fit_transform(X_test[i])
    return(X_train,y_train,X_test, y_test)



# split into train test for each fold
# currentTrainSubjInfo contains train indices for train set of current fold
# currentTestSubjInfo contains test indices for test set of current fold
# subjInfo contains main dataset indices before splitting
# X and Y are main dataset X and lables Y before splitting
def fold_train_test_windows_visual(X,Y,currentTrainSubjInfo,currentTestSubjInfo, subjInfo):
    # create training set
    for i in range(len(currentTrainSubjInfo)):
        currentTrainSubj = currentTrainSubjInfo[i, 0]
        currentTrainPar = currentTrainSubjInfo[i, 1]
        currentTrainIdx = \
        np.where((subjInfo[:, 0] == currentTrainSubj) & (subjInfo[:, 1] == currentTrainPar))[0]
        if i == 0:
            foldTrainIdx = currentTrainIdx
        else:
            foldTrainIdx = np.hstack((foldTrainIdx, currentTrainIdx))

    y_train = np.array(Y[foldTrainIdx])
    X_train = np.array([X[idx] for idx in foldTrainIdx])

    # create testing set
    for i in range(len(currentTestSubjInfo)):
        currentTestSubj = currentTestSubjInfo[i, 0]
        currentTestPar = currentTestSubjInfo[i, 1]
        currentTestIdx = np.where((subjInfo[:, 0] == currentTestSubj) & (subjInfo[:, 1] == currentTestPar))[
            0]
        if i == 0:
            foldTestIdx = currentTestIdx  # these are already testing indices that include lines
        else:
            foldTestIdx = np.hstack((foldTestIdx, currentTestIdx))

    y_test = np.array(Y[foldTestIdx])
    X_test = np.array([X[idx] for idx in foldTestIdx])

    return(X_train,y_train,X_test, y_test)


# plot confusion matrix for each fold
def plot_cm(y_test,yhat,path,filename,ifold):
    cm = confusion_matrix(y_test, yhat, labels=np.unique(yhat))
    classNames = np.unique(yhat)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    div = np.divide(cm.T, np.sum(cm, 1)).T.flatten()
    res = np.argwhere(np.isnan(div)).flatten()
    div[res] = 0
    group_percentages = ["{0:.2%}".format(value) for value in div]  # cm.flatten()/np.sum(cm)]
    labels = ["{}\n{}".format(v1, v2) for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(len(classNames), len(classNames))
    plt.figure(figsize=(15, 8))
    sns.set(font_scale=1.4)  # for label size
    np.min(cm)
    div = (np.divide(cm.T, np.sum(cm, 1)).T) * 100
    res = np.argwhere(np.isnan(div))
    for i, j in res:
        div[i, j] = 0
    ax = sns.heatmap(div,
                     cbar_kws={'ticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}, vmin=0, vmax=100.0,
                     annot=labels, annot_kws={"size": 20}, fmt='', cmap="Blues")  # font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(axis='x', which='major', pad=-3)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
    ax.set_yticklabels(classNames, rotation=0, fontsize="20", va="center")
    ax.set_xticklabels(classNames, rotation=0, fontsize="20", ha="right")
    ax.set_ylabel("True Label", fontsize="51")
    ax.set_xlabel("Predicted Label", fontsize="51")
    plt.tight_layout()
    if ifold is None:
        plt.savefig(path+'cm_' + filename + '.pdf')
    else:
        plt.savefig(path+'cm_' + filename + '_fold_' + str(ifold) + '.pdf')


def plot_cm_50(y_test,yhat,path,filename,ifold):
    # plot confusion matrix
    cm = confusion_matrix(y_test, yhat, labels=np.unique(yhat))
    classNames = np.unique(yhat)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    div = np.divide(cm.T, np.sum(cm, 1)).T.flatten()
    res = np.argwhere(np.isnan(div)).flatten()
    div[res] = 0
    group_percentages = ["{0:.2%}".format(value) for value in div]  # cm.flatten()/np.sum(cm)]
    labels = ["{}\n{}".format(v1, v2) for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(len(classNames), len(classNames))
    plt.figure(figsize=(15, 8))
    sns.set(font_scale=1.4)  # for label size
    np.min(cm)
    div = (np.divide(cm.T, np.sum(cm, 1)).T) * 100
    res = np.argwhere(np.isnan(div))
    for i, j in res:
        div[i, j] = 0
    ax = sns.heatmap(div,
                     cbar_kws={'ticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}, vmin=0, vmax=100.0,
                      fmt='', cmap="Blues")  # font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(axis='x', which='major', pad=-3)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
    ax.set_yticks(classNames)
    ax.set_yticklabels(classNames, rotation=0, fontsize="8", va="center")
    ax.set_xticks(classNames)
    ax.set_xticklabels(classNames, rotation=0, fontsize="8", ha="right")
    ax.set_ylabel("True Label", fontsize="20")
    ax.set_xlabel("Predicted Label", fontsize="20")
    plt.tight_layout()
    if ifold is None:
        plt.savefig(path+'cm_' + filename + '.pdf')
    else:
        plt.savefig(path+'cm_' + filename + '_fold_' + str(ifold) + '.pdf')

def plot_avg_cm_50(cmPerFold,path, filename,classNames=None ):
    cm = np.mean(np.array(cmPerFold), axis=0)
    if classNames is None:
        classNames = np.arrange(1,51)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    div = np.divide(cm.T, np.sum(cm, 1)).T.flatten()
    res = np.argwhere(np.isnan(div)).flatten()
    div[res] = 0
    group_percentages = ["{0:.2%}".format(value) for value in div]  # cm.flatten()/np.sum(cm)]
    labels = ["{}\n{}".format(v1, v2) for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(len(classNames), len(classNames))
    plt.figure(figsize=(15, 8))
    sns.set(font_scale=1.4)  # for label size
    np.min(cm)
    div = (np.divide(cm.T, np.sum(cm, 1)).T) * 100
    res = np.argwhere(np.isnan(div))
    for i, j in res:
        div[i, j] = 0
    ax = sns.heatmap(div,
                     cbar_kws={'ticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}, vmin=0, vmax=100.0,
                      fmt='', cmap="Blues")  # font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(axis='x', which='major', pad=-3)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
    ax.set_yticks(classNames)
    ax.set_yticklabels(classNames, rotation=0, fontsize="8", va="center")
    ax.set_xticks(classNames)
    ax.set_xticklabels(classNames, rotation=0, fontsize="8", ha="right")
    ax.set_ylabel("True Label", fontsize="20")
    ax.set_xlabel("Predicted Label", fontsize="20")
    plt.tight_layout()
    plt.savefig(path+'avg_cm_' + filename + '_fold.pdf')


def plot_avg_cm(cmPerFold,path, filename,classNames=None ):
    avgCm = np.mean(np.array(cmPerFold), axis=0)
    if classNames is None:
        classNames = [0,1,2]
    df_cm = pd.DataFrame(avgCm, index=classNames, columns=classNames)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in avgCm.flatten()]
    div = np.divide(avgCm.T, np.sum(avgCm, 1)).T.flatten()
    res = np.argwhere(np.isnan(div)).flatten()
    div[res] = 0
    group_percentages = ["{0:.2%}".format(value) for value in div]  # cm.flatten()/np.sum(cm)]
    labels = ["{}\n{}".format(v1, v2) for v1, v2 in zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(len(classNames), len(classNames))

    plt.figure(figsize=(15, 8))
    sns.set(font_scale=1.4)  # for label size
    np.min(avgCm)
    div = (np.divide(avgCm.T, np.sum(avgCm, 1)).T) * 100
    res = np.argwhere(np.isnan(div))
    for i, j in res:
        div[i, j] = 0
    ax = sns.heatmap(div,
                     cbar_kws={'ticks': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}, vmin=0, vmax=100.0,
                     annot=labels, annot_kws={"size": 20}, fmt='', cmap="Blues")  # font size
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.tick_params(axis='x', which='major', pad=-3)
    ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
    ax.set_yticklabels(classNames, rotation=0, fontsize="20", va="center")
    ax.set_xticklabels(classNames, rotation=0, fontsize="20", ha="right")
    ax.set_ylabel("True Label", fontsize="51")
    ax.set_xlabel("Predicted Label", fontsize="51")
    plt.tight_layout()
    plt.savefig(path+'avg_cm_' + filename + '.pdf')


# plot loss for each fold
def plot_loss(history,epochs,loss,path, filename, ifold):
    plt.style.use('default')
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig=plt.figure(figsize=(18, 8))
    plt.plot(history.history['loss'], 'b-', label="train loss")
    plt.plot(history.history['val_loss'], 'r-', label="val loss")
    plt.xticks(np.arange(0, epochs + 1, step=0.1 * epochs))
    plt.xlabel("epochs")
    plt.xlim([-0.5, epochs])
    plt.ylabel(loss)
    plt.legend(bbox_to_anchor=(1.0, 0.98))
    plt.grid()
    plt.tight_layout()
    plt.savefig(path+'loss_' + filename + '_fold_' + str(
        ifold) + '.pdf')
    plt.close(fig)

# plot accuracy for each fold
def plot_accuracy(history, epochs, accuracy, path, filename, ifold):
    plt.style.use('default')
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig=plt.figure(figsize=(18, 8))
    plt.plot(history.history[accuracy], 'b-', label="train accuracy")
    plt.plot(history.history['val_'+accuracy], 'r-', label="val accuracy")
    plt.xticks(np.arange(0, epochs + 1, step=0.1 * epochs))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel("epochs")
    plt.xlim([-0.5, epochs])
    plt.ylabel(accuracy)
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path+'accuracy_' + filename + '_fold_' + str(
        ifold) + '.pdf')
    plt.close(fig)

def output_report_per_fold(reportsStrPerFold):
    for rep in reportsStrPerFold:
        print(rep)

def output_avg_values(reportsPerFold,num_classes):
    avgAccuracy = []
    avgRecall = []
    avgPrecision = []
    avgF1 = []
    for rep in reportsPerFold:
        print(pd.DataFrame(rep))
        avgAccuracy.append(rep["accuracy"])
        for i in range(num_classes):
            avgRecall.append(rep[str(i)]["recall"])
            avgPrecision.append(rep[str(i)]["precision"])
            avgF1.append(rep[str(i)]["f1-score"])

    print("Average accuracy: %.3f\nAverage precision:  %.3f\nAverage recall:  %.3f\nAverage F1 : %.3f\n " % (
    np.mean(avgAccuracy), np.mean(avgPrecision), np.mean(avgRecall), np.mean(avgF1)))


def fold_train_test_windows_multilabel(X,Y,currentTrainSubjInfo,currentTestSubjInfo, subjInfo, winSize, overlap, scale = True):
    # create training set
    for i in range(len(currentTrainSubjInfo)):
        currentTrainSubj = currentTrainSubjInfo[i, 0]
        currentTrainPar = currentTrainSubjInfo[i, 1]
        currentTrainIdx = np.where((subjInfo[:, 0] == currentTrainSubj) & (subjInfo[:, 1] == currentTrainPar))[0]
        if i == 0:
            foldTrainIdx = currentTrainIdx
        else:
            foldTrainIdx = np.hstack((foldTrainIdx, currentTrainIdx))

    pre_y_train = Y[foldTrainIdx,:]
    pre_X_train = [X[idx] for idx in foldTrainIdx]

    # create testing set
    for i in range(len(currentTestSubjInfo)):
        currentTestSubj = currentTestSubjInfo[i, 0]
        currentTestPar = currentTestSubjInfo[i, 1]
        currentTestIdx = np.where((subjInfo[:, 0] == currentTestSubj) & (subjInfo[:, 1] == currentTestPar))[
            0]
        if i == 0:
            foldTestIdx = currentTestIdx  # these are already testing indices that include lines
        else:
            foldTestIdx = np.hstack((foldTestIdx, currentTestIdx))

    pre_y_test = Y[foldTestIdx,:]
    pre_X_test = [X[idx] for idx in foldTestIdx]

    # now for each line split the data with the sliding window
    trainSet_X = []
    trainSet_Y = []
    testSet_X = []
    testSet_Y = []

    for i in range(len(pre_X_train)):
        tmpX = pre_X_train[i]
        # sliding window
        a = get_strides(tmpX, winSize, overlap)
        trainSet_X.append(a)
        arrays = [pre_y_train[i] for _ in range(a.shape[0])]
        trainSet_Y.append(np.stack(arrays, axis=0))

    # plt.figure(100)
    # plt.plot(tmpX[:,0])
    # for j in range(len(a)):
    #     plt.figure(j)
    #     plt.plot(a[j][:,0])

    for i in range(len(pre_X_test)):
        tmpX = pre_X_test[i]
        # sliding window
        a = get_strides(tmpX, winSize, overlap)
        testSet_X.append(a)
        arrays = [pre_y_test[i] for _ in range(a.shape[0])]
        testSet_Y.append(np.stack(arrays, axis=0))


    for i in range(len(testSet_X)):
        if i == 0:
            X_test = testSet_X[i]
            y_test = testSet_Y[i]
        else:
            X_test = np.vstack((X_test, testSet_X[i]))
            y_test = np.vstack((y_test, testSet_Y[i]))
    print("Finished stacking X_test")

    for i in range(len(trainSet_X)):
        if i == 0:
            X_train = trainSet_X[i]
            y_train = trainSet_Y[i]
        else:
            X_train = np.vstack((X_train, trainSet_X[i]))
            y_train = np.vstack((y_train, trainSet_Y[i]))
    print("Finished stacking X_train")
    if scale:
        # scale
        s = sklearn.preprocessing.StandardScaler()
        for i in range(X_train.shape[0]):
            X_train[i] = s.fit_transform(X_train[i])

        for i in range(X_test.shape[0]):
            X_test[i] = s.fit_transform(X_test[i])
    return(X_train,y_train,X_test, y_test)

# filesArray -- array of the subject files with the kinematics data
# labelDf -- dataframe where all the labels are stored
# labelName -- the name of the label we want to use for classification
# applyFilter -- boolean value if we want to clean the data with median filter
# MED_FILTER_WIN -- the length of median filter
def load_multilabel_kinematics_data(datapath, labelDf, applyFilter, MED_FILTER_WIN = 20):
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []
    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*.csv")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)

    X = []
    Y = []
    subjInfo = []
    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[2])
            repNum = int(currentFile.split('/')[-1].split('_')[3][-1])
        df = pd.read_csv(currentFile)
        colsToRemove = ['handId', 'sec', 'min', 'hour', 'lifetimeOfThisHandObject', 'confidence']
        df = df.drop(colsToRemove, axis=1)
        featuresNames = df.columns.to_numpy()
        tmpX = df[featuresNames].to_numpy().astype(np.float32)
        tmpX = tmpX[2:-2, :]

        tmpDf = labelDf.set_index(['subj', 'rep'])
        label1 = tmpDf.at[(subjNum, repNum), "sub_id"].astype(int)

        Y.append(label1)

        w = np.zeros(shape=tmpX.shape)
        # apply median filter
        if applyFilter:
            for j in range(tmpX.shape[1]):
                w[:, j] = median_filter(tmpX[:, j], MED_FILTER_WIN)
            X.append(w)
        else:
            X.append(tmpX)
        subjInfo.append([subjNum, repNum, label1])
    return(X,Y, subjInfo)


# downsample EMG data
# X original EMG series
# Y EMG labels
# main subject info -- fail that contains information about sibject paragraph and legibility score
# output_fs -- final sampling rate
def downsample_emg(X,Y,mainSubjInfo, output_fs):
    new_X=[]
    new_Y = []
    # resample
    Fs =1260
    scale = output_fs / Fs

    zeroLengthSamplesIds=[]
    for i in range(len(X)):
        if(len(X[i])>0):
            n = round(len(X[i]) * scale) # calculate new length of sample
            X1=signal.resample(X[i], n) # resample
            new_X.append(X1)
            new_Y.append(Y[i])
        else:
            print(i)
            zeroLengthSamplesIds.append(i)
    # discarded all paragraphs above
    for i in zeroLengthSamplesIds:
        del mainSubjInfo[i]
    return(new_X,new_Y,mainSubjInfo)


# for reading EMG data

def read_emg(dfExpertLabels):
    # load experts labels data
    #dfExpertLabels = pd.read_csv(path_variables.EXPERT_LABELS_PATH)

    # load the main data
    datapath = path_variables.KINEMATICS_RAW_DATA
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []

    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*.csv")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)


    subjInfo = []
    X_t = []
    main_sr_array = []

    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[2])
            repNum = int(currentFile.split('/')[-1].split('_')[3][-1])

        tmpDf = dfExpertLabels.set_index(['subj', 'rep'])
        legScore = tmpDf.at[(subjNum, repNum), 'legibility_score'].astype(int)

        df = pd.read_csv(currentFile)
        colsToRemove = ['handId', 'lifetimeOfThisHandObject', 'confidence']
        df = df.drop(colsToRemove, axis=1)
        # get a list of columns
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('sec')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('min')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('hour')))
        df = df.loc[:, cols]
        df["time"] = ""
        tmpTimeArray = []
        for k in range(df.shape[0]):
            tmpTimeArray.append(str(df["hour"][k]) + ":" + str(df["min"][k]) + ":" + str(df["sec"][k]))
            h, m, s = tmpTimeArray[k].split(':')
            tmpTimeArray[k] = float(
                datetime.timedelta(hours=float(h), minutes=float(m), seconds=float(s)).total_seconds())

        df["time"] = tmpTimeArray
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('time')))
        df = df.loc[:, cols]
        df["time"] = df["time"] - df["time"][0]
        sr_array = []
        s = 0
        f_count = 0
        for k in range(len(df)):
            if (df["time"][k] - df["time"][s] > 1):
                sr_array.append(f_count)
                s = k
                f_count = 0
            else:
                f_count += 1
        main_sr_array.append(sr_array)
        X_t.append(df)
        subjInfo.append([subjNum, repNum, legScore])

    X = []
    Y = []
    mainSubjInfo = []
    # with moving mean filter
    for parId in range(len(X_t)):
        t = X_t[parId].iloc[:, 0]
        # read emg data
        emg_file = path_variables.NEW_EMG_PREPROCESSED_DATA + 'subj_' + str(subjInfo[parId][0]) + "_par_" + str(
            subjInfo[parId][1]) + ".csv"
        df_emg = pd.read_csv(emg_file)
        df_emg_tmp = df_emg.iloc[np.where((df_emg.iloc[:, 0] > t[0]) & (df_emg.iloc[:, 0] < t[len(t)-1]))]
        X.append(df_emg_tmp.to_numpy()[:, 1:])
        Y.append(subjInfo[parId][2])

        mainSubjInfo.append([subjInfo[parId][0], subjInfo[parId][1]])
    return(X, Y, mainSubjInfo, subjInfo)

def read_multilabel_emg():
    # load experts labels data
    dfExpertLabels = pd.read_csv("avg_legibility_score.csv")

    # load the main data
    datapath = path_variables.KINEMATICS_RAW_DATA
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []

    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*.csv")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)


    subjInfo = []
    X_t = []
    main_sr_array = []

    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[2])
            repNum = int(currentFile.split('/')[-1].split('_')[3][-1])


        tmpDf = dfExpertLabels.set_index(['subj', 'rep'])
        label1 = tmpDf.at[(subjNum, repNum), "legibility_score_1"].astype(int)
        label2 = tmpDf.at[(subjNum, repNum), "legibility_score_2"].astype(int)
        label3 = tmpDf.at[(subjNum, repNum), "legibility_score_3"].astype(int)

        df = pd.read_csv(currentFile)
        colsToRemove = ['handId', 'lifetimeOfThisHandObject', 'confidence']
        df = df.drop(colsToRemove, axis=1)
        # get a list of columns
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('sec')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('min')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('hour')))
        df = df.loc[:, cols]
        df["time"] = ""
        tmpTimeArray = []
        for k in range(df.shape[0]):
            tmpTimeArray.append(str(df["hour"][k]) + ":" + str(df["min"][k]) + ":" + str(df["sec"][k]))
            h, m, s = tmpTimeArray[k].split(':')
            tmpTimeArray[k] = float(
                datetime.timedelta(hours=float(h), minutes=float(m), seconds=float(s)).total_seconds())

        df["time"] = tmpTimeArray
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('time')))
        df = df.loc[:, cols]
        df["time"] = df["time"] - df["time"][0]
        sr_array = []
        s = 0
        f_count = 0
        for k in range(len(df)):
            if (df["time"][k] - df["time"][s] > 1):
                sr_array.append(f_count)
                s = k
                f_count = 0
            else:
                f_count += 1
        main_sr_array.append(sr_array)
        X_t.append(df)
        subjInfo.append([subjNum, repNum, label1,label2,label3])

    X = []
    Y = []
    mainSubjInfo = []
    # with moving mean filter
    for parId in range(len(X_t)):
        t = X_t[parId].iloc[:, 0]
        # read emg data
        emg_file = path_variables.NEW_EMG_PREPROCESSED_DATA + 'subj_' + str(subjInfo[parId][0]) + "_par_" + str(
            subjInfo[parId][1]) + ".csv"
        df_emg = pd.read_csv(emg_file)
        df_emg_tmp = df_emg.iloc[np.where((df_emg.iloc[:, 0] > t[0]) & (df_emg.iloc[:, 0] < t[len(t)-1]))]
        X.append(df_emg_tmp.to_numpy()[:, 1:])
        Y.append(subjInfo[parId][2:])

        mainSubjInfo.append([subjInfo[parId][0], subjInfo[parId][1]])
    return(X, Y, mainSubjInfo, subjInfo)

# for segmenting EMG data into lines
# gradThreshold  threshold for gradient cutoff, default 1.2
# saveToFile -- saves into pickle
def segment_emg(gradThreshold = 1.2, saveToFile = True):
    # load experts labels data
    dfExpertLabels = pd.read_csv(path_variables.EXPERT_LABELS_PATH)

    # load the main data
    datapath = path_variables.KINEMATICS_RAW_DATA
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []

    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*.csv")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)

    # first calculate average sampling rate
    subjInfo = []
    X_t = []
    main_sr_array = []

    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[2])
            repNum = int(currentFile.split('/')[-1].split('_')[3][-1])

        tmpDf = dfExpertLabels.set_index(['subj', 'rep'])
        legScore = tmpDf.at[(subjNum, repNum), 'legibility_score'].astype(int)

        df = pd.read_csv(currentFile)
        colsToRemove = ['handId', 'lifetimeOfThisHandObject', 'confidence']
        df = df.drop(colsToRemove, axis=1)
        # get a list of columns
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('sec')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('min')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('hour')))
        df = df.loc[:, cols]
        df["time"] = ""
        tmpTimeArray = []
        for k in range(df.shape[0]):
            tmpTimeArray.append(str(df["hour"][k]) + ":" + str(df["min"][k]) + ":" + str(df["sec"][k]))
            h, m, s = tmpTimeArray[k].split(':')
            tmpTimeArray[k] = float(
                datetime.timedelta(hours=float(h), minutes=float(m), seconds=float(s)).total_seconds())

        df["time"] = tmpTimeArray
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('time')))
        df = df.loc[:, cols]
        df["time"] = df["time"] - df["time"][0]
        sr_array = []
        s = 0
        f_count = 0
        for k in range(len(df)):
            time0 = df["time"][s]
            if (df["time"][k] - df["time"][s] > 1):
                sr_array.append(f_count)
                s = k
                f_count = 0
            else:
                f_count += 1
        main_sr_array.append(sr_array)
        tmpX = df.iloc[:, 4:].to_numpy(dtype=np.float32)
        X_t.append(df)
        subjInfo.append([subjNum, repNum, legScore])


    main_idx_array = []
    tmp_timestamps_array = []
    X = []
    Y = []
    mainSubjInfo = []
    # with moving mean filter
    for parId in range(len(X_t)):
        t = X_t[parId].iloc[:, 0]
        x = X_t[parId].iloc[:, 7]

        # number of rows
        peaks_x, peaks_x_prop = find_peaks(x, prominence=0.5)
        numRows = peaks_x.shape[0]

        # split by rows using gradient
        b = np.gradient(x)
        tmp_idx_array = []
        tmp_time_array = []
        idx = np.where(abs(b) > gradThreshold)[0]
        k = 0
        while True:
            if b[idx[k]] < 0 and k >= 2:
                if b[idx[k + 1]] > 0:
                    idx = np.delete(idx, [k - 2, k - 1, k, k + 1])
                    k = k - 1
                else:
                    idx = np.delete(idx, [k - 1, k, k + 1])
                    k = k - 2
            else:
                k = k + 1
            if k == len(idx) - 1:
                break

        if idx[0] > 20:
            tmp_idx_array.append([0, idx[0]])
            tmp_time_array.append([t[0], t[idx[0]]])
        for j in range(len(idx) - 1):
            if idx[j + 1] - idx[j] > 1:
                if idx[j + 1] - idx[j] < 120:
                    if len(tmp_idx_array) > 0:
                        tmp_idx_array[-1][-1] = idx[j + 1]
                else:
                    tmp_idx_array.append([idx[j], idx[j + 1]])
                    tmp_time_array.append([t[idx[j]], t[idx[j + 1]]])
        tmp_idx_array.append([idx[-1], len(x) - 1])
        tmp_time_array.append([t[idx[-1]], t[len(x) - 1]])
        print("number of rows " + str(numRows) + "   " + str(len(tmp_idx_array)))
        # read emg data
        emg_file = path_variables.NEW_EMG_PREPROCESSED_DATA + 'subj_' + str(subjInfo[parId][0]) + "_par_" + str(
            subjInfo[parId][1]) + ".csv"
        df_emg = pd.read_csv(emg_file)
        l = 0
        for current_idx in range(len(tmp_idx_array)):
            time_interval = tmp_time_array[current_idx]
            df_emg_tmp = df_emg.iloc[
                np.where((df_emg.iloc[:, 0] > time_interval[0]) & (df_emg.iloc[:, 0] < time_interval[1]))]
            X.append(df_emg_tmp.to_numpy()[:, 1:])
            Y.append(subjInfo[parId][2])
            l = l + 1
            mainSubjInfo.append([subjInfo[parId][0], subjInfo[parId][1], l])
    if(saveToFile):
        with open(path_variables.NEW_EMG_SLICED_DATA + '/emg_all_lines_labels.pickle',
                  'wb') as handle:
            pickle.dump((X, Y, mainSubjInfo, subjInfo), handle)
    return(X, Y, mainSubjInfo, subjInfo)

# for segmenting EMG data into lines
# gradThreshold  threshold for gradient cutoff, default 1.2
# saveToFile -- saves into pickle
def segment_multilabel_emg(gradThreshold = 1.2, saveToFile = True):
    # load experts labels data
    dfExpertLabels = pd.read_csv("avg_legibility_score.csv")

    # load the main data
    datapath = path_variables.KINEMATICS_RAW_DATA
    subjArray = ["subj_" + str(i) for i in range(1, 51)]
    filesArray = []

    for subj in subjArray:
        tmpFilesFound = glob.glob(datapath + subj + "/*.csv")
        tmpFilesFound.sort()
        filesArray.extend(tmpFilesFound)

    # first calculate average sampling rate
    subjInfo = []
    X_t = []
    main_sr_array = []

    for currentFile in filesArray:
        if platform == "darwin":
            subjNum = int(currentFile.split('/')[-1].split('_')[1])
            repNum = int(currentFile.split('/')[-1].split('_')[2][-1])
        elif platform == "win32":
            subjNum = int(currentFile.split('/')[-1].split('_')[2])
            repNum = int(currentFile.split('/')[-1].split('_')[3][-1])

        tmpDf = dfExpertLabels.set_index(['subj', 'rep'])

        label1 = tmpDf.at[(subjNum, repNum), "legibility_score_1"].astype(int)
        label2 = tmpDf.at[(subjNum, repNum), "legibility_score_2"].astype(int)
        label3 = tmpDf.at[(subjNum, repNum), "legibility_score_3"].astype(int)

        df = pd.read_csv(currentFile)
        colsToRemove = ['handId', 'lifetimeOfThisHandObject', 'confidence']
        df = df.drop(colsToRemove, axis=1)
        # get a list of columns
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('sec')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('min')))
        df = df.loc[:, cols]
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('hour')))
        df = df.loc[:, cols]
        df["time"] = ""
        tmpTimeArray = []
        for k in range(df.shape[0]):
            tmpTimeArray.append(str(df["hour"][k]) + ":" + str(df["min"][k]) + ":" + str(df["sec"][k]))
            h, m, s = tmpTimeArray[k].split(':')
            tmpTimeArray[k] = float(
                datetime.timedelta(hours=float(h), minutes=float(m), seconds=float(s)).total_seconds())

        df["time"] = tmpTimeArray
        cols = list(df)
        # move the column to head of list using index, pop and insert
        cols.insert(0, cols.pop(cols.index('time')))
        df = df.loc[:, cols]
        df["time"] = df["time"] - df["time"][0]
        sr_array = []
        s = 0
        f_count = 0
        for k in range(len(df)):
            time0 = df["time"][s]
            if (df["time"][k] - df["time"][s] > 1):
                sr_array.append(f_count)
                s = k
                f_count = 0
            else:
                f_count += 1
        main_sr_array.append(sr_array)
        tmpX = df.iloc[:, 4:].to_numpy(dtype=np.float32)
        X_t.append(df)
        subjInfo.append([subjNum, repNum, label1,label2,label3])


    main_idx_array = []
    tmp_timestamps_array = []
    X = []
    Y = []
    mainSubjInfo = []
    # with moving mean filter
    for parId in range(len(X_t)):
        t = X_t[parId].iloc[:, 0]
        x = X_t[parId].iloc[:, 7]

        # number of rows
        peaks_x, peaks_x_prop = find_peaks(x, prominence=0.5)
        numRows = peaks_x.shape[0]

        # split by rows using gradient
        b = np.gradient(x)
        tmp_idx_array = []
        tmp_time_array = []
        idx = np.where(abs(b) > gradThreshold)[0]
        k = 0
        while True:
            if b[idx[k]] < 0 and k >= 2:
                if b[idx[k + 1]] > 0:
                    idx = np.delete(idx, [k - 2, k - 1, k, k + 1])
                    k = k - 1
                else:
                    idx = np.delete(idx, [k - 1, k, k + 1])
                    k = k - 2
            else:
                k = k + 1
            if k == len(idx) - 1:
                break

        if idx[0] > 20:
            tmp_idx_array.append([0, idx[0]])
            tmp_time_array.append([t[0], t[idx[0]]])
        for j in range(len(idx) - 1):
            if idx[j + 1] - idx[j] > 1:
                if idx[j + 1] - idx[j] < 120:
                    if len(tmp_idx_array) > 0:
                        tmp_idx_array[-1][-1] = idx[j + 1]
                else:
                    tmp_idx_array.append([idx[j], idx[j + 1]])
                    tmp_time_array.append([t[idx[j]], t[idx[j + 1]]])
        tmp_idx_array.append([idx[-1], len(x) - 1])
        tmp_time_array.append([t[idx[-1]], t[len(x) - 1]])
        print("number of rows " + str(numRows) + "   " + str(len(tmp_idx_array)))
        # read emg data
        emg_file = path_variables.NEW_EMG_PREPROCESSED_DATA + 'subj_' + str(subjInfo[parId][0]) + "_par_" + str(
            subjInfo[parId][1]) + ".csv"
        df_emg = pd.read_csv(emg_file)
        l = 0
        for current_idx in range(len(tmp_idx_array)):
            time_interval = tmp_time_array[current_idx]
            df_emg_tmp = df_emg.iloc[
                np.where((df_emg.iloc[:, 0] > time_interval[0]) & (df_emg.iloc[:, 0] < time_interval[1]))]
            X.append(df_emg_tmp.to_numpy()[:, 1:])
            Y.append(subjInfo[parId][2:])
            l = l + 1
            mainSubjInfo.append([subjInfo[parId][0], subjInfo[parId][1], subjInfo[parId][2],subjInfo[parId][3], subjInfo[parId][4]])
    if(saveToFile):
        with open(path_variables.NEW_EMG_SLICED_DATA + '/emg_all_lines_labels.pickle',
                  'wb') as handle:
            pickle.dump((X, Y, mainSubjInfo, subjInfo), handle)
    return(X, Y, mainSubjInfo, subjInfo)
