# architecture search for stylus kinematics
import gc
import os.path
import pickle
import random
import time

import numpy as np
import pandas as pd
import sklearn
import tensorflow
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import set_session


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
    if a.shape[0] < L:
        out = np.pad(a, ((0, L - a.shape[0]), (0, 0)), 'constant', constant_values=0)
        out = np.expand_dims(out, 0)
    else:
        out = []
        for i in range(0, a.shape[0] - L + 1, L - ov):
            out.append(a[i:i + L, :])
        tmpA = np.zeros((L, a.shape[1]))
        tmpA[:L - (i + 2 * L - ov - a.shape[0]), :] = a[i + L - ov - 1:a.shape[0] - 1, :]
        out.append(tmpA)
    return np.array(out)


# split into train test for each fold
# currentTrainSubjInfo contains train indices for train set of current fold
# currentTestSubjInfo contains test indices for test set of current fold
# subjInfo contains main dataset indices before splitting
# X and Y are main dataset X and lables Y before splitting
def fold_train_test_windows(X, Y, currentTrainSubjInfo, currentTestSubjInfo, subjInfo, winSize, overlap, scale=True):
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
    X_train = np.concatenate(trainSet_X, axis=0)
    y_train = np.concatenate(trainSet_Y, axis=0)
    print("Finished stacking X_train")

    for i in range(len(pre_X_test)):
        tmpX = pre_X_test[i]
        # sliding window
        a = get_strides(tmpX, winSize, overlap)
        testSet_X.append(a)
        testSet_Y.append(np.repeat(pre_y_test[i], a.shape[0]))
    X_test = np.concatenate(testSet_X, axis=0)
    y_test = np.concatenate(testSet_Y, axis=0)
    print("Finished stacking X_test")

    if scale:
        # scale
        s = sklearn.preprocessing.StandardScaler()
        for i in range(X_train.shape[0]):
            X_train[i] = s.fit_transform(X_train[i])

        for i in range(X_test.shape[0]):
            X_test[i] = s.fit_transform(X_test[i])
    return (X_train, y_train, X_test, y_test)


def create_model(X_train, y_train, kernel_1, num_ch_1, dropout, EPOCHS, BATCH_SIZE, lr, LOSS):
    ########################################################################################################################
    # clear the memory from previous models
    ########################################################################################################################
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.backend.clear_session()
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tensorflow.compat.v1.Session(config=config)
    set_session(sess)
    ########################################################################################################################
    # define model
    ########################################################################################################################
    input_shape = X_train.shape[1:]
    input = Input(shape=input_shape)
    tcnn_block_1 = Conv1D(num_ch_1, kernel_size=kernel_1, activation='relu', padding='valid', name="tcnn_block_1")(
        input)
    activation_1 = Activation("relu")(tcnn_block_1)
    bn_1 = BatchNormalization()(activation_1)
    drop_1 = Dropout(dropout)(bn_1)
    self_att_1 = Attention(32, "CustomLayer")(drop_1)
    flat_1 = Flatten(name='Flatten')(self_att_1)
    f_out = Dense(50, activation='softmax', name="f_out")(flat_1)
    model = Model(inputs=input, outputs=f_out)

    opt=tf.keras.optimizers.Adadelta(learning_rate=lr)
    model.compile(optimizer=opt,  # tf.keras.optimizers.Adadelta(),#adam,
                  loss=LOSS,
                  metrics=['accuracy'])

    model.summary()

    ########################################################################################################################
    # train model
    ########################################################################################################################
    hist = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=2
    )


    ########################################################################################################################
    # evaluate model
    ########################################################################################################################
    yhat = model.predict(X_test)
    sess.close()
    ops.reset_default_graph()
    return yhat


#################################################################################################################
#
# parameters
#
#################################################################################################################
EPOCHS = 100
BATCH_SIZE = 128
lr = 1e-3
nfold = 5
dropout = 0.3
LOSS = tf.keras.losses.categorical_crossentropy


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super(Attention, self).__init__(name=name)
        self.units = units
        self.W = Dense(units)
        self.V = Dense(1)
        super(Attention, self).__init__(**kwargs)

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"units": self.units, })
        return config

    def call(self, inputs):
        # Compute attention scores
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


#################################################################################################################
#
# load the data
#
#################################################################################################################
# load main data
# VERY IMPORTANT
# do not filter
with open('subj_labels.pickle','rb') as handle:
   (X, Y, subjInfo) = pickle.load(handle)

#################################################################################################################
#
# Use Expert  labels
#
#################################################################################################################
num_classes = np.unique(Y).shape[0]
#################################################################################################################
#
# split into test -train
#
#################################################################################################################
# here we want to prepare fold so that testing and training set for each fold contain lines
# of different paragraph (even for the same subject)
# and later these folds will be modified to adapt to the lines for each subject's paragraph
# i.e. we want the lines from same paragraph remain in either testing or in traing sets,
# but not one line from the same paragraph is in training and another line is in testing set
X_subj_par = subjInfo[:, 1:]
Y_subj_par = subjInfo[:, 0]  # stratify by subjects

RANDOM_STATE = 42
ros = RandomOverSampler(random_state=RANDOM_STATE)

winSize = 896 # something in between 64 and 1774 (max 1728) median 896
ovr = 0.9

for num_ch_1 in [ 8, 16, 32, 64, 128,  256, 512, 1024]:
    for kernel_1 in [3, 5, 10, 25, 50, 75, 100, 250, 500, 650, 700, 750, 800]:
        MAIN_FILENAME ="AUTHENTICATION_ARCHITECTURE_STYLUS/authen_ch1_" + str(num_ch_1) + "_k1_" + str(kernel_1)
        report_filename = MAIN_FILENAME + ".csv"
        # Create cross-validation object
        kf = StratifiedKFold(nfold, shuffle=True, random_state=RANDOM_STATE)
        folds = list(enumerate(kf.split(X_subj_par, Y_subj_par)))
        for ifold in range(5):
            gc.enable()
            overlap = int(winSize * ovr)
            #################################################################################################################
            #
            # create a dataframe for storing the reports
            #
            #################################################################################################################
            if os.path.isfile(report_filename):
                reports_df = pd.read_csv(report_filename, index_col=False)
            else:
                reports_df = pd.DataFrame(
                    columns=['WINDOW', 'OVERLAP', "KERNEL_1", "N_CH_1", 'EPOCHS', 'LEARNING_RATE', 'BATCH_SIZE',
                             'DROPOUT', 'ACC', 'PRECISION',
                             'RECALL', 'F1', 'FOLD', 'AVG_ACC', 'AVG_PREC', 'AVG_RECALL', 'AVG_F1', 'TIME'])


            # Run the cross-validation
            reportsPerFold = []

            start = time.time()
            print('running fold = %d' % ifold)
            foldIdx = folds[ifold][0]
            foldTrainIdx = folds[ifold][1][0]
            foldTestIdx = folds[ifold][1][1]
            currentTrainSubjInfo = subjInfo[foldTrainIdx, :]
            currentTestSubjInfo = subjInfo[foldTestIdx, :]

            ########################################################################################################################
            # to balance the test
            ########################################################################################################################
            # find missing subjects
            labels_tr, counts_tr = np.unique(currentTrainSubjInfo[:, 0], return_counts=True)
            labels_ts, counts_ts = np.unique(currentTestSubjInfo[:, 0], return_counts=True)
            missing_test = np.setdiff1d(labels_tr, labels_ts)
            if len(missing_test!=0):
                removed_indices = []
                for tr_i in range(missing_test.shape[0]):
                    missing_test_idx = np.where(currentTrainSubjInfo[:, 0] == missing_test[tr_i])[0]
                    if (missing_test_idx.shape[0] > 1):
                        random.seed(RANDOM_STATE)
                        idx_to_remove = missing_test_idx[random.randint(0, missing_test_idx.shape[0] - 1)]
                        currentTestSubjInfo = np.vstack((currentTestSubjInfo, currentTrainSubjInfo[idx_to_remove, :]))
                        removed_indices.append(idx_to_remove)
                currentTrainSubjInfo = np.delete(currentTrainSubjInfo, removed_indices, 0)
            else:
                currentTrainSubjInfo = subjInfo[foldTrainIdx, :]
            ########################################################################################################################
            # get train and test sets split into windows
            ########################################################################################################################
            (X_train, y_train, X_test, y_test) = fold_train_test_windows(X, Y, currentTrainSubjInfo,
                                                                         currentTestSubjInfo, subjInfo,
                                                                         winSize, overlap, scale=True)
            X_train = X_train[:, :, :7]
            X_test = X_test[:, :, :7]
            n_features = X_train.shape[-1]
            y_train_orig = y_train
            y_test_orig = y_test

            a = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
            X1_train, y1_train = ros.fit_resample(a, y_train)
            X1_train = X1_train.reshape(X1_train.shape[0], winSize, n_features)
            del a

            # Create an instance of One-hot-encoder
            label_binarizer = LabelBinarizer()
            y_train = label_binarizer.fit_transform(y1_train)
            y_test = label_binarizer.fit_transform(y_test)

            yhat = create_model(X1_train, y_train, kernel_1, num_ch_1, dropout, EPOCHS, BATCH_SIZE, lr, LOSS)
            del X_test, X1_train

            gc.collect()
            yhat1 = label_binarizer.inverse_transform(yhat)
            y_test = label_binarizer.inverse_transform(y_test)

            ########################################################################################################################
            # get reports on model performance
            ########################################################################################################################
            report = classification_report(y_test, yhat1, output_dict=True, labels=np.unique(y_test))
            # record variables per fold
            reportsPerFold.append(report)

            tmp_avg_precision = []
            tmp_avg_recall = []
            tmp_avg_F1 = []
            for cl in np.unique(y_train_orig):
                tmp_avg_precision.append(report[str(cl)]["precision"])
                tmp_avg_recall.append(report[str(cl)]["recall"])
                tmp_avg_F1.append(report[str(cl)]["f1-score"])

            avg_precision = np.mean(tmp_avg_precision)
            avg_recall = np.mean(tmp_avg_recall)
            avg_F1 = np.mean(tmp_avg_F1)

            # print mean values
            avgAccuracy = []
            avgRecall = []
            avgPrecision = []
            avgF1 = []
            for rep in reportsPerFold:
                print(pd.DataFrame(rep))
                avgAccuracy.append(rep["accuracy"])
                for i in np.unique(y_train_orig):
                    avgRecall.append(rep[str(i)]["recall"])
                    avgPrecision.append(rep[str(i)]["precision"])
                    avgF1.append(rep[str(i)]["f1-score"])

            print("Average accuracy: %.3f\nAverage precision:  %.3f\nAverage recall:  %.3f\nAverage F1 : %.3f\n " % (
                np.mean(avgAccuracy), np.mean(avgPrecision), np.mean(avgRecall), np.mean(avgF1)))

            end = time.time()
            print("Time elapsed ", str(end - start))
            reports_df.loc[len(reports_df)] = [winSize, ovr, kernel_1, num_ch_1, EPOCHS, lr, BATCH_SIZE, dropout,
                                               report["accuracy"],
                                               avg_precision,
                                               avg_recall, avg_F1, ifold, np.mean(avgAccuracy), np.mean(avgPrecision),
                                               np.mean(avgRecall), np.mean(avgF1), end - start]
            reports_df.infer_objects().dtypes
            reports_df.to_csv(report_filename, index=False)
            gc.collect()
        gc.collect()
    gc.collect()