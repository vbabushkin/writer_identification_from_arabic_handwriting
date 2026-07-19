# ablation experiments normalizing stylus kinematics coordinates wrt tablet
import gc
import os
import os.path
import pickle
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.framework import ops
from tensorflow.python.keras.backend import set_session

plt.style.use('default')
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.ticker import MaxNLocator

# Modern TF2 GPU Configuration & Immediate Initialization
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # FORCE IMMEDIATE INITIALIZATION: This registers the python process
        # inside nvidia-smi immediately so the HPC reaper won't kill your job.
        with tf.device('/GPU:0'):
            _ = tf.zeros([1])
        print("GPU Context initialized successfully at startup.")
    except RuntimeError as e:
        print(f"GPU initialization error: {e}")

features_7 = ["pressure",
                "azimuth",
                "altitude",
                "x",
                "y",
                "z",
                "proximity",
                ]

# --- DEFINE COORDINATE GROUPS & INDEX MAPS ---
# Find Stylus coordinates
stylus_x_idx = features_7.index("x")
stylus_y_idx = features_7.index("y")


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


def fold_train_test_windows(X, Y, currentTrainSubjInfo, currentTestSubjInfo, subjInfo, winSize, overlap, scale=True):
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

    for i in range(len(currentTestSubjInfo)):
        currentTestSubj = currentTestSubjInfo[i, 0]
        currentTestPar = currentTestSubjInfo[i, 1]
        currentTestIdx = np.where((subjInfo[:, 0] == currentTestSubj) & (subjInfo[:, 1] == currentTestPar))[0]
        if i == 0:
            foldTestIdx = currentTestIdx
        else:
            foldTestIdx = np.hstack((foldTestIdx, currentTestIdx))

    pre_y_test = Y[foldTestIdx]
    pre_X_test = [X[idx] for idx in foldTestIdx]

    trainSet_X = []
    trainSet_Y = []
    testSet_X = []
    testSet_Y = []

    for i in range(len(pre_X_train)):
        tmpX = pre_X_train[i]
        a = get_strides(tmpX, winSize, overlap)
        trainSet_X.append(a)
        trainSet_Y.append(np.repeat(pre_y_train[i], a.shape[0]))
    X_train = np.concatenate(trainSet_X, axis=0)
    y_train = np.concatenate(trainSet_Y, axis=0)
    print("Finished stacking X_train")

    for i in range(len(pre_X_test)):
        tmpX = pre_X_test[i]
        a = get_strides(tmpX, winSize, overlap)
        testSet_X.append(a)
        testSet_Y.append(np.repeat(pre_y_test[i], a.shape[0]))
    X_test = np.concatenate(testSet_X, axis=0)
    y_test = np.concatenate(testSet_Y, axis=0)
    print("Finished stacking X_test")

    if scale:
        s = sklearn.preprocessing.StandardScaler()
        for i in range(X_train.shape[0]):
            X_train[i] = s.fit_transform(X_train[i])

        for i in range(X_test.shape[0]):
            X_test[i] = s.fit_transform(X_test[i])
    return (X_train, y_train, X_test, y_test)


def create_model(train_dataset, val_dataset, kernel_1, num_ch_1, dropout, EPOCHS, lr, LOSS, MAIN_FILENAME, ifold):
    tf.random.set_seed(RANDOM_STATE)

    # Clears memory and resets the Keras graph from previous folds
    tf.keras.backend.clear_session()

    # FORCE COMPUTATIONS TO GPU
    with tf.device('/GPU:0'):
        # Get input shape dynamically from the dataset's element spec
        # val_dataset.element_spec is a tuple of (X_spec, y_spec)
        input_shape = val_dataset.element_spec[0].shape[1:]
        input_layer = Input(shape=input_shape)

        tcnn_block_1 = Conv1D(num_ch_1, kernel_size=kernel_1, activation='relu', padding='valid', name="tcnn_block_1")(
            input_layer)
        activation_1 = Activation("relu")(tcnn_block_1)
        bn_1 = BatchNormalization()(activation_1)

        drop_1 = Dropout(dropout)(bn_1)
        self_att_1 = Attention(32, "CustomLayer")(drop_1)
        flat_1 = Flatten(name='Flatten')(self_att_1)
        dense_2 = Dense(512, activation='relu', name='dense_2')(flat_1)
        bn_4 = BatchNormalization()(dense_2)
        dense_3 = Dense(256, activation='relu', name='dense_3')(bn_4)
        bn_5 = BatchNormalization()(dense_3)
        f_out = Dense(50, activation='softmax', name="f_out")(bn_5)
        model = Model(inputs=input_layer, outputs=f_out)

        opt = tf.keras.optimizers.Adadelta(learning_rate=lr)
        model.compile(optimizer=opt, loss=LOSS, metrics=['accuracy'])
        model.summary()

        hist = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            verbose=2
        )

        # Predict using the highly optimized tf.data.Dataset pipeline
        yhat = model.predict(val_dataset)

    tf.keras.models.save_model(model, f"{MAIN_FILENAME}_model_fold_{ifold}.h5")

    return yhat


#################################################################################################################
# parameters
#################################################################################################################
EPOCHS = 100
BATCH_SIZE = 128
lr = 1e-3
nfold = 5
dropout = 0.3
LOSS = tf.keras.losses.categorical_crossentropy
RANDOM_STATE = 9#42 for parameter search, 9 for calculating Shapley values
winSize = 1152
ovr = 0.9
num_ch_1 = 2048
kernel_1 = 100
RESULTS_PATH = "/CODE/AUTH_STYLUS/"
MAIN_FILENAME = RESULTS_PATH + "abl_auth_ch1_" + str(num_ch_1) + "_k1_" + str(kernel_1) + "_w_" + str(
    winSize) + "_stylus_normalized_parallelized"

report_filename = MAIN_FILENAME + ".csv"


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
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


# --- READ SINGLE TARGET FOLD VIA SLURM ARRAY ---
if "SLURM_ARRAY_TASK_ID" in os.environ:
    ifold = int(os.environ["SLURM_ARRAY_TASK_ID"])
else:
    raise ValueError("This script must be executed within a SLURM Job Array environment.")

# Load main data
with open('subj_labels.pickle', 'rb') as handle:
    (X, Y, subjInfo) = pickle.load(handle)

num_classes = np.unique(Y).shape[0]
X_subj_par = subjInfo[:, 1:]
Y_subj_par = subjInfo[:, 0]

ros = RandomOverSampler(random_state=RANDOM_STATE)

if os.path.isfile(MAIN_FILENAME + '_folds.pickle'):
    with open(MAIN_FILENAME + '_folds.pickle', 'rb') as handle:
        folds = pickle.load(handle)
else:
    kf = StratifiedKFold(nfold, shuffle=True, random_state=RANDOM_STATE)
    folds = list(enumerate(kf.split(X_subj_par, Y_subj_par)))
    # Only let task index 0 dump the cross-validation partition map to prevent IO lock issues
    if ifold == 0:
        with open(MAIN_FILENAME + '_folds.pickle', 'wb') as handle:
            pickle.dump(folds, handle)

# --- EXECUTE INDIVIDUAL FOLD PROCESSING ---
gc.enable()
overlap = int(winSize * ovr)

reportsPerFold = []
start = time.time()
print('Running parallelized SLURM fold task = %d' % ifold)

# Double check GPU visibility right before execution
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("CITICAL WARNING: TensorFlow cannot see any GPUs! This job is running on CPU.")
else:
    print(f"SUCCESS: TensorFlow detects {len(gpus)} GPU(s).")
overlap = int(winSize * ovr)

print('Running parallelized SLURM fold task = %d' % ifold)

foldIdx = folds[ifold][0]
foldTrainIdx = folds[ifold][1][0]
foldTestIdx = folds[ifold][1][1]
currentTrainSubjInfo = subjInfo[foldTrainIdx, :]
currentTestSubjInfo = subjInfo[foldTestIdx, :]

labels_tr, counts_tr = np.unique(currentTrainSubjInfo[:, 0], return_counts=True)
labels_ts, counts_ts = np.unique(currentTestSubjInfo[:, 0], return_counts=True)
missing_test = np.setdiff1d(labels_tr, labels_ts)
if len(missing_test) != 0:
    removed_indices = []
    for tr_i in range(missing_test.shape[0]):
        missing_test_idx = np.where(currentTrainSubjInfo[:, 0] == missing_test[tr_i])[0]
        if (missing_test_idx.shape[0] > 1):
            random.seed(RANDOM_STATE)
            idx_to_remove = missing_test_idx[random.randint(0, missing_test_idx.shape[0] - 1)]
            currentTestSubjInfo = np.vstack((currentTestSubjInfo, currentTrainSubjInfo[idx_to_remove, :]))
            removed_indices.append(idx_to_remove)
    currentTrainSubjInfo = np.delete(currentTrainSubjInfo, removed_indices, 0)

(X_train, y_train, X_test, y_test) = fold_train_test_windows(X, Y, currentTrainSubjInfo, currentTestSubjInfo, subjInfo,
                                                             winSize, overlap, scale=True)
print(X_train.shape)
print(X_test.shape)

# --- Tablet / Writing Surface Coordinate Normalization (For Stylus Coordinates) ---
# Min-max scales stylus coordinates to [0, 1] relative to the active writing window boundaries.
for i in range(len(X_test)):
    X_test[i, :, stylus_x_idx] = (X_test[i, :, stylus_x_idx] - X_test[i, :, stylus_x_idx].min()) / (
                X_test[i, :, stylus_x_idx].max() - X_test[i, :, stylus_x_idx].min() + 1e-8)
    X_test[i, :, stylus_y_idx] = (X_test[i, :, stylus_y_idx] - X_test[i, :, stylus_y_idx].min()) / (
                X_test[i, :, stylus_y_idx].max() - X_test[i, :, stylus_y_idx].min() + 1e-8)

for i in range(len(X_train)):
    X_train[i, :, stylus_x_idx] = (X_train[i, :, stylus_x_idx] - X_train[i, :, stylus_x_idx].min()) / (
            X_train[i, :, stylus_x_idx].max() - X_train[i, :, stylus_x_idx].min() + 1e-8)
    X_train[i, :, stylus_y_idx] = (X_train[i, :, stylus_y_idx] - X_train[i, :, stylus_y_idx].min()) / (
            X_train[i, :, stylus_y_idx].max() - X_train[i, :, stylus_y_idx].min() + 1e-8)



print(X_train.shape)
print(X_test.shape)
n_features = X_train.shape[-1]
y_train_orig = y_train
y_test_orig = y_test

a = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X1_train, y1_train = ros.fit_resample(a, y_train)
X1_train = X1_train.reshape(X1_train.shape[0], winSize, n_features)
del a

label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y1_train)
y_test = label_binarizer.fit_transform(y_test)


train_dataset = tf.data.Dataset.from_tensor_slices((X1_train, y_train)).batch(BATCH_SIZE).prefetch(
    buffer_size=tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(
    buffer_size=tf.data.AUTOTUNE)

yhat = create_model(train_dataset, val_dataset, kernel_1, num_ch_1, dropout, EPOCHS, lr, LOSS, MAIN_FILENAME, ifold)

with open(MAIN_FILENAME + '_train_test_fold_' + str(ifold) + '.pickle', 'wb') as handle:
    pickle.dump((X_train, y_train_orig, X_test, y_test_orig), handle, protocol=4)
del X_test, X1_train
gc.collect()

yhat1 = label_binarizer.inverse_transform(yhat)
y_test = label_binarizer.inverse_transform(y_test)

report = classification_report(y_test, yhat1, output_dict=True, labels=np.unique(Y))
cm = confusion_matrix(y_test, yhat1, labels=np.unique(Y))
with open(MAIN_FILENAME + f'_cm_fold_{ifold}.pickle', 'wb') as f:
    pickle.dump(cm, f)

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
end = time.time()

# Write isolated metrics per fold into target tracking sheet safely
fold_report_df = pd.DataFrame([{
    'WINDOW': winSize, 'OVERLAP': ovr, "KERNEL_1": kernel_1, "N_CH_1": num_ch_1,
    'EPOCHS': EPOCHS, 'LEARNING_RATE': lr, 'BATCH_SIZE': BATCH_SIZE, 'DROPOUT': dropout,
    'ACC': report["accuracy"], 'PRECISION': avg_precision, 'RECALL': avg_recall, 'F1': avg_F1,
    'FOLD': ifold, 'TIME': end - start
}])

# Thread-safe appending to shared evaluation csv sheet
fold_report_df.to_csv(report_filename, mode='a', header=not os.path.exists(report_filename), index=False)
print(f"Fold {ifold} finished execution successfully in {end - start:.2f} seconds.")