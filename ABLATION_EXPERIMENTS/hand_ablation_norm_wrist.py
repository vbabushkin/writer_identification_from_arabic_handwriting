# ablation experiments normalizing hand kinematics coordinates wrt wrist
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
        


features_110 = [
    "x_thumb_dist",
    "y_thumb_dist",
    "z_thumb_dist",
    "x_thumb_intermed",
    "y_thumb_intermed",
    "z_thumb_intermed",
    "x_thumb_proxi",
    "y_thumb_proxi",
    "z_thumb_proxi",
    "x_thumb_metacarpal",
    "y_thumb_metacarpal",
    "z_thumb_metacarpal",
    "x_index_dist",
    "y_index_dist",
    "z_index_dist",
    "x_index_intermed",
    "y_index_intermed",
    "z_index_intermed",
    "x_index_proxi",
    "y_index_proxi",
    "z_index_proxi",
    "x_index_metacarpal",
    "y_index_metacarpal",
    "z_index_metacarpal",
    "x_index_metacarpal_base",
    "y_index_metacarpal_base",
    "z_index_metacarpal_base",
    "x_middle_dist",
    "y_middle_dist",
    "z_middle_dist",
    "x_middle_intermed",
    "y_middle_intermed",
    "z_middle_intermed",
    "x_middle_proxi",
    "y_middle_proxi",
    "z_middle_proxi",
    "x_middle_metacarpal",
    "y_middle_metacarpal",
    "z_middle_metacarpal",
    "x_middle_metacarpal_base",
    "y_middle_metacarpal_base",
    "z_middle_metacarpal_base",
    "x_ring_dist",
    "y_ring_dist",
    "z_ring_dist",
    "x_ring_intermed",
    "y_ring_intermed",
    "z_ring_intermed",
    "x_ring_proxi",
    "y_ring_proxi",
    "z_ring_proxi",
    "x_ring_metacarpal",
    "y_ring_metacarpal",
    "z_ring_metacarpal",
    "x_ring_metacarpal_base",
    "y_ring_metacarpal_base",
    "z_ring_metacarpal_base",
    "x_pinky_dist",
    "y_pinky_dist",
    "z_pinky_dist",
    "x_pinky_intermed",
    "y_pinky_intermed",
    "z_pinky_intermed",
    "x_pinky_proxi",
    "y_pinky_proxi",
    "z_pinky_proxi",
    "x_pinky_metacarpal",
    "y_pinky_metacarpal",
    "z_pinky_metacarpal",
    "x_pinky_metacarpal_base",
    "y_pinky_metacarpal_base",
    "z_pinky_metacarpal_base",
    "handCenter_x",
    "handCenter_y",
    "handCenter_z",
    "handSpeed_x",
    "handSpeed_y",
    "handSpeed_z",
    "handWidth",
    "pitch",
    "yaw",
    "roll",
    "handPinchStrength",
    "handPinchDistance",
    "handPinchPosition_x",
    "handPinchPosition_y",
    "handPinchPosition_z",
    "handPredictedPinchPosition_x",
    "handPredictedPinchPosition_y",
    "handPredictedPinchPosition_z",
    "handWristPosition_x",
    "handWristPosition_y",
    "handWristPosition_z",
    "handPalmNormal_x",
    "handPalmNormal_y",
    "handPalmNormal_z",
    "handGrabAngle",
    "handGrabStrength",
    "rotationOfHand_x",
    "rotationOfHand_y",
    "rotationOfHand_z",
    "rotationOfHand_w",
    "handArmLength",
    "handArmWidth",
    "elbowPosition_x",
    "elbowPosition_y",
    "elbowPosition_z",
    "handArmCenter_x",
    "handArmCenter_y",
    "handArmCenter_z"]

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
EPOCHS = 200
BATCH_SIZE = 128
lr = 1e-3
nfold = 5
dropout = 0.3
LOSS = tf.keras.losses.categorical_crossentropy
RANDOM_STATE = 9#42 for parameter search, 9 for calculating Shapley values
winSize = 1344
ovr = 0.9
num_ch_1 = 1024
kernel_1 = 100
RESULTS_PATH = "/CODE/AUTH_HAND/"
num_classes = 50
MAIN_FILENAME = RESULTS_PATH + "abl_auth_ch1_" + str(num_ch_1) + "_k1_" + str(kernel_1) + "_w_" + str(winSize) + "_hand_norm_wrist_parallelized_200"

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

excluded_keywords = ["speed", "normal", "rotation", "width", "length", "angle", "strength", "distance", "altitude", "azimuth", "pitch", "yaw", "roll"]
hand_x_features = [f for f in features_110 if (f.startswith("x_") or f.endswith("_x")) and f != "x" and not any(k in f.lower() for k in excluded_keywords)]
hand_y_features = [f for f in features_110 if (f.startswith("y_") or f.endswith("_y")) and f != "y" and not any(k in f.lower() for k in excluded_keywords)]
hand_z_features = [f for f in features_110 if (f.startswith("z_") or f.endswith("_z")) and f != "z" and not any(k in f.lower() for k in excluded_keywords)]

hand_x_indices = [features_110.index(f) for f in hand_x_features]
hand_y_indices = [features_110.index(f) for f in hand_y_features]
hand_z_indices = [features_110.index(f) for f in hand_z_features]

wrist_x_idx = features_110.index("handWristPosition_x")
wrist_y_idx = features_110.index("handWristPosition_y")
wrist_z_idx = features_110.index("handWristPosition_z")



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
X_train = X_train[:, :, 7:]
X_test = X_test[:, :, 7:]

# --- Wrist-Relative Coordinate Normalization ---
X_train[:, :, hand_x_indices] = X_train[:, :, hand_x_indices] - X_train[:, :, wrist_x_idx:wrist_x_idx+1]
X_train[:, :, hand_y_indices] = X_train[:, :, hand_y_indices] - X_train[:, :, wrist_y_idx:wrist_y_idx+1]
X_train[:, :, hand_z_indices] = X_train[:, :, hand_z_indices] - X_train[:, :, wrist_z_idx:wrist_z_idx+1]

X_test[:, :, hand_x_indices] = X_test[:, :, hand_x_indices] - X_test[:, :, wrist_x_idx:wrist_x_idx+1]
X_test[:, :, hand_y_indices] = X_test[:, :, hand_y_indices] - X_test[:, :, wrist_y_idx:wrist_y_idx+1]
X_test[:, :, hand_z_indices] = X_test[:, :, hand_z_indices] - X_test[:, :, wrist_z_idx:wrist_z_idx+1]

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