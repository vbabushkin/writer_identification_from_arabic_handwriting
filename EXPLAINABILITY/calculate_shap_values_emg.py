# Shapley values for EMG
import gc

import numpy as np
import shap
import tensorflow as tf

print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)
import tensorflow
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.python.keras.backend import set_session
features_emg = ["FDI","EDC","BB"]


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
# parameters
#
#################################################################################################################
n = 200  # samples from train and test to calculate Shapley values

EPOCHS = 500
BATCH_SIZE = 128
lr = 1e-2
nfold = 5
dropout = 0.3
LOSS = tf.keras.losses.categorical_crossentropy
RANDOM_STATE = 9#42 for parameter search, 9 for calculating Shapley values
winSize = 1000
ovr = 0.9
num_ch_1 = 576
kernel_1 = 10
RESULTS_PATH = "/CODE/AUTH_EMG/"
MAIN_FILENAME = "emg_auth_ch1_" + str(num_ch_1) + "_k1_" + str(kernel_1)+"_w_"+str(winSize)+"_all"

overlap = int(winSize * ovr)

shapValuesByClassLabelArr = []
mean_shap_valuesq_arr =[]
gc.enable()
for ifold in range(5):
    ########################################################################################################################
    # clear the memory from previous models
    ########################################################################################################################
    tf.random.set_seed(RANDOM_STATE)
    tf.keras.backend.clear_session()
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tensorflow.compat.v1.Session(config=config)
    set_session(sess)
    modelName = MAIN_FILENAME + "_model_fold_" + str(ifold) + ".h5"
    model = tf.keras.models.load_model(RESULTS_PATH+modelName, custom_objects={'CustomLayer': Attention})
    print(model.summary())

    with open(RESULTS_PATH+MAIN_FILENAME + '_train_test_fold_' + str(ifold) + '.pickle',
              'rb') as handle:
        (X_train, y_train, X_test, y_test) = pickle.load(handle)

    label_binarizer = LabelBinarizer()
    # Check the performance of the model

    yhat = model.predict(X_test)
    label_binarizer.fit(y_train)
    yhat1 = label_binarizer.inverse_transform(yhat)

    reportStr = classification_report(y_test, yhat1, output_dict=False)
    print(reportStr)

    cm = confusion_matrix(y_test, yhat1, labels=np.unique(yhat1))
    print(cm)

    ##########################################################################################################################
    #  combine across classes (automate)
    ##########################################################################################################################
    #  Applying SHAP to  training set will help to inspect the ML model, and better understand the model's decision-making
    #  process ("opening up the black box"). However, it is only by applying SHAP to  testing set that  will be able to
    #  figure out how the features impact the model's generalization performance, so it is recommended applying SHAP
    #  to the testing set.
    # we look for only the class that coincides with the true label.
    # select a set of background examples to take an expectation over
    # https://github.com/shap/shap
    # we look for only the class that coincides with the true label.
    # select a set of background examples to take an expectation over
    #n=X_test.shape[0]
    randomIdxTrain = np.random.choice(X_train.shape[0], n, replace=False)
    randomIdxTest = np.random.choice(X_test.shape[0], n, replace=False)
    background = X_train[randomIdxTrain]
    backgroundLabels = y_train[randomIdxTrain]
    testLabels = y_test[randomIdxTest]
    data_for_prediction = pd.DataFrame(X_test[0], columns= features_emg)

    print(model.layers[-1].output.name)  # print out the layer's name
    # passing tensors directly
    # this line needs to be here for HPC. On Lab PC this line is not needed
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer(
        (model.layers[0].input, model.layers[-1].output), background  # last layer output of the model
    )
    del background,backgroundLabels,X_train
    gc.collect()
    shap_values = explainer.shap_values(X_test[randomIdxTest], check_additivity=False)
    #https://towardsdatascience.com/shap-for-binary-and-multiclass-target-variables-ff2f43de0cf4
    #https://towardsdatascience.com/explainable-ai-xai-with-shap-multi-class-classification-problem-64dd30f97cea
    shap_valuesq = np.array(shap_values)
    # before
    shap_valuesq.shape  # (num_samples, num_timepoints, num_features, num_classes) (100, 768, 117, 2)
    shap_valuesq = np.transpose(shap_valuesq, (3, 0, 1, 2))
    #https: // www.datacamp.com / tutorial / introduction - to - shap - values - machine - learning - interpretability
    shap_valuesq.shape # (num_classes, num_samples, num_timepoints, num_features) (2, 100, 768, 117)

    # we select only those shapeley values that correspond to the true labels
    labelsClass = testLabels.astype(int)
    idxClass = labelsClass - 1  # check

    # here we store shapley values only for classes that correspond to true labels.
    uniqueClasses = np.unique(idxClass)
    shapValuesByClassLabel = []
    for k in range(n):
        shapValuesByClassLabel.append(shap_valuesq[np.where(uniqueClasses == idxClass[k])[0][0], k, :, :])

    shapValuesByClassLabel = np.array(shapValuesByClassLabel)
    # shapValuesByClassLabel.shape = (100, 234, 117)
    # averaged shapley values across all 100 samples
    mean_shap_valuesq = np.swapaxes(np.mean(np.abs(shapValuesByClassLabel), axis=0), 0, 1)

    with open(RESULTS_PATH + 'all_shap_values_' + modelName[:-3] +'.pickle', 'wb') as handle:
        pickle.dump((explainer.expected_value,X_test[randomIdxTest],testLabels,shap_values), handle)

    mean_shap_valuesq_arr.append(mean_shap_valuesq)
    custom_palette = sns.color_palette("Blues", 5)

    fig = plt.Figure()
    # bar plot graph
    shap.summary_plot(mean_shap_valuesq.T, data_for_prediction, plot_type='bar', plot_size = (8.5,6), max_display = 3,color=custom_palette[3])
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_PATH+'shap_summary_plot_'+modelName[:-3]+"_fold_"+str(ifold)+'_3.pdf')
    plt.close()
total_mean_shap_valuesq = np.mean(np.array(mean_shap_valuesq_arr), axis = 0)
fig = plt.Figure()
# bar plot graph
shap.summary_plot(total_mean_shap_valuesq.T, data_for_prediction, plot_type='bar', plot_size=(8.5, 6), max_display=3,
                  color=custom_palette[3])
fig = plt.gcf()
fig.set_size_inches(15, 8)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(
    RESULTS_PATH + 'shap_summary_plot_expert_n_' + str(n) + "_avg_across_folds_3_v4.pdf")
plt.close()


importance_df = pd.DataFrame({"column_name": data_for_prediction.columns, "shap_values": np.mean(total_mean_shap_valuesq, axis=1)})
importance_df.to_csv( RESULTS_PATH + "importance_df.csv")