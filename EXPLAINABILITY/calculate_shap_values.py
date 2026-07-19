# Shapley values for kinematics features
import gc
import numpy as np
import shap
import tensorflow as tf

print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)

import pickle
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

features_117 = ["pressure",
                "azimuth",
                "altitude",
                "x",
                "y",
                "z",
                "proximity",
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
# parameters (modify accordingly for each model)
#
#################################################################################################################
n = 1000  # samples from train and test to calculate Shapley values
EPOCHS = 100
BATCH_SIZE = 128
lr = 1e-3
nfold = 5
dropout = 0.3
LOSS = tf.keras.losses.categorical_crossentropy
RANDOM_STATE = 9#42 for parameter search, 9 for calculating Shapley values
ovr = 0.9

MODEL_NAMES = ['HAND','STYLUS','HAND_STYLUS']

for model_name in MODEL_NAMES:
    if model_name == 'HAND':
        winSize = 1344
        num_ch_1 = 1024
        kernel_1 = 100
    if model_name == 'STYLUS':
        winSize = 1152
        num_ch_1 = 2048
        kernel_1 = 100
    if model_name == 'HAND_STYLUS':
        winSize = 1024
        num_ch_1 = 512
        kernel_1 = 100

    RESULTS_PATH = f"/CODE/AUTH_{model_name}/"
    MAIN_FILENAME = "auth_ch1_" + str(num_ch_1) + "_k1_" + str(kernel_1)+"_w_"+str(winSize)
    overlap = int(winSize * ovr)

    shapValuesByClassLabelArr = []
    mean_shap_valuesq_arr =[]
    gc.enable()
    for ifold in range(5):
        modelName = MAIN_FILENAME + "_model_fold_" + str(ifold) + ".h5"
        model = tf.keras.models.load_model(RESULTS_PATH+modelName, custom_objects={'CustomLayer': Attention}, compile=False)
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
        data_for_prediction = pd.DataFrame(X_test[0], columns= features_117)

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

        with open(RESULTS_PATH + 'all_shap_values_' + modelName[:-3] +'_'+str(n)+'.pickle', 'wb') as handle:
            pickle.dump((explainer.expected_value,X_test[randomIdxTest],testLabels,shap_values), handle)

        mean_shap_valuesq_arr.append(mean_shap_valuesq)
        custom_palette = sns.color_palette("Blues", 5)

        fig = plt.Figure()
        # bar plot graph
        shap.summary_plot(mean_shap_valuesq.T, data_for_prediction, plot_type='bar', plot_size = (8.5,6), max_display = 20,color=custom_palette[3])
        fig = plt.gcf()
        fig.set_size_inches(15, 8)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(RESULTS_PATH+'shap_summary_plot_expert_'+modelName[:-3]+"_fold_"+str(ifold)+'_top_20_'+str(n)+'.pdf')
        plt.close()


        # bar plot graph
        fig = plt.Figure()
        fig.set_size_inches(12, 25)
        shap.summary_plot(mean_shap_valuesq.T, data_for_prediction, plot_type='bar', max_display = 117,color=custom_palette[3])
        #fig = plt.gcf()
        plt.tight_layout()
        plt.savefig(RESULTS_PATH+'shap_summary_plot_expert_n_'+str(n)+"_"+modelName[:-3]+"_fold_"+str(ifold)+'_'+str(n)+'_all.pdf')
        plt.close()

    total_mean_shap_valuesq = np.mean(np.array(mean_shap_valuesq_arr), axis = 0)
    fig = plt.Figure()
    # bar plot graph
    shap.summary_plot(total_mean_shap_valuesq.T, data_for_prediction, plot_type='bar', plot_size=(8.5, 6), max_display=20,
                      color=custom_palette[3])
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        RESULTS_PATH + 'shap_summary_plot_expert_n_' + str(n) + '_avg_across_folds_top_20_'+str(n)+'.pdf')
    plt.close()

    # bar plot graph
    fig = plt.Figure()
    fig.set_size_inches(12, 25)
    shap.summary_plot(total_mean_shap_valuesq.T, data_for_prediction, plot_type='bar', max_display=117, color=custom_palette[3])
    # fig = plt.gcf()
    plt.tight_layout()
    plt.savefig(
        RESULTS_PATH + 'shap_summary_plot_expert_n_' + str(n) + '_avg_across_folds_all_'+str(n)+'.pdf')
    plt.close()

    importance_df = pd.DataFrame({"column_name": data_for_prediction.columns, "shap_values": np.mean(total_mean_shap_valuesq, axis=1)})
    importance_df.to_csv( RESULTS_PATH + 'importance_df_expert_'+str(n)+'.csv')