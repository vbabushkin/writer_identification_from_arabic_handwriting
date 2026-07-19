import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('default')
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

RESULTS_PATH =  "RESULTS/PARAM_SEARCH_AUTHENTICATION_WIN_STYLUS/"
# RESULTS_PATH = "RESULTS/PARAM_SEARCH_AUTHENTICATION_WIN_HAND_STYLUS/"
# RESULTS_PATH = "RESULTS/PARAM_SEARCH_AUTHENTICATION_WIN_HAND/"
# RESULTS_PATH = "RESULTS/PARAM_SEARCH_AUTHENTICATION_WIN_EMG/"
####################################################################
# WINDOWS SIZE MULTIPLE RUNS ACCURACY ONLY (MULTICLASS)
####################################################################
# set the optimal channel and kernel
ch1 = 2048 #1024#2048
k1 = 100
windows = np.arange(128, 1774, 64)#1774
total_accuracy_array = []

for run in range(1,5):
    df = pd.read_csv(RESULTS_PATH + "authen_win_ch1_"+str(ch1)+"_k1_"+str(k1)+"_run_"+ str(run + 1) +".csv")
    tmp_avg_acc_array_per_run = []

    for win in windows:
        tmp_df = df.iloc[np.where(df["WINDOW"] == win)[0],:]
        tmp_avg_acc_array_per_run.append(np.mean(tmp_df["ACC"].to_numpy()))
    total_accuracy_array.append(tmp_avg_acc_array_per_run)


mean_accuracy = np.mean(total_accuracy_array, axis = 0)

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig, axs = plt.subplots(figsize=(8,5))
plt.plot(windows ,mean_accuracy, 'bo-', label = "accuracy")
plt.xlabel("window size", fontsize = 24)
plt.ylabel("accuracy, %", fontsize = 24)
plt.xticks([50, 250, 500, 750, 1000, 1250,1500,1750],[50,250, 500, 750, 1000, 1250,1500,1750])
#plt.yticks(np.arange(0.55,0.95,0.05),[str(int(a*100)) for a in np.arange(0.55,0.95,0.05)])
plt.yticks(np.arange(0.15,0.65,0.05),[str(int(a*100)) for a in np.arange(0.15,0.65,0.05)])
axs.tick_params(axis='x', labelsize=18)
axs.tick_params(axis='y', labelsize=18)
plt.xlim([56,1800])
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig(RESULTS_PATH + 'win_auth_5_runs_5_folds_average_accuracy_v1.pdf')
plt.savefig(RESULTS_PATH + "win_auth_5_runs_5_folds_average_accuracy_v1.png", dpi=600)
plt.close()
