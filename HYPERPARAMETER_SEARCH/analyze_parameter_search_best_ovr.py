import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('default')
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

RESULTS_PATH =  "RESULTS/PARAM_SEARCH_AUTHENTICATION_OVR_STYLUS/"
# RESULTS_PATH = "RESULTS/PARAM_SEARCH_AUTHENTICATION_OVR_HAND_STYLUS/"
# RESULTS_PATH = "RESULTS/PARAM_SEARCH_AUTHENTICATION_OVR_HAND/"
# RESULTS_PATH = "RESULTS/PARAM_SEARCH_AUTHENTICATION_OVR_EMG/"

####################################################################
# Overlap MULTIPLE RUNS ACCURACY ONLY (MULTICLASS)
####################################################################
windows = [128,896,1728]
total_accuracy_array = []
# insert optimal parameters for corresponding modality
ch1 = 1024
k1 = 100

for run in range(1,5):
    df = pd.read_csv(RESULTS_PATH + "auth_ovr_ch1_"+str(ch1)+"_k1_"+str(k1)+"_run_"+ str(run) +".csv")
    tmp_acc_array_per_win_run = [[],[],[]]

    for j in range(3):
        tmp_df = df.iloc[np.where(df["WINDOW"] == windows[j])[0],:]
        tmp_acc_array_per_win_run[j].append(tmp_df["ACC"].to_numpy())
    tmp_acc_array_per_win_run = np.squeeze(np.array(tmp_acc_array_per_win_run))
    total_accuracy_array.append(tmp_acc_array_per_win_run)


total_accuracy_array = np.array(total_accuracy_array)#5 runs, 3 windows,


matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# average by overlaps
x = np.arange(0,100,10)
avg_acc_array = np.mean(total_accuracy_array,axis = 0)

accuracy_per_win = [[], [], []]

for i in range(3):
    tmp_acc = np.mean(total_accuracy_array[:,i,:], axis = 0)


    for ovr in np.arange(0,50,5):
        accuracy_per_win[i].append(np.mean(tmp_acc[ovr:ovr+5]))


accuracy_per_win = np.array(accuracy_per_win)


y1=accuracy_per_win[0,:]
y2=accuracy_per_win[1,:]
y3=accuracy_per_win[2,:]
fig, axs = plt.subplots(figsize=(8,5))
plt.plot(x, y1, 'r.-', label='128')
plt.plot(x, y2, 'g^-', label='896')
plt.plot(x, y3, 'bs-', label='1728')
plt.xticks(x,x)
plt.yticks([0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9],[10,20,30, 40 ,50, 60, 70, 80, 90])
plt.xlabel("overlap, %", fontsize = 24)
plt.ylabel("accuracy, %", fontsize = 24,labelpad=20)
plt.legend(title="window", fontsize = 18)
plt.grid()
plt.tight_layout()
plt.show()
plt.savefig(RESULTS_PATH + 'auth_5_runs_5_folds_average_accuracy.pdf')
plt.savefig(RESULTS_PATH + "auth_5_runs_5_folds_average_accuracy.png", dpi=600)
plt.close()
