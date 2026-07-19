import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('default')
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns
# RESULTS_PATH = "RESULTS/AUTHENTICATION_ARCHITECTURE_HAND/"
# RESULTS_PATH = "RESULTS/AUTHENTICATION_ARCHITECTURE_HAND_AND_STYLUS/"
RESULTS_PATH = "RESULTS/AUTHENTICATION_ARCHITECTURE_STYLUS/"
channels = [8, 16, 32, 64, 128, 256, 512, 1024]
kernels = [3, 5, 10, 25, 50, 75, 100, 250, 500, 650, 700, 750, 800]
avg_accuracy = np.zeros((len(channels),len(kernels)))
for i in range(len(channels)):
    for j in range(len(kernels)):
        num_ch_1 = channels[i]
        kernel_1 = kernels[j]
        MAIN_FILENAME = "authen_ch1_" + str(num_ch_1) + "_k1_" + str(kernel_1)
        report_filename = RESULTS_PATH + MAIN_FILENAME + ".csv"
        df = pd.read_csv(report_filename)
        avg_accuracy[i,j] = np.mean(df["ACC"])
plt.close("all")
plt.figure(figsize=(13, 8))
sns.set(font_scale=1.4)

ax = plt.gca()
labels = np.zeros((avg_accuracy.shape[0],avg_accuracy.shape[1]))
for y in range(avg_accuracy.shape[0]):
    for x in range(avg_accuracy.shape[1]):
        val = avg_accuracy[y, x]
        labels[y, x] = np.round(val*100,1)
ax = sns.heatmap(avg_accuracy*100,
                 cbar_kws={'ticks': [40, 50, 60, 70, 80, 90, 100]}, vmin=40, vmax=100.0,
                 annot=labels, annot_kws={"size": 20}, fmt='', cmap="jet")  # font size
cbar1 = ax.collections[0].colorbar
cbar1.ax.tick_params(labelsize=20)
cbar1 = ax.collections[0].colorbar
cbar1.ax.tick_params(labelsize=16)
bottom, top = ax.get_ylim()
ax.set_ylim( top - 0.5,bottom + 0.5)
ax.tick_params(axis='x', which='major', pad=-3)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_xticklabels(kernels, rotation=0, fontsize="20", va="center", ha="center")
ax.set_yticklabels(channels, rotation=0, fontsize="20")
ax.set_ylabel("Number of channels in 1D CNN layer", fontsize="24",labelpad=15)
ax.set_xlabel("Size of kernel of 1D CNN layer", fontsize="24",labelpad=15)
ax.tick_params(axis='both', which='major', pad=15)
ax.invert_yaxis()
plt.tight_layout()
plt.ylim([0,len(channels)])
plt.tight_layout()
plt.savefig(RESULTS_PATH + MAIN_FILENAME + '.pdf')
plt.savefig(RESULTS_PATH + MAIN_FILENAME + '.png')
plt.close()
