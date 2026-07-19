import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.ticker import MaxNLocator
# Load CSV file
# Create a figure and axis for the plot
plt.figure(figsize=(9, 4.5))

# Generate colors from the Blues palette
colors = sns.color_palette("Blues", n_colors=7)
x_tick_positions = []
x_tick_str = []
means_array = [[] for _ in range(5)]
# Define properties for the whiskers, including thickness of the whisker hats
whisker_props = dict(color='black', linewidth=0.5)  # capthick controls the width of whisker hats

# Define properties for the median lines
median_props = dict(color='black', linewidth=0.5)  # Adjust linewidth for the median line

# Define properties for the caps (ends of the whiskers)
cap_props = dict(color='black', linewidth=0.5)  # Change linewidth to control cap thickness

for numpar in range(1,6):
    s_path = '/STYLUS_NUM_TRAIN/auth_ch1_2048_k1_100_w_1152_numpar_' + str(numpar) + '_stylus.csv'
    h_path = '/HAND_NUM_TRAIN/auth_ch1_1024_k1_100_w_1344_numpar_'+str(numpar)+'_hand.csv'
    hs_path = '/HAND_STYLUS_NUM_TRAIN/auth_ch1_512_k1_100_w_1024_numpar_'+str(numpar)+'_hand_stylus.csv'

    s_df = pd.read_csv(s_path)
    h_df = pd.read_csv(h_path)
    hs_df = pd.read_csv(hs_path)

    a1 = np.asarray(s_df["ACC"]).reshape(-1,1)
    a2 = np.asarray(h_df["ACC"]).reshape(-1,1)
    a3 = np.asarray(hs_df["ACC"]).reshape(-1,1)
    a = np.hstack((a1,a2,a3))
    a = np.round(a*100, 2)
    means_array[numpar - 1].append(np.mean(a,axis = 0))

    # Convert the array to a DataFrame for easier plotting
    df = pd.DataFrame(a, columns=['stylus', 'hand', 'stylus & hand'])

    # Set width of the boxplots
    box_width = 0.2  # Controls the width of the boxes
    positions = np.array([numpar])  # Shared tick at position 0


    # Plot the first boxplot (Column1) using a color from the Blues palette
    bp1 = plt.boxplot(df['stylus'], positions=positions - box_width, widths=box_width, patch_artist=True,
                      boxprops=dict(facecolor=colors[1], color="black", linewidth=0.5), medianprops=median_props, capprops = cap_props, showfliers=False, whiskerprops=whisker_props, zorder=2)

    # Plot the second boxplot (Column2) using another color from the Blues palette
    bp2 = plt.boxplot(df['hand'], positions=positions, widths=box_width, patch_artist=True,
                      boxprops=dict(facecolor=colors[5], color="black", linewidth=0.5), medianprops=median_props, capprops = cap_props, showfliers=False, whiskerprops=whisker_props, zorder=2)

    # Plot the third boxplot (Column2) using another color from the Blues palette
    bp3 = plt.boxplot(df['stylus & hand'], positions=positions + box_width, widths=box_width, patch_artist=True,
                      boxprops=dict(facecolor=colors[3], color="black", linewidth=0.5), medianprops=median_props, capprops = cap_props, showfliers=False, whiskerprops=whisker_props, zorder=2)
    x_tick_positions.append(positions[0])
    x_tick_str.append(str(numpar))

means_array = np.squeeze(np.array(means_array))

# Set x-ticks and labels
plt.xticks(x_tick_positions, x_tick_str)

plt.plot(np.array(x_tick_positions) - box_width, means_array[:,0], marker='o', markersize=3,
         markeredgecolor='black',markerfacecolor='white', markeredgewidth=0.5,linestyle='-.',color = colors[1], zorder=1 )
plt.plot(np.array(x_tick_positions), means_array[:,1], marker='o', markersize=3,
         markeredgecolor='black',markerfacecolor='white', markeredgewidth=0.5,linestyle=':',color = colors[5], zorder=1 )
plt.plot(np.array(x_tick_positions) + box_width, means_array[:,2], marker='o', markersize=3,
         markeredgecolor='black',markerfacecolor='white', markeredgewidth=0.5,linestyle='--',color = colors[3], zorder=1 )


plt.plot(np.array(x_tick_positions) - box_width, means_array[:,0], marker='o', markersize=3,
         markeredgecolor='black',markerfacecolor='white', markeredgewidth=0.5,linestyle="None", zorder=3 )
plt.plot(np.array(x_tick_positions), means_array[:,1], marker='o', markersize=3,
         markeredgecolor='black',markerfacecolor='white', markeredgewidth=0.5,linestyle="None", zorder=3 )
plt.plot(np.array(x_tick_positions) + box_width, means_array[:,2], marker='o', markersize=3,
         markeredgecolor='black',markerfacecolor='white', markeredgewidth=0.5,linestyle="None", zorder=3 )

plt.ylim([20,100])
plt.xlim([0.5,5.5])
plt.grid()
# Add labels and title
#plt.title('Boxplots of Three Columns Centered')
plt.ylabel('Accuracy, %')
plt.xlabel('Number of paragraphs in training set')
# Add legend
plt.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['stylus', 'hand', 'stylus & hand'], title="Features Used")
plt.tight_layout()
# Save the plot as a file
plt.savefig('centered_boxplot_with_legend.pdf')
