# Multimodal Approach to Writer Identification from Arabic Handwriting

This document provides technical details of code used in "Multimodal Approach to Writer Identification from Arabic Handwriting" paper.
## 1. Data Preprocessing and Cleaning


The `DATA_PREPROCESSING` folder contains following files for Hand/Stylus Kinematics and Surface Electromyography (sEMG) signals preprocessing:  
* `kinematics_preprocess_data.py`, 
* `utilities.py`,
* `emg_preprocess_data.py`.

---

### 1.1. Hand/Stylus Kinematics Preprocessing Steps

To retain the fine-grained, personalized characteristics of handwriting, a minimal preprocessing  is applied to the stylus and hand-kinematics data:

#### 1. Metadata Exclusion
* Non-kinematic data and metadata are completely excluded from the dataset to ensure privacy of the subjects.
* The features `handId`, `sec`, `min`, `hour`, `lifetimeOfThisHandObject`, and `confidence` are dropped from the final DataFrames to keep only structural movement data.

#### 2. Feature Type Casting
* The remaining kinematic features are extracted and converted from structured DataFrames into raw NumPy arrays, cast to a 32-bit floating-point precision format (`np.float32`) to achieve optimal computational efficiency during training.

#### 3. Edge Sample Trimming
* To eliminate initialization noise and artifacts at the boundaries of the text-writing sessions, the start and end of each recording are trimmed by discarding the first and the last two records of the sequence (`tmpX = tmpX[2:-2, :]`).

#### 4. Dynamics Preservation (No Smoothing)
* Unlike traditional signal processing pipelines that usually smooth input signals, this pipeline strictly avoids applying any median filtering or signal smoothing techniques to ensure that the individual writer-specific fluctuations in hand and stylus dynamics are fully preserved.

---
### 1.2. EMG Preprocessing Steps

The preprocessing pipeline converts raw EMG signals into structured, normalized feature arrays ready for machine learning models.
It contains the following steps:
#### 1. Time-Alignment & Trimming
* Raw EMG sequences are synchronized with recorded kinematic data by aligning the start and stop triggers inserted during the data acquisition stage.
* Data points are filtered to fit strictly between the first and last timestamps of the corresponding kinematics recording (`t[0]` and `t[len(t)-1]`), to achieve high temporal alignment across modalities.

#### 2. Feature Selection & Type Casting
* The initial timestamp metadata columns  are dropped to keep only the functional EMG muscle activity channels.
* Structured DataFrames are converted into NumPy arrays and downcasted to 32-bit floating-point precision (`np.float32`) for computational efficiency.

#### 3. Kinematic-based Line Segmentation
* Continuous paragraph-level EMG sequences are sliced into discrete lines.
* Line boundaries are identified by evaluating structural peaks (`find_peaks`) and computing spatial gradients (`np.gradient`) from the synchronized kinematic data.

#### 4. Downsampling
* To minimize dimensionality and computational complexity, the raw signals are downsampled from their native acquisition frequency ($Fs = 1260\text{ Hz}$) to a standard target frequency of **128 Hz** using the `scipy.signal.resample` method.

#### 5. Paragraph Recombination
* For paragraph-level evaluations, individual segmented lines belonging to the same subject and writing task are vertically restacked into a unified array using `np.vstack`.
---
### 1.3. Handling of Missing, Empty, and Short Sequences

Instead of using statistical imputation methods (e.g., mean substitution or interpolation),  invalid or insufficient sequences are removed or trimmed according to explicit rules.
#### 1.3.1. Handling Integrity of  Kinematics Data

| Data Issue                      | Detection Rule / Threshold                                                                                                                           | Handling Mechanism                                                                                                                                                   |
|:--------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Boundary Artifacts**          | The boundary margins of a text-writing sequence block, defined as the first 2 and final 2 sampled indices (`tmpX[2:-2, :]`).                         | **Trimmed & Discarded**: The edge frames are sliced off the array entirely to avoid capturing initialization lags or pen lift/lower instability.                     |
| **Non-Kinematic Features**      | Non-kinematics features including tracking ID and timestamps (`'handId'`, `'sec'`, `'min'`, `'hour'`, `'lifetimeOfThisHandObject'`, `'confidence'`). | **Filtered & Purged**: Removed from dataframe using `df.drop()`.                                                                                                     |
| **High-Frequency Fluctuations** | Signal perturbations across all spatial dimensions evaluated at the data loading stage.                                                              | **Preserved Intact**: Does not use any filtering, such as median filter (`applyFilter = False`), preserving the handwriting dynamics specific to individual writers. |


#### 1.3.2. Handling Integrity of EMG Data  

| Data Issue | Detection Rule / Threshold | Handling Mechanism                                                                                                                                                                                                                                                     |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Zero-Length Samples** | Identified when a sequence has no records (`len(X[i]) == 0`) during downsampling. | **Skipped entirely**: The sample is completely omitted, and its corresponding entry is deleted from the metadata list (`mainSubjInfo`).                                                                                                                                |
| **Short Artifacts / Commas / Dots** | Sequences containing fewer than 100 frames (`X[k].shape[0] < 100`). | **Flagged & Removed**: Identified as empty frames or brief non-word strokes rather than legibly structured text segments.                                                                                                                                              |
| **Sub-Window Shortfalls** | Any sequence whose total length falls below the mandatory sliding window size (`winSize = 1024`). | **Purged**: The script scans the dataset dimensions, flags indices failing to meet the `winSize` criteria, and removes them from the feature array `X`, labels `Y`, and metadata trackers. Deletions are processed in reverse index order to preserve array stability. |



---

## 2. Hyperparameter Search and Optimization Pipeline

The scripts for identifying optimal deep learning architectures, optimal window sizes and overlaps are stored in `HYPERPARAMETER_SEARCH` directory. To isolate fine-grained neuromuscular and kinematic behavioral patterns, optimization is executed across three primary dimensions: identifying 1D-CNN architectural constraints (channel and kernel sizes), determining the ideal temporal sliding window lengths, and finding optimal window overlaps.

### 2.1. Hyperparameter Grid Space Configurations

| Optimization Domain                          | Scripts                          | Search Space                                            | Optimization Target                                                  |
|:---------------------------------------------|:---------------------------------|:--------------------------------------------------------|:---------------------------------------------------------------------|
| **Stylus Architecture**                      | `arch_search_stylus.py`          | Channels: `[8 to 2560]` <br> Kernel Sizes: `[3 to 800]` | Optimal 1D-CNN architecture 110 stylus kinematics features.          |
| **Hand Kinematics Architecture**             | `arch_search_hand.py`            | Channels: `[8 to 1024]` <br> Kernel Sizes: `[3 to 800]` | Optimal 1D-CNN architecture 110 hand kinematics features.            |
| **Hand and Stylus Architecture**             | `arch_search_hand_and_stylus.py` | Channels: `[8 to 1024]` <br> Kernel Sizes: `[3 to 800]` | Optimal 1D-CNN architecture 117 hand and stylus kinematics features. |
| **EMG Architecture**                         | `arch_search_emg.py`             | Channels: `[32 to 576]` <br> Kernel Sizes: `[5 to 750]` | Optimal 1D-CNN architecture surface EMG records for EDC, BB, FDI.    |
| **Stylus Kinematic Windows**                 | `kin_win_search_stylus.py`       | Length: `[128 to 1728]` timepoints                      | Optimal window length for stylus kinematics.                         |
| **Hand and Stylus Kinematic Windows**        | `kin_win_search_hand_stylus.py`  | Length: `[128 to 1728]` timepoints                      | Optimal window length for hand and stylus  kinematics.               |
| **Hand Kinematic Windows**                   | `kin_win_search_hand.py`         | Length: `[128 to 1728]` timepoints                      | Optimal window length for hand kinematics.                           |
| **sEMG Windows**                             | `emg_win_search.py`              | Window: `[100 to 25,000]` timepoints                    | Optimal window length for EMG signals.                               |
| **Stylus Kinematic Window Overlap**          | `kin_ovr_search_stylus.py`       | Overlap Percentages: `0% to 90%`                        | Optimal overlap for stylus  kinematics.                              |
| **Hand and Stylus Kinematic Window Overlap** | `kin_ovr_search_hand_stylus.py`  | Overlap Percentages: `0% to 90%`                        | Optimal overlap for stylus and hand kinematics.                      |
| **Hand Kinematic Window Overlap**            | `kin_ovr_search_hand.py`         | Overlap Percentages: `0% to 90%`                        | Optimal overlap for hand kinematics.                                 |
| **sEMG Overlap**                             | `emg_ovr_search.py`              | Overlap Percentages: `0% to 90%`                        | Optimal overlap for for EMG signals.                                 |


### 2.2. Visualization Scripts Overview

Below is the summary of the dedicated post-processing and plotting scripts used to visualize the hyperparameter search performance metrics.

| File Name | Graph Type        | Purpose                                                                                                              |
| :--- |:------------------|:---------------------------------------------------------------------------------------------------------------------|
| `analyze_parameter_search_best_ovr.py` | Multi -Line Plot  | Visualizes model accuracy trends across  different sliding window overlap percentages (0% to 90%).                   |
| `analyze_parameter_search_best_window.py` | Single Line Curve | Visualizes model  accuracy across different sliding window sizes.                                                    |
| `emg_analyze_param_search_architecture.py` | Matrix Heatmap    | Generates heatmap of 1D-CNN layer channels and kernel sizes optimized for surface electromyography (sEMG) signals.   |
| `kin_analyze_parameter_search_architecture.py` | Matrix Heatmap    | Generates heatmap of 1D-CNN layer channels and kernel sizes for Stylus/Hand and combined Stylus and Hand kinematics. |


## 3. Performance Evaluation

The performance of the models across different modalities (EMG signals, hand kinematics, stylus kinematics, and their combination) is evaluated using a rigorous validation protocol designed to prevent data leakage and handle class imbalance. 

### 3.1. Evaluation Protocol
* **Stratified 5-Fold Cross-Validation:** The dataset is split into 5 folds stratified by subject. To ensure realistic evaluation, paragraph-level integrity is maintained; lines belonging to the same paragraph are completely isolated within either the training set or the testing set, never split across both.
* **Class Balancing:** To address discrepancies in subject samples per fold, missing subjects in the test partition are dynamically balanced. Training data is upsampled using `RandomOverSampler` to handle minor class imbalances prior to feeding into the network.
* **Data Standardization:** Within each cross-validation loop, feature vectors are normalized independently using standard scaling (`StandardScaler`) to prevent temporal data leakage.
* **Metrics:** The pipeline automatically tracks and exports cross-validation logs containing:
  * Accuracy, Precision, Recall, and $F_1$-score per fold.
  * Macro-averaged performance metrics.
  * Normalized average confusion matrices (saved as both `.pdf` and `.png` visual plots).

---

### 3.2.  Configuration Hyperparameters

The model architecture utilizes a 1D Temporal Convolutional Network (TCN) block combined with a Custom Attention layer, followed by fully connected dense layers. The parameters are tailored specifically to the characteristics of each input modality as outlined below:

| Hyperparameter / Architecture Component               | EMG Signals Model | 110 Hand Kinematics Model | 117 Hand & Stylus Model | 7 Stylus Kinematics Model |
|:------------------------------------------------------| :--- | :--- | :--- | :--- |
| **Input Feature Slicing**                             | Full Matrix | `[:, :, 7:]` (Hand features only) | Full Matrix (Combined) | `[:, :, :7]` (Stylus features only) |
| **Window Size (`winSize`)**                           | 1000 | 1344 | 1024 | 1152 |
| **Overlap Ratio (`ovr`)**                             | 0.9 (90%) | 0.9 (90%) | 0.9 (90%) | 0.9 (90%) |
| **Conv1D Filters (`num_ch_1`)**                       | 576 | 1024 | 512 | 2048 |
| **Conv1D Kernel Size (`kernel_1`)**                   | 10 | 100 | 100 | 100 |
| **Optimizer & Learning Rate**                         | Adadelta ($10^{-2}$) | Adadelta ($10^{-3}$) | Adadelta ($10^{-3}$) | Adadelta ($10^{-3}$) |
| **Training Epochs**                                   | 500 | 100 | 100 | 100 |
| **Batch Size**                                        | 64 | 128 | 128 | 128 |
| **Dropout Rate**                                      | 0.3 | 0.3 | 0.3 | 0.3 |
| **Hidden Fully Connected Layers**                     | 576 → 256 → 128 → 64 | 512 → 256 | 512 → 256 | 512 → 256 |
| **Output Layer**                                      | 50 (Softmax) | 50 (Softmax) | 50 (Softmax) | 50 (Softmax) |
| **Random State Seed**                                 | 9 | 9 | 9 | 9 |

---

### 3.3. Outputs and Directory Structure
The evaluation script is stored in `PERFORMANCE_EVALUATION` folder and saves configuration-specific metrics automatically into the assigned results paths (`/CODE/AUTH_.../`):
* `*_model_fold_X.h5`: Trained Keras model weights per fold.
* `*.csv`: Comprehensive evaluation matrix logging runtimes and global metrics.
* `*_avg_cm.pdf` / `*.png`: Heatmaps visualising model confusion matrices across 50 classes.


## 4. Model Interpretability (Shapley Values)

To understand which features are important for models to identify writers, feature contributions are interpreted globally and locally using **SHAP (SHapley Additive exPlanations)** via the `DeepExplainer` framework. This evaluates how individual features influence the writer identification decisions.

### 4.1. Implementation Workflow

1. A representative background data allocation is sampled uniformly from the  training sets ($n = 200$ for EMG; $n = 1000$ for Kinematic features).
2. Because writer identification uses 50-classes, the high-dimensional SHAP arrays are transformed and isolated to extract feature importances matching only the targeted true label class.
3. SHAP values are extracted independently across all 5 validation folds to maintain strict out-of-fold testing integrity and to prevent structural data leaks.

---

### 4.2. Execution Parameters & Targets

| Modality  | Analyzed Features                 | Reference Sample Size ($n$) | Primary Visual Export                 |
| :--- |:----------------------------------| :--- |:--------------------------------------|
| **EMG Signals** | 3 Channels (`FDI`, `EDC`, `BB`)   | 200 | Full Feature Contribution Bar Plot    |
| **HAND Kinematics** | 110 Hand Kinematics Features      | 1000 | Top 20 Features Contribution Bar Plot |
| **STYLUS Kinematics** | 7 Stylus Kinematics Features      | 1000 | Top 20 Features Contribution Bar Plot |
| **COMBINED (Hand/Stylus)** | 117 Stylus and Hand Kinematics Features | 1000 | Top 20 Features Contribution Bar Plot |

> 💡 **HPC Cluster Deployment Note:** 
> When executing the deep graph evaluation over specialized GPU nodes, an internal handler override for TensorFlow operation graphs is executed inside the script (`AddV2` mapped to `passthrough`). This prevents operational compilation blocks within the `DeepExplainer` module.

---

### 4.3. Outputs and Directory Structure
The explainability code is stored in 'EXPLAINABILITY' folder. All interpretability data are saved directly into corresponding modality destination folders (`/CODE/AUTH_[MODALITY]/`):
* `all_shap_values_*.pickle`: Complete serialized data dumps tracking expected values, target arrays, and raw SHAP values.
* `importance_df_*.csv`: Aggregated global feature weight score arrays, sorted uniformly by cumulative mean absolute impact.
* `shap_summary_plot_*.pdf`: Plots of global features impact profiles (both per-fold isolates and consolidated multi-fold averages).

## 5. Search for Optimal Number of Paragraphs for  Training  

The goal is to determine the optimal number of paragraphs  required to successfully train the models using the kinematics data. The scripts are stored within the `OPTIMAL_NUMBER_OF_PARAGRAPHS_SEARCH` workspace folder.

### 5.1. Optimization Protocol
* **Volume Range Verification:** The code evaluates model classification accuracy changes with the step-wise increments of the number of paragraphs in training set ranging from 1 to 5 paragraphs per subject.
* **Combinatorial Paragraph Partitioning:** For each target paragraphs number, the pipeline uses the `combinations` utility to sample combinations from a subject's available paragraphs. Five combinations are randomly drawn to assemble strict cross-validation folds, guaranteeing that lines belonging to the same text paragraph do not span across train and test sets simultaneously.
* **Modality Profile Isolation:** The following modalities were analyzed:
  * **Stylus Only Kinematics:** Evaluated with 7 stylus kinematics features.
  * **Hand Only Kinematics:** Evaluated with 110 hand kinematics features.
  * **Combined Hand & Stylus Kinematics:** Evaluated over the full set of 117 stylus and hand kinematics features. 
* **Balanced Testing Control:** To prevent evaluation bias caused by structural drops in baseline content, the pipeline dynamically detects missing validation subjects across the sparse combinations and injects single samples balanced with `RandomOverSampler`.

### 5.2. Outputs & Directory Structure
Execution logs and visualization scripts save the results  to the  following directories :
* `/CODE/STYLUS_NUM_TRAIN/`, `/CODE/AUTH_HAND_NUM_TRAIN/`, and `/CODE/HAND_STYLUS_NUM_TRAIN/`: contain serialized fold models (`*.h5`), split definitions (`_folds.pickle`), and model performance evaluations (`*.csv`)  for each number of paragraphs.
* `centered_boxplot_with_legend.pdf`: An aggregated grouped boxplot figure showing accuracy metrics  with the trend lines  as a function of the model performance with the number of  paragraphs in the  training set.

## 6. Ablation & Normalization Experiments

The scripts are stored in  the `ABLATION_EXPERIMENTS/` 
### 6.1. Objectives & Rationale
These experiments evaluate  whether the models' identification accuracy is driven primarily by  handwriting dynamics behavioral characteristics or if it relies on static, participant-specific features that provide an easier shortcut to identity matching. Specifically, the models are evaluated against the targeted removal of stable physical body morphologies (e.g., hand width, arm length, arm width) and absolute spatial positioning coordinates. 

The system compares baseline models directly in three experimental designs:
* **Excluding Hand Morphology:** Ablates  anatomical features from the feature sets to prevent identification based purely on physical hand morphologys.
* **Coordinate Normalization:** Transforms spatial coordinates  into local referenced coordinates  (wrist-, palm-, or tablet-centered) to eliminate absolute location bias.
* **Excluding Raw Coordinate Channels:** Completely removes raw spatial positioning coordinates to evaluate if high-order kinematic derivatives are sufficient on their own.

### 6.2. Performance Interpretation
* **Large Performance Reductions:** Indicate that the removed structural or positioning group contributes to writer identification, revealing that a model takes spatial or anatomical shortcuts.
* **Preserved Performance:** Demonstrates that the model utilizes  purely dynamic behavioral signals  for writer identification.

### 6.3. Ablation and Normalization Scripts 

The table below contains scripts for ablation/normalization experiments for Hand Kinematics, Stylus Kinematics, and Stylus and Hand Kinematics modalities:

|  Modality  | Ablation / Normalization             | File Name                          |
| :--- |:-------------------------------------|:-----------------------------------|
| **Hand-Only Kinematics** | Static Features Removal (Partial)    | hand_ablation_static.py            |
| | Static Features Removal (All)        | hand_ablation_static_all.py        |
| | Palm-Normalized Spatial Coordinates  | hand_ablation_norm_palm.py         |
| | Wrist-Normalized Spatial Coordinates | hand_ablation_norm_wrist.py        |
| **Stylus-Only Kinematics** | Static Features Removal (Partial)    | stylus_ablation_static.py          |
| | Static Features Removal (All)        | stylus_ablation_static_all.py      |
| | Static Feature Normalized Tablet     | stylus_ablation_norm_static.py     |
| **Combined Hand & Stylus** | Static Features Removal (Partial)    | hand_stylus_ablation_static.py     |
| | Static Features Removal (All)        | hand_stylus_ablation_static_all.py |
| | Palm-Normalized Spatial Coordinates  | hand_stylus_ablation_norm_palm.py  |
| | Wrist-Normalized Spatial Coordinates | hand_stylus_ablation_norm_wrist.py |

> **Note on Validation Consistency:** 
> Every script for ablation experiments follows the baseline protocol's strict cross-validation boundaries, isolating paragraph-level segments to maintain parity and ensure performance variations are directly comparable.