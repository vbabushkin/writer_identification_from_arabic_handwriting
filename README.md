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
* The remaining kinematic features are extracted and converted from structured DataFrames into raw NumPy arrays.
* These arrays are explicitly cast to a 32-bit floating-point precision format (`np.float32`) to achieve optimal computational efficiency during training.

#### 3. Edge Sample Trimming
* To eliminate initialization noise and artifacts at the boundaries of the text-writing sessions, the beginning and ending periods of each recording are trimmed.
* The pipeline achieves this by discarding the first and the last two records of the sequence using array slicing (`tmpX = tmpX[2:-2, :]`).

#### 4. Dynamics Preservation (No Smoothing)
* Unlike traditional signal processing pipelines that heavily smooth inputs, this pipeline strictly avoids applying any median filtering or signal smoothing techniques, ensuring that unique, writer-specific dynamic fluctuations are fully preserved.

---
### 1.2. EMG Preprocessing Steps

The preprocessing pipeline converts raw EMG signals into structured, normalized feature arrays ready for machine learning models:
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

| Data Issue                      | Detection Rule / Threshold                                                                                                                                | Handling Mechanism                                                                                                                                             |
|:--------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Boundary Artifacts**          | The boundary margins of a text-writing sequence block, defined as the first 2 and final 2 sampled indices (`tmpX[2:-2, :]`).                              | **Trimmed & Discarded**: The edge frames are sliced off the array entirely to avoid capturing initialization lags or pen lift/lower instability.               |
| **Non-Kinematic Tracking**      | Extraneous dataset metrics including tracking ID and clock fields (`'handId'`, `'sec'`, `'min'`, `'hour'`, `'lifetimeOfThisHandObject'`, `'confidence'`). | **Filtered & Purged**: Removed from dataframe using `df.drop()`.                                                                                               |
| **High-Frequency Fluctuations** | Signal perturbations across all spatial dimensions evaluated at the data loading stage.                                                                   | **Preserved Intact**: Does not use any filtering, such as median filter (`applyFilter = False`), preserving the micro-dynamics specific to individual writers. |


#### 1.3.2. Handling Integrity of EMG Data  

| Data Issue | Detection Rule / Threshold | Handling Mechanism                                                                                                                                                                                                                                                     |
| :--- | :--- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Zero-Length Samples** | Identified when a sequence has no records (`len(X[i]) == 0`) during downsampling. | **Skipped entirely**: The sample is completely omitted, and its corresponding entry is deleted from the metadata list (`mainSubjInfo`).                                                                                                                                |
| **Short Artifacts / Commas / Dots** | Sequences containing fewer than 100 frames (`X[k].shape[0] < 100`). | **Flagged & Removed**: Identified as empty frames or brief non-word strokes rather than legibly structured text segments.                                                                                                                                              |
| **Sub-Window Shortfalls** | Any sequence whose total length falls below the mandatory sliding window size (`winSize = 1024`). | **Purged**: The script scans the dataset dimensions, flags indices failing to meet the `winSize` criteria, and removes them from the feature array `X`, labels `Y`, and metadata trackers. Deletions are processed in reverse index order to preserve array stability. |



---

## 2. Hyperparameter Search and Optimization Pipeline

The `HYPERPARAMETER_SEARCH` directory contains following files for  identifying optimal deep learning architectures and data-splitting configurations. To isolate fine-grained neuromuscular and kinematic behavioral patterns, optimization is executed across three primary dimensions: identifying 1D-CNN architectural constraints (channel and kernel sizes), determining the ideal temporal sliding window lengths, and finding optimal window overlaps.

| Script | Purpose                                                                             | Search space                                                           |
| :--- |:------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|
| `arch_search_stylus.py` | Finds a suitable 1D-CNN architecture for stylus kinematics.                         | Conv1D channels from 8 to 2560 and kernel sizes from 3 to 800.         |
| `arch_search_hand.py` | Finds a suitable 1D-CNN architecture for the 110 hand-kinematic features.           | Conv1D channels up to 1024, with several kernel sizes.                 |
| `arch_search_hand_and_stylus.py` | Searches architectures for the combined 117-feature hand-and-stylus input.          | Channel and kernel configurations for the full multimodal feature set. |
| `kin_win_search_hand.py` | Finds the optimal window length for hand-kinematics.                                | Window lengths from 128 to 1728 time points.                           |
| `kin_ovr_search_hand_stylus.py` | Finds  the overlap between consecutive windows containg hand and stylus kinematics. | Overlap values from 0% to 90%.                                         |
| `emg_win_search.py` | Find optimal window lengths for the sEMG signals.                                   | Window lengths from 100 to 25,000 time points.                         |

### 2.2. Hyperparameter Grid Space Configurations

The complete evaluation ranges and search parameters evaluated across the pipeline scripts are summarized below:

| Optimization Domain |  Scripts                         |  Search Space                                               | Optimization Target                                                  |
| :--- |:---------------------------------|:------------------------------------------------------------|:---------------------------------------------------------------------|
| **Stylus Architecture** | `arch_search_stylus.py`          | Channels: `[8 to 2560]` <br> Kernel Sizes: `[3 to 800]`     | MOptimal 1D-CNN architecture 110 stylus kinematics features.         |
| **Hand Kinematics Architecture** | `arch_search_hand.py`            | Channels: Up to `1024` <br> Varied Kernel Widths            | Optimal 1D-CNN architecture 110 hand kinematics features.            |
| **Multimodal Joint Architecture** | `arch_search_hand_and_stylus.py` | Multi-stream balanced layers <br> Full 117 Feature Set      | Optimal 1D-CNN architecture 117 hand and stylus kinematics features. |
| **Kinematic Windows** | `kin_win_search_hand.py`         | Length: `[128 to 1728]` timepoints                          | Optimal window length for hand kinematics .                          |
| **Kinematic Window Overlap** | `kin_ovr_search_hand_stylus.py`  | Overlap Percentages: `0% to 90%`                            | Optimal overlap for stylus and hand kinematics.                      |
| **sEMG Window & Overlap** | `emg_win_search.py`              | Window: `[100 to 25,000]` timepoints <br> Variable Overlaps | Optimal window length for EMG signals.                               |


### 2.3. Visualization Scripts Overview

Below is the summary of the dedicated post-processing and plotting scripts used to visualize the hyperparameter search performance metrics.

| File Name | Graph Type        | Purpose |
| :--- |:------------------| :--- |
| `analyze_parameter_search_best_ovr.py` | Multi -Line Plot  | Evaluates and plots multi-run accuracy trends against changing temporal window overlap percentages (0% to 90%). |
| `analyze_parameter_search_best_window.py` | Single Line Curve | Visualizes model identification performance across a progressive sliding window size spectrum. |
| `emg_analyze_param_search_architecture.py` | Matrix Heatmap    | Generates matrix intensity charts profiling cross-combinations of 1D-CNN layer channels and kernel sizes optimized for surface electromyography (sEMG) signals. |
| `kin_analyze_parameter_search_architecture.py` | Matrix Heatmap    | Visualizes multi-dimensional parameter search layouts (channels vs. kernels) for positional tracking features (Stylus/Hand kinematics configurations). |


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

### 3.4. Outputs and Directory Structure
The evaluation script is stored in `PERFORMANCE_EVALUATION` folder and saves configuration-specific metrics automatically into the assigned results paths (`/CODE/AUTH_.../`):
* `*_model_fold_X.h5`: Trained Keras model weights per fold.
* `*.csv`: Comprehensive evaluation matrix logging runtimes and global metrics.
* `*_avg_cm.pdf` / `*.png`: Heatmaps visualising model confusion matrices across 50 classes.


## 4. Model Interpretability (Shapley Values)

To demystify the black-box nature of the 1D Temporal Convolutional Networks, feature contributions are interpreted globally and locally using **SHAP (SHapley Additive exPlanations)** via the `DeepExplainer` framework. This evaluates how individual physical feature groups influence the multi-class biometric authentication decisions.

### 4.1. Implementation Workflow

1. **Expectation Baselines:** A representative background data allocation is sampled uniformly from the isolated training sets ($n = 200$ for EMG; $n = 1000$ for Kinematic streams) to compute base value expectations.
2. **True Class Isolation:** Because authentication involves a 50-class setup, the high-dimensional SHAP arrays are transformed and isolated (`np.transpose`) to extract feature importances matching only the targeted **true label class** for realistic evaluation.
3. **Cross-Validation Integration:** SHAP values are extracted independently across all **5 validation folds** to maintain strict out-of-fold testing integrity and to prevent structural data leaks.

---

### 4.2. Execution Parameters & Targets

| Modality Config | Analyzed Features                 | Reference Sample Size ($n$) | Primary Visual Export              |
| :--- |:----------------------------------| :--- |:-----------------------------------|
| **EMG Signals** | 3 Channels (`FDI`, `EDC`, `BB`)   | 200 | Full Feature Contribution Bar Plot |
| **HAND Kinematics** | 110 Hand Kinematics Features      | 1000 | Top 20 & Global Feature Metrics    |
| **STYLUS Kinematics** | 7 Stylus Kinematics Features      | 1000 | Top 20 & Global Feature Metrics    |
| **COMBINED (Hand/Stylus)** | 117 Stylus and Hand Kinematics Features | 1000 | Top 20 & Global Feature Metrics                                   |

> 💡 **HPC Cluster Deployment Note:** 
> When executing the deep graph evaluation over specialized GPU nodes, an internal handler override for TensorFlow operation graphs is executed inside the script (`AddV2` mapped to `passthrough`). This prevents operational compilation blocks within the `DeepExplainer` module.

---

### 4.3. Outputs and Directory Structure
The explainability code is stored in 'EXPLAINABILITY' folder. All interpretability data maps directly into respective modality destination roots (`/CODE/AUTH_[MODALITY]/`):
* `all_shap_values_*.pickle`: Complete serialized data dumps tracking expected values, target arrays, and raw SHAP multi-dimensional weights.
* `importance_df_*.csv`: Aggregated global feature weight score arrays, sorted uniformly by cumulative mean absolute impact.
* `shap_summary_plot_*.pdf`: Scaled visual vector graphs displaying global impact profiles (both per-fold isolates and consolidated multi-fold averages).

## 5. Training Size Optimization 

To determine the optimal number of paragraphs  required to successfully train the models, a search of optimal training set size (in paragraphs) is performed across the kinematics data. The scripts are stored within the `OPTIMAL_NUMBER_OF_PARAGRAPHS_SEARCH` workspace folder.

### 5.1. Optimization Protocol
* **Volume Range Verification:** The code evaluates model classification accuracy variations by step-wise increments of the number of paragraphs in training set ranging from 1 to 5 paragraphs per subject.
* **Combinatorial Paragraph Partitioning:** For each target paragraph volume constraint, the pipeline uses the `combinations` utility to sample combinations from a subject's available paragraphs. Five combinations are randomly drawn to assemble strict cross-validation folds, guaranteeing that lines belonging to the same text paragraph do not span across train and test sets simultaneously[cite: 29].
* **Modality Profile Isolation:** Training volume loops verify performance drops and scaling consistency across all individual feature configurations:
  * **Stylus Only Kinematics:** Evaluated with isolated features `[:, :, :7]` using 2048 Conv1D channels and a window length of 1152.
  * **Hand Only Kinematics:** Evaluated with isolated features `[:, :, 7:]` using 1024 Conv1D channels and a window length of 1344.
  * **Combined Hand & Stylus Kinematics:** Evaluated over the full feature block using 512 Conv1D channels and a window length of 1024.
* **Balanced Testing Control:** To prevent evaluation bias caused by structural drops in baseline content, the pipeline dynamically detects missing validation subjects across the sparse combinations and injects single samples balanced via `RandomOverSampler`.

### 5.2. Outputs & Directory Structure
Execution logs and visualization scripts output empirical performance data directly into respective subdirectory locations:
* `/CODE/STYLUS_NUM_TRAIN/`, `/CODE/AUTH_HAND_NUM_TRAIN/`, and `/CODE/HAND_STYLUS_NUM_TRAIN/`: Location of serialized fold models (`*.h5`), split definitions (`_folds.pickle`), and performance verification matrix indices (`*.csv`) mapped per paragraph configuration.
* `centered_boxplot_with_legend.pdf`: An aggregated, publication-grade grouped boxplot figure tracking accuracy metrics (scaled inside a 20% to 100% window) overlaid with trend lines tracking mean changes as a function of the paragraph training volumes.

## 6. Ablation & Normalization Experiments

The execution scripts inside the `ABLATION_EXPERIMENTS/` workspace directory isolate and evaluate the underlying behavioral drivers of the participant-identification framework.

### 6.1. Objectives & Rationale
These experiments assess whether the framework's identification accuracy is driven primarily by nuanced, dynamic behavioral characteristics (**handwriting dynamics**) or if it relies on static, participant-specific cues that provide an easier shortcut to identity matching. Specifically, the network is evaluated against the targeted removal of stable physical body morphologies (e.g., hand width, arm length, arm width) and absolute spatial positioning coordinates within the recording arena. 

The system compares baseline models directly against three explicit experimental strategies:
* **Excluding Explicit Morphology:** Strips permanent anatomical measurements from the feature sets to prevent identification based purely on skeletal dimensions.
* **Coordinate Normalization:** Transforms coordinate spatial channels into local reference metrics (wrist-, palm-, or tablet-normalized frameworks) to eliminate absolute location bias.
* **Excluding Raw Coordinate Channels:** Completely removes raw positioning coordinates to evaluate if high-order kinematic derivatives are sufficient on their own.

### 6.2. Performance Interpretation
* **Large Performance Reductions:** Indicate that the removed structural or positioning group contributes heavily to identification, revealing a reliance on spatial or anatomical shortcuts.
* **Preserved Performance:** Demonstrates that the remaining purely dynamic behavioral signals are robust, secure, and sufficient for identification.

### 6.3. Cross-Modality Ablation Mapping

These tests are executed across three standalone input configurations (**Hand Kinematics**, **Stylus Kinematics**, and **Stylus and Hand Kinematics**) to uncover whether reliance on structural or recording-specific artifacts diverges across sensing modalities.

| Target Modality Configuration | Ablation / Normalization             | Script File Path |
| :--- |:-------------------------------------| :--- |
| **Hand-Only Kinematics** | Static Features Removal (Partial)    | hand_ablation_static.py` |
| | Static Features Removal (All)        | hand_ablation_static_all.py` |
| | Palm-Normalized Spatial Coordinates  | hand_ablation_norm_palm.py` |
| | Wrist-Normalized Spatial Coordinates | hand_ablation_norm_wrist.py` |
| **Stylus-Only Kinematics** | Static Features Removal (Partial)    | stylus_ablation_static.py` |
| | Static Features Removal (All)        | stylus_ablation_static_all.py` |
| | Static Feature Normalized Tablet     | stylus_ablation_norm_static.py` |
| **Combined Hand & Stylus** | Static Features Removal (Partial)    | hand_stylus_ablation_static.py` |
| | Static Features Removal (All)        | hand_stylus_ablation_static_all.py` |
| | Palm-Normalized Spatial Coordinates  | hand_stylus_ablation_norm_palm.py` |
| | Wrist-Normalized Spatial Coordinates | hand_stylus_ablation_norm_wrist.py` |

> **Note on Validation Consistency:** 
> Every script within the ablation suite enforces the baseline protocol's strict cross-validation boundaries, isolating paragraph-level segments to maintain parity and ensure performance variations are directly comparable.