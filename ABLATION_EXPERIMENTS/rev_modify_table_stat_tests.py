import itertools
import os

import pandas as pd
import scipy.stats as st

# 1. Define local file names (or absolute paths)
files = {
    "Stylus": "/PAST_RESULTS/auth_ch1_2048_k1_100_w_1152_stylus.csv",
    "Hand": "/PAST_RESULTS/auth_ch1_1024_k1_100_w_1344_hand.csv",
    "Stylus & Hand": "/PAST_RESULTS/auth_ch1_512_k1_100_w_1024_stylus_hand.csv",
    "EMG": "/PAST_RESULTS/auth_ch1_576_k1_10_w_1000_emg.csv"
}

# 2. Metadata for labels
metadata = {
    "Stylus": {"label": "Stylus kinematics", "n": 7},
    "Hand": {"label": "Hand kinematics", "n": 110},
    "Stylus & Hand": {"label": "Stylus \\& Hand kinematics", "n": 117},
    "EMG": {"label": "Surface EMG", "n": 3}
}

metrics = ['ACC', 'PRECISION', 'RECALL', 'F1']

# Load data and compute basic statistics (no stars)
stats_dict = {}
for name, file in files.items():
    if not os.path.exists(file):
        raise FileNotFoundError(f"Could not find {file}. Ensure the path is correct.")

    df = pd.read_csv(file)
    stats_dict[name] = {}

    for m in metrics:
        vals = df[m] * 100  # Convert to percentages
        mean = vals.mean()
        std = vals.std()
        ci = st.t.interval(0.95, df=len(vals) - 1, loc=mean, scale=st.sem(vals))

        stats_dict[name][m] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'vals': vals,
            'raw_df': df  # Save raw dataframe for fold-level output
        }

# ==========================================
# OUTPUT 1: Clean Performance Table (LaTeX)
# ==========================================
main_rows = []
for name in metadata.keys():
    info = metadata[name]
    label = info["label"]
    n = info["n"]

    mean_std_strs = []
    ci_strs = []

    for m in metrics:
        stats = stats_dict[name][m]
        mean_std_strs.append(f"${stats['mean']:.2f} \\pm {stats['std']:.2f}$")
        ci_strs.append(f"$[{stats['ci_lower']:.1f}, {stats['ci_upper']:.1f}]$")

    main_rows.append(f"\t\t{label} & {n} & " + " & ".join(mean_std_strs) + " \\\\")
    main_rows.append(f"\t\t & & " + " & ".join(ci_strs) + " \\\\")

clean_table = f"""\\begin{{table}}[h!]
\t\\centering
\t\\begin{{tabular}}{{l c c c c c}}
\t\t\\hline
\t\t\\textbf{{Modality}} & \\textbf{{$\\boldsymbol{{n}}$}} & \\textbf{{Acc.,\\%}} & \\textbf{{Prec.,\\%}} & \\textbf{{Rec.,\\%}} & \\textbf{{F1,\\%}} \\\\
\t\t\\hline
{main_rows[0]}
{main_rows[1]}
\t\t \\addlinespace
{main_rows[2]}
{main_rows[3]}
\t\t \\addlinespace
{main_rows[4]}
{main_rows[5]}
\t\t \\addlinespace
{main_rows[6]}
{main_rows[7]}
\t\t\\hline
\t\\end{{tabular}}
\t\\caption{{Model performance evaluation for different modalities and number of features ($n$) in terms of accuracy (Acc.), precision (Prec.), recall (Rec.), and F1-score (F1).}}
\t\\label{{tab:tab_3_model_performance}}
\\end{{table}}"""

with open("RESULTS/summary_table.tex", "w") as f:
    f.write(clean_table)

# ==========================================
# OUTPUT 2: Pairwise Comparison Table (LaTeX)
# ==========================================
pairwise_pairs = list(itertools.combinations(files.keys(), 2))
# ==========================================
#  NORMALITY CHECK BEFORE PAIRWISE COMPARISON: Shapiro-Wilk
# ==========================================
alpha = 0.05
print("\n" + "=" * 18 + " SHAPIRO-WILK NORMALITY TESTS " + "=" * 18)
print("Normality is assessed for the paired fold-wise differences (alpha = 0.05).")

for mod1, mod2 in pairwise_pairs:
    m1_label = metadata[mod1]["label"].replace("\\&", "&")
    m2_label = metadata[mod2]["label"].replace("\\&", "&")
    print(f"\n{m1_label} vs. {m2_label}")

    for m in metrics:
        v1 = stats_dict[mod1][m]['vals']
        v2 = stats_dict[mod2][m]['vals']
        differences = (v1 - v2).dropna()
        shapiro_w, shapiro_p = st.shapiro(differences)
        conclusion = (
            "normality not rejected"
            if shapiro_p >= alpha
            else "normality rejected"
        )
        print(
            f"  {m:<9} W = {shapiro_w:.4f}, p = {shapiro_p:.4f} "
            f"-> {conclusion}"
        )

print("=" * 75)




pairwise_rows = []


def format_pairwise_cell(t_stat, df, p_val):
    if p_val < 0.001:
        p_str = "< 0.001^{***}"
    elif p_val < 0.01:
        p_str = f"= {p_val:.3f}^{{**}}"
    elif p_val < 0.05:
        p_str = f"= {p_val:.3f}^{{*}}"
    else:
        p_str = f"= {p_val:.3f}^{{\\text{{ns}}}}"
    return f"t({df}) = {t_stat:.2f}, p {p_str}"


for mod1, mod2 in pairwise_pairs:
    m1_label = metadata[mod1]["label"]
    m2_label = metadata[mod2]["label"]
    pair_label = f"{m1_label} vs. {m2_label}"

    p_strs = []
    for m in metrics:
        v1 = stats_dict[mod1][m]['vals']
        v2 = stats_dict[mod2][m]['vals']
        stat, p = st.ttest_rel(v1, v2)
        df_val = len(v1) - 1
        p_strs.append(f"${format_pairwise_cell(stat, df_val, p)}$")

    pairwise_rows.append(f"\t\t{pair_label} & " + " & ".join(p_strs) + " \\\\")

pairwise_table = f"""\\begin{{table}}[h!]
\t\\centering
\t\\begin{{tabular}}{{l c c c c}}
\t\t\\hline
\t\t\\textbf{{Comparison}} & \\textbf{{Acc. ($t$, $p$-val)}} & \\textbf{{Prec. ($t$, $p$-val)}} & \\textbf{{Rec. ($t$, $p$-val)}} & \\textbf{{F1 ($t$, $p$-val)}} \\\\
\t\t\\hline
""" + "\n".join(pairwise_rows) + """
\t\t\\hline
\t\\end{{tabular}}
\t\\caption{{Pairwise statistical comparison of metrics across modalities using two-tailed paired $t$-tests ($df = 4$). Superscripts denote significance thresholds: $^{***}$ $p < 0.001$, $^{**}$ $p < 0.01$, $^{*}$ $p < 0.05$, and $^{\\text{{ns}}}$ $p \\ge 0.05$.}}
\t\\label{{tab:pairwise_tests}}
\\end{{table}}"""

with open("RESULTS/pairwise_table.tex", "w") as f:
    f.write(pairwise_table)


# ==========================================
# OUTPUT 3: New Fold-Level Table (LaTeX)
# ==========================================
fold_rows = []
for name in metadata.keys():
    label = metadata[name]["label"]
    n = metadata[name]["n"]
    # Retrieve raw dataframe saved in stats_dict to access fold-by-fold entries
    df = stats_dict[name]['ACC']['raw_df']

    for idx, row in df.iterrows():
        fold = int(row['FOLD'])
        acc = row['ACC'] * 100
        prec = row['PRECISION'] * 100
        rec = row['RECALL'] * 100
        f1 = row['F1'] * 100

        if fold == 0:
            # Multirow grouping setup for both the modality and feature size (n)
            # Modality is rotated 90 degrees
            row_str = f"\t\t\\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{label}}}}} & \\multirow{{5}}{{*}}{{{n}}} & {fold} & {acc:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f} \\\\"
        else:
            row_str = f"\t\t & & {fold} & {acc:.2f} & {prec:.2f} & {rec:.2f} & {f1:.2f} \\\\"
        fold_rows.append(row_str)

    fold_rows.append("\t\t\\hline")

fold_level_table = f"""\\begin{{table}}[h!]
\t\\centering
\t\\begin{{tabular}}{{c c c c c c c}}
\t\t\\hline
\t\t\\textbf{{Modality}} & \\textbf{{$\\boldsymbol{{n}}$}} & \\textbf{{Fold}} & \\textbf{{Acc.,\\%}} & \\textbf{{Prec.,\\%}} & \\textbf{{Rec.,\\%}} & \\textbf{{F1,\\%}} \\\\
\t\t\\hline
""" + "\n".join(fold_rows) + """
\t\\end{{tabular}}
\t\\caption{{Fold-by-fold classification metrics for each evaluated modality across the 5-fold cross-validation. Values represent exact percentage scores for each metric per partition. Modalities are visually grouped utilizing vertical rotated headers.}}
\t\\label{{tab:fold_level_metrics}}
\\end{{table}}"""

with open("RESULTS/fold_level_table.tex", "w") as f:
    f.write(fold_level_table)

print("\nProcessing complete! The following files have been saved in your directory:")
print(" - summary_table.tex")
print(" - pairwise_table.tex")
print(" - fold_level_table.tex")