import os
import sys
import pandas as pd
import subprocess

dir = os.path.join(os.path.abspath(os.curdir), "plots/scripts")

output_path = os.path.join(dir, "../")

# get path to data folder as argument
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_data_csv>")
    sys.exit(1)

data_csv_path = sys.argv[1]

df = pd.read_csv(data_csv_path)
df["data_structure_name"] = df["data_structure"]
df["run_time"] = df["wall_time_mean_ns"]
df["dim"] = df["dimension"]
df["n"] = df["node_count"]
embedding_df = df

#double time of the brute force method to compensate for asymetric queries
embedding_df.loc[embedding_df["data_structure_name"] == "brute-force", "run_time"] *= 2

embedding_df = embedding_df.sort_values(by=["n", "dim"])

# average run_time for each combination of data_structure_name, dim and n
embedding_df = embedding_df.groupby(["data_structure_name", "dim", "n"], as_index=False)["run_time"].mean()


# divide run_time by 1000 (to convert from ns to us) and round to integer
embedding_df["run_time"] = (embedding_df["run_time"] / 1000).round(1)

embedding_df["radius"] = "1.0"
embedding_df["distribution"] = embedding_df.apply(lambda row: f"Embedding ({row['n']/1000}k,{row['dim']})", axis=1)
embedding_df["structure"] = embedding_df["data_structure_name"]
embedding_df["wall_time_ns"] = embedding_df["run_time"]
embedding_df["wall_time_std_ns"] = 0  # no std available
embedding_df["instructions"] = 0  # no instructions available
embedding_df["instructions_std"] = 0
embedding_df["cycles"] = 0  # no cycles available
embedding_df["cycles_std"] = 0
embedding_df["samples"] = 1  # only one sample available
embedding_df["category"] = "embedding"

def get_display_names(strings):

    path = os.path.join(dir, "R_Interface.R")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"R script not found at {path}")

    result = subprocess.run(
        ["Rscript", path] + strings.tolist(),
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    display_names = result.stdout.strip().split("\n")
    mapping = dict(zip(strings, display_names))
    mapping["sklearn_balltree"] = "Balltree"
    mapping["atree"] = "SPRK"
    mapping["py_snn"] = "SNN"
    mapping["snn"] = "our SNN"
    mapping["naive_snn"] = "our naive SNN"
    return mapping

categories = ["nn", "clustering", "poi"]

df = embedding_df
df["run_time_ns"] = df["wall_time_ns"]
# remove all _train and .csv from the distribution column
df["distribution"] = df["distribution"].str.replace("_train", "", regex=False)
df["distribution"] = df["distribution"].str.replace(".csv", "", regex=False)

# --- Generate combined table with runtimes averaged over radii ---

# structures = ["brute-force", "orthtree", "kiddo", "sklearn_balltree", "py_snn", "atree"]
structures = ["brute-force", "py_snn", "snn", "atree"]

latex = []

df = df[df["structure"].isin(structures)]  # remove all structures that are not in the list of structures to include in the table

# Apppend \sisetup{detect-weight=true, detect-family=true} to use bold fonts for SI numbers
latex.append("\\sisetup{detect-weight=true, detect-family=true}")

num_cols = len(structures) + 1
latex.append("\\begin{tabular}{l@{\\,$\\,$}rr@{\\,$\\,$}r@{\\,$\\,$}" + "r@{\\,$\\,$}" * (num_cols-1) + "}") # add extra space between SPRK and SNN
latex.append("\\toprule")   
# Row: structures
row_d = ["Data Set", "\\multicolumn{1}{c}{$n$}", "\\multicolumn{1}{c}{$d$}"]
for struct in structures:
    display_name = get_display_names(pd.Series([struct]))[struct]
    row_d.append(f"\\multicolumn{{1}}{{r}}{{\\text{{{display_name}}}}}")
    # row_d.append(f"{display_name}")
latex.append(" & ".join(row_d) + " \\\\")
latex.append("\\midrule")

# Data sets are available at~\cite{aumullerANNBenchmarks2020},~\cite{duaUCI2017},~\cite{OsmGermany}
category_labels = {
    "nn": "Real-World high-dimensional datasets (in \\qty{}{\\milli\\second}) \\cite{aumullerANNBenchmarks2020}",
    "poi": "Point of Interest Search (in \\qty{}{\\nano\\second}) \\cite{OsmGermany}",
    "clustering": "Clustering (in \\qty{}{\\nano\\second}) \\cite{duaUCI2017}",
    "embedding": "Embedding (in \\qty{}{\\micro\\second})"
}

distribution_labels = {
    "banknote": "Banknote (1.5k, 4)",
    "phoneme": "Phoneme (4.5k, 256)",
    "wine": "Wine (0.2k, 13)",
    "dermatology": "Dermatology (0.4k, 34)",
    "ecoli": "Ecoli (0.3k, 7)",
    "sift_large": "SIFT1M (100k, 128)",
    "sift": "SIFT10K (25k, 128)",
    "gist": "GIST (1M, 960)",
    "glo": "GloVe100 (1M, 100)",
    "deep": "Deep1B (10M, 96)",
    "fmn": "F-MNIST (60k, 784)",
    "atm": "ATM (12k, 2)",
    "pharmacy": "Pharmacy (17k, 2)",
    "parking": "Parking (769k, 2)",
    "restaurant": "Restaurant (103k, 2)",
    "busstop": "Bus Stop (761k, 2)",
    "hospital": "Hospital (2k, 2)",
    "bakery": "Bakery (33k, 2)",
}


df_save = df.copy()

# divide runtime by 1000 * 1000 for nn category to convert to milliseconds and round to 1 decimal place
df.loc[df["category"] == "nn", "run_time_ns"] = (
    df.loc[df["category"] == "nn", "run_time_ns"] / (1000 * 1000)
).round(1)

for category in categories:
    latex.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{{category_labels[category]}}}}} \\\\")
    latex.append("\\midrule")

    df_subset = df[df["category"] == category]
    distributions = df_subset["distribution"].unique()
    # average runtime over radii for each distribution and structure
    df_avg = df_subset.groupby(["distribution", "structure"], as_index=False)["run_time_ns"].mean().round(3)
    df_avg["run_time_ns"] = df_avg["run_time_ns"].round(1)

    df_avg["is_fastest"] = df_avg.groupby("distribution")["run_time_ns"].transform(lambda x: x == x.min())
    df_avg["is_second_fastest"] = df_avg.groupby("distribution")["run_time_ns"].transform(lambda x: x == x.nsmallest(2).iloc[-1])
    for distribution in distributions:
        df_dist = df_avg[df_avg["distribution"] == distribution]
        lable = distribution_labels.get(distribution, distribution)
        row = [f"{lable.split('(')[0].replace('_', r'\_')}"]
        # extract N and D from the distribution label
        if "(" in lable and ")" in lable:
            n, d = lable.split("(")[1].split(")")[0].split(",")
            n_number = n[0:-1]  # remove K or M
            unit = n[-1]
            row.append(f"\\SI{{{n_number}}}{{{unit}}}")
            row.append(f"\\num{{{d.strip()}}}")
        else:
            row.append("-")
            row.append("-")
        # row = [f"{lable}"]
        for struct in structures:
            df_struct = df_dist[df_dist["structure"] == struct]
            if not df_struct.empty:
                val = df_struct["run_time_ns"].values[0]
                is_fastest = df_struct["is_fastest"].values[0]
                if is_fastest:
                    row.append(f"\\textbf{{\\num[detect-all]{{{val}}}}}")  # Highlight fastest runtime in bold
                else:
                    is_second_fastest = df_struct["is_second_fastest"].values[0]
                    if is_second_fastest:
                        row.append(f"\\underline{{\\num[detect-all]{{{val}}}}}")  # Highlight second fastest runtime with underline
                    else:
                        row.append(f"\\num{{{val}}}")
            else:
                row.append("-")
        latex.append(" & ".join(row) + " \\\\")
    if category != categories[-1]:  # Add midrule after each category except the last one
        latex.append("\\midrule")

latex.append("\\bottomrule")
latex.append("\\end{tabular}")

with open(os.path.join(output_path, "table_7.tex"), "w") as f:
    f.write("\n".join(latex))
print("Combined LaTeX table written to", os.path.join(output_path, "table_7.tex"))