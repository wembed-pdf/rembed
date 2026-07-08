import os
import sys
import pandas as pd # pyright: ignore[reportMissingModuleSource]

import subprocess


dir = os.path.join(os.path.abspath(os.curdir), "plots/scripts")

output_path = os.path.join(dir, "../")

# get path to data folder as argument
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_data_csv>")
    sys.exit(1)

data_csv_path = sys.argv[1]

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
    if "sklearn_balltree" in mapping:
        mapping["sklearn_balltree"] = "Balltree"
    if "atree" in mapping:
        mapping["atree"] = "SPRK"
    return mapping

categories = ["nn", "clustering", "poi"]

df = pd.read_csv(data_csv_path)
df["run_time_ns"] = (df["wall_time_mean_ns"]).round(1)
# remove all _train and .csv from the distribution column
df["distribution"] = df["name"].str.replace("_train", "", regex=False)
df["distribution"] = df["name"].str.replace(".csv", "", regex=False)
df["structure"] = df["data_structure"]

##############################

# --- Generate combined table with runtimes averaged over radii ---

###############################

latex = []
df = df[df["structure"] != "dyn_atree"] 
df = df[df["structure"] != "wembed_snn"]  
df = df[df["structure"] != "quadtree"]  
df = df[df["structure"] != "naive_snn"]  
df = df[df["structure"] != "snn"]  
structures = ['brute-force', 'orthtree', 'kiddo', 'sklearn_balltree', 'py_snn', 'atree']
num_cols = len(structures) + 1
latex.append("\\begin{tabular}{l@{\\,$\\,$}rr@{\\,$\\,$}r@{\\,$\\,$}" + "r@{\\,$\\,$}" * (num_cols-1) + "}") # add extra space between SPRK and SNN
latex.append("\\toprule")   
row_d = ["Data Set", "\\multicolumn{1}{c}{$n$}", "\\multicolumn{1}{c}{$d$}"]
for struct in structures:
    display_name = get_display_names(pd.Series([struct]))[struct]
    row_d.append(f"\\multicolumn{{1}}{{r}}{{\\text{{{display_name}}}}}")
latex.append(" & ".join(row_d) + " \\\\")
latex.append("\\midrule")

category_labels = {
    "nn": "Real-World high-dimensional datasets (in \\qty{}{\\milli\\second}) \\cite{aumullerANNBenchmarks2020}",
    "poi": "Point of Interest Search (in \\qty{}{\\nano\\second}) \\cite{OsmGermany}",
    "clustering": "Clustering (in \\qty{}{\\nano\\second}) \\cite{duaUCI2017}",
}

category_to_table = {
    "nn": "table_4",
    "poi": "table_5",
    "clustering": "table_6",
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

# remove all grid rows from the dataframe
df = df[~df["structure"].str.contains("grid")]

# divide runtime by 1000 * 1000 for nn category to convert to milliseconds and round to 1 decimal place
df["run_time_ns"] = df["run_time_ns"].astype(float)
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
                row.append("oom")
        latex.append(" & ".join(row) + " \\\\")
    if category != categories[-1]:  # Add midrule after each category except the last one
        latex.append("\\midrule")

latex.append("\\bottomrule")
latex.append("\\end{tabular}")

with open(os.path.join(output_path, "table_3.tex"), "w") as f:
    f.write("\n".join(latex))
print("Combined LaTeX table written to", os.path.join(output_path, "table_3.tex"))





###################################

# Appendix Full Tables

###################################



df = df_save.copy()

# divide runtime by 1000 * 1000 for nn category to convert to milliseconds and round to 1 decimal place
df["run_time_ns"] = df["run_time_ns"].astype(float)
df.loc[df["category"] == "nn", "run_time_ns"] = (
    df.loc[df["category"] == "nn", "run_time_ns"] / (1000 * 1000)
).round(2)

# Identify the minimum runtime for each combination of category, distribution and radius
def highlight_fastest_runtime(df):
    df['is_fastest'] = df.groupby(['category', 'distribution', 'radius'])['run_time_ns'].transform(lambda x: x == x.min())
    return df

def highlight_second_fastest_runtime(df):
    df['is_second_fastest'] = df.groupby(['category', 'distribution', 'radius'])['run_time_ns'].transform(lambda x: x == x.nsmallest(2).iloc[-1])
    return df

df = highlight_fastest_runtime(df)
df = highlight_second_fastest_runtime(df)

structure_ordering = structures

poi_dataset_header_mapping = {
    "atm": "ATM (12K) queried from supermarkets",
    "parking": "Parking (769K) queried from hospitals",
    "restaurant": "Restaurant (103K) queried from train stations",
    "busstop": "Bus Stop (761K) queried from train stations",
    "bakery": "Bakery (33K) queried from universities",
}

for category in categories:

    structures = df[df["category"] == category]["structure"].unique()

    # Build LaTeX
    latex = []
    num_cols = len(structures) + 1
    if category != "poi":
        latex.append("\\begin{tabular}{@{}l@{\\,$\\,$}rr@{\\,$\\,$}r@{\\,$\\,$}" + "r@{\\,$\\,$}" * (num_cols-2) + "r@{}}") # add extra space between SPRK and SNN
    else:
        latex.append("\\begin{tabular}{@{}r@{\\,$\\,$}r@{\\,$\\,$}" + "r@{\\,$\\,$}" * (num_cols-2) + "r@{}}") # add extra space between SPRK and SNN
    latex.append("\\toprule")

    # Map data structure names to display names using R script
    # sort display names by structure ordering defined above
    display_names = dict(sorted(get_display_names(structures).items(), key=lambda item: structure_ordering.index(item[0]) if item[0] in structure_ordering else 3))

    df["display_name"] = df["structure"].map(display_names).replace("_", r"\_")

    # Row: structures
    if category != "poi":
        row_d = ["Data Set", "\\multicolumn{1}{c}{$n$}", "\\multicolumn{1}{c}{$d$}", "Radius"]
    else:
        row_d = ["Radius"]
    for display_name in display_names.values():
        row_d.append(f"\\multicolumn{{1}}{{c}}{{\\text{{{display_name}}}}}")
    latex.append(" & ".join(row_d) + " \\\\")

    latex.append("\\midrule")

    distributions = df["distribution"].unique()

    # Data rows
    for distribution in distributions:
        radii = df[df["distribution"] == distribution]["radius"].unique()
        if category == "poi" and distribution in poi_dataset_header_mapping:
            # add a header row for the poi datasets with the name of the dataset and the number of points in the dataset
            dataset_name = distribution
            header = poi_dataset_header_mapping.get(dataset_name, dataset_name)
            if distribution != "atm": 
                latex.append("\\midrule")
            latex.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{{header}}}}} \\\\")
            latex.append("\\midrule")
        for radius in radii:
            subset = df[(df["category"] == category) & (df["distribution"] == distribution) & (df["radius"] == radius)]
            if subset.empty:
                continue

            lable = distribution_labels.get(distribution, distribution)

            if category != "poi":
                # extract N and D from the distribution label
                row = [f"{lable.split('(')[0].replace('_', r'\_')}"]
                if "(" in lable and ")" in lable:
                    n, d = lable.split("(")[1].split(")")[0].split(",")
                    n_number = n[0:-1]  # remove K or M
                    unit = n[-1]
                    row.append(f"\\SI{{{n_number}}}{{{unit}}}")
                    row.append(f"\\num{{{d.strip()}}}")
                else:
                    row.append("-")
                    row.append("-")
            else:
                row = []
                
            row.append(f"\\num{{{radius}}}")
            for display_name in display_names.values():
                struct_subset = subset[subset["display_name"] == display_name]
                if not struct_subset.empty:
                    val = struct_subset["run_time_ns"].values[0]
                    if category != "nn":
                        val = int(round(val, 0))  # Round to nearest integer for non-nn categories
                    is_fastest = struct_subset["is_fastest"].values[0]
                    if is_fastest:
                        row.append(f"\\textbf{{\\num[detect-all]{{{val}}}}}")  # Highlight fastest runtime in bold
                    else:
                        is_second_fastest = struct_subset["is_second_fastest"].values[0]
                        if is_second_fastest:
                            row.append(f"\\underline{{\\num[detect-all]{{{val}}}}}")  # Highlight second fastest runtime with underline
                        else:
                            row.append(f"\\num{{{val}}}")
                else:
                    row.append("oom")
            latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    # output = os.path.join(output_path, f"{category}_extern_table.tex")
    output = os.path.join(output_path, f"{category_to_table[category]}.tex")
    with open(output, "w") as f:
        f.write("\n".join(latex))

    print("LaTeX table written to", output)