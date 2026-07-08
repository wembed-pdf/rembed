import os
import sys
import pandas as pd
import subprocess

def get_display_names(strings):

    path = os.path.join(os.path.abspath(os.curdir), "plots/scripts/R_Interface.R")

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
    return dict(zip(strings, display_names))

dir = os.path.join(os.path.abspath(os.curdir), "plots/scripts")

output_path = os.path.join(dir, "../")

# get path to data folder as argument
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_data_csv>")
    sys.exit(1)

data_csv_path = sys.argv[1]

df = pd.read_csv(data_csv_path)
df["data_structure_name"] = df["data_structure"]
df = df[df["data_structure_name"] != "agrid"]
df = df[df["data_structure_name"] != "dyn_atree"]
df = df[df["data_structure_name"] != "naive_atree"]
df = df[df["data_structure_name"] != "quadtree"]
df = df[df["data_structure_name"] != "snn"]
df = df[df["data_structure_name"] != "wembed_snn"]
df = df[df["data_structure_name"] != "naive_snn"]
df = df[df["data_structure_name"] != "naive_atree_non_progressive"]

# remove rows where the runtime is 0 (mostly quadtrees in >2 dimensions)
df["run_time"] = df["wall_time_mean_ns"]
df = df[df["run_time"] > 1]

# average run_time for each combination of data_structure_name, dim and n
df["dim"] = df["dimension"]
df["n"] = df["node_count"]
df = df.groupby(["data_structure_name", "dim", "n"], as_index=False)["run_time"].mean()

# Sort values
df = df.sort_values(by=["n", "dim"])

# divide run_time by 1000 and round to integer
df["run_time"] = (df["run_time"] / 1000).round(1)

ns = sorted(df["n"].unique())
dims = sorted(df["dim"].unique())
structures = df["data_structure_name"].unique()

#double time of the brute force method to compensate for asymetric queries
df.loc[df["data_structure_name"] == "brute-force", "run_time"] *= 2

# Identify the minimum runtime for each combination of n and dim
def highlight_fastest_runtime(df):
    df['is_fastest'] = df.groupby(['n', 'dim'])['run_time'].transform(lambda x: x == x.min())
    return df

def highlight_second_fastest_runtime(df):
    df['is_second_fastest'] = df.groupby(['n', 'dim'])['run_time'].transform(lambda x: x == x.nsmallest(2).iloc[-1])
    return df

df = highlight_fastest_runtime(df)
df = highlight_second_fastest_runtime(df)

# Build LaTeX
latex = []
num_cols = len(ns) * len(dims)
latex.append("\\begin{tabular}{l" + "r" * num_cols + "}")
latex.append("\\toprule")

# Row: n
row_n = ["num vertices"]
for n in ns:
    row_n.append(f"\\multicolumn{{{len(dims)}}}{{c}}{{\\num{{{n}}}}}")
latex.append(" & ".join(row_n) + " \\\\")

# Row: d
row_d = ["dimensions"]
for _ in ns:
    for d in dims:
        row_d.append(f"\\multicolumn{{1}}{{c}}{{\\num{{{d}}}}}")
latex.append(" & ".join(row_d) + " \\\\")

latex.append("\\midrule")

# Add a category column for grouping data structures (e.g., "KD-Tree", "Own", "Other")
def categorize_structure(name):
    if name in ["kiddo", "sklearn_kdtree", "kd_tree", "sif", "nabo", "nanoflann", "cgal_kdtree", "neighbourhood"]:
        return "KD-Tree"
    elif name.lower() in ["atree", "atree2", "brute-force", "line-lsh", "lsh-2", "quadtree", "grid", "orthtree"]:
        return "Own"
    # elif name.lower() in ["wembed_snn", "snn", "py_snn", "naive_snn"]:
    #     return "SNN"
    else:
        return "Other"
df["category"] = df["data_structure_name"].apply(categorize_structure)

# Map data structure names to display names using R script
display_names = get_display_names(structures)
df["data_structure_name"] = df["data_structure_name"].map(display_names)
structures = df["data_structure_name"].unique()

# order rows by descending performance in 8 dimensions and 100000 vertices
structures = sorted(
    structures,
    key=lambda s: df[(df["data_structure_name"] == s) & (df["dim"] == 8) & (df["n"] == 100000)]["run_time"].values[0]
    if not df[(df["data_structure_name"] == s) & (df["dim"] == 8) & (df["n"] == 100000)].empty
    else float(10)
)
structures.reverse()  # Reverse the order to have the fastest at the bottom

# Data rows
# categories = ["Own", "SNN", "KD-Tree", "Other"]
categories = ["KD-Tree", "Other", "Own"]
for category in categories:
    for struct in structures:
        # print(f"Processing structure: {struct} in category: {category}")
        if not (df[df["data_structure_name"] == struct]["category"] == category).any():
            continue
        # Format name like \text{ds}_1
        name = struct.replace("_", r"\_")
        name = f"$\\text{{{name}}}$"

        row = [name]

        for n in ns:
            for d in dims:
                subset = df[
                    (df["data_structure_name"] == struct) &
                    (df["n"] == n) &
                    (df["dim"] == d)
                ]

                if not subset.empty:
                    val = subset["run_time"].values[0]
                    # Highlight fastest runtime in bold
                    if subset["is_fastest"].values[0]:
                        row.append(f"\\textbf{{\\num[detect-all]{{{val}}}}}")  
                    # Highlight second fastest runtime with underline
                    elif subset["is_second_fastest"].values[0]:
                        row.append(f"\\underline{{\\num[detect-all]{{{val}}}}}")  
                    else:
                        row.append(f"\\num{{{val}}}")
                else:
                    row.append("oom")

        latex.append(" & ".join(row) + " \\\\")
    if category != categories[-1]:  # Add midrule after each category except the last one
        latex.append("\\midrule")

latex.append("\\bottomrule")
latex.append("\\end{tabular}")

output_path = os.path.join(output_path, "table_2.tex")

with open(output_path, "w") as f:
    f.write("\n".join(latex))

print("LaTeX table written to", output_path)