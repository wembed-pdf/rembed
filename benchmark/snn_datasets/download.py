
import os
import subprocess
import tarfile
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASETS = [
    "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
    "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
    "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "http://ann-benchmarks.com/deep-image-96-angular.hdf5"
]

DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "extracted"
CSV_DIR = "csv_output"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# -----------------------------
# Download files
# -----------------------------
def download(url):
    filename = os.path.join(DOWNLOAD_DIR, url.split("/")[-1])
    if os.path.exists(filename):
        print(f"[SKIP] {filename} already exists")
        return filename

    print(f"[DOWNLOAD] {url}")
    subprocess.run(["wget", "-O", filename, url], check=True)
    return filename


# -----------------------------
# Extract tar.gz
# -----------------------------
def extract_tar(file_path):
    print(f"[EXTRACT] {file_path}")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)


# -----------------------------
# Save CSV
# -----------------------------
def save_csv(data, name, suffix):
    output_file = os.path.join(CSV_DIR, f"{name}_{suffix}.csv")
    print(f"[SAVE] {output_file}")
    pd.DataFrame(data).to_csv(output_file, index=False)


# -----------------------------
# Process HDF5 datasets
# -----------------------------
def process_hdf5(file_path):
    print(f"[PROCESS HDF5] {file_path}")
    name = os.path.basename(file_path).replace(".hdf5", "")

    with h5py.File(file_path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])

        save_csv(train, name, "train")
        save_csv(test, name, "test")


# -----------------------------
# Process SIFT format (fvecs/ivecs)
# -----------------------------
def read_fvecs(filename):
    vectors = []
    with open(filename, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
            vectors.append(vec)
    return np.array(vectors)


def process_sift(folder):
    print(f"[PROCESS SIFT] {folder}")

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".fvecs"):
            data = read_fvecs(path)
            name = file.replace(".fvecs", "")

            # heuristic split (80/20)
            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]

            save_csv(train, name, "train")
            save_csv(test, name, "test")


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    downloaded_files = []

    for url in DATASETS:
        file_path = download(url)
        downloaded_files.append(file_path)

    # Extract archives
    for file_path in downloaded_files:
        if file_path.endswith(".tar.gz"):
            extract_tar(file_path)

    # Process files
    for file_path in downloaded_files:
        if file_path.endswith(".hdf5"):
            process_hdf5(file_path)

    # Process extracted SIFT data
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for d in dirs:
            process_sift(os.path.join(root, d))


if __name__ == "__main__":
    main()
