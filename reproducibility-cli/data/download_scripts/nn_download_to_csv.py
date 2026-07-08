# This script is based on the loader code from https://github.com/nla-group/snn/

import h5py # type: ignore
import pandas as pd
import os
import numpy as np

if not os.path.basename(os.getcwd()) == "reproducibility-cli":
    raise Exception("Please run this script from the reproducibility-cli directory root.")

if not os.path.exists("data/realworld/nearest_neighbor_data"):
    os.makedirs("data/realworld/nearest_neighbor_data")

print("=== Converting nearest neighbor files to csv ... this may take quite a while and need about 50G of disk space. ===")

DOWNLOAD_CACHE = "data/realworld/nearest_neighbor_data/.download_cache"
NEAREST_NEIGHBOR_DATA_DIR = "data/realworld/nearest_neighbor_data"

def hdf5_to_csv(hdf5_file_name, csv_file_name):
    with h5py.File(f"{DOWNLOAD_CACHE}/{hdf5_file_name}", "r") as f:
        # distances, neighbors, train, test
        test_data = f["test"][:]
        train_data = f["train"][:]
        test_df = pd.DataFrame(test_data)
        train_df = pd.DataFrame(train_data)
        # skip first row of the dataframe (column numbers), save to 18 decimal places for numpy compatibility
        test_df.to_csv(f"{NEAREST_NEIGHBOR_DATA_DIR}/{csv_file_name}_query.csv", index=False, header=False, float_format="%.18e")
        train_df.to_csv(f"{NEAREST_NEIGHBOR_DATA_DIR}/{csv_file_name}_train.csv", index=False, header=False, float_format="%.18e")

print("Converting fashion-mnist-784-euclidean.hdf5 to csv ...")
hdf5_to_csv("fashion-mnist-784-euclidean.hdf5", "fmn")
print(f'Converted fashion-mnist-784-euclidean.hdf5 to {NEAREST_NEIGHBOR_DATA_DIR}/fmn_query.csv and {NEAREST_NEIGHBOR_DATA_DIR}/fmn_train.csv')
print("Converting gist-960-euclidean.hdf5 to csv ...")
hdf5_to_csv("gist-960-euclidean.hdf5", "gist")
print(f'Converted gist-960-euclidean.hdf5 to {NEAREST_NEIGHBOR_DATA_DIR}/gist_query.csv and {NEAREST_NEIGHBOR_DATA_DIR}/gist_train.csv')
print("Converting glove-100-angular.hdf5 to csv ...")
hdf5_to_csv("glove-100-angular.hdf5", "glo")
print(f'Converted glove-100-angular.hdf5 to {NEAREST_NEIGHBOR_DATA_DIR}/glo_query.csv and {NEAREST_NEIGHBOR_DATA_DIR}/glo_train.csv')
print("Converting deep-image-96-angular.hdf5 to csv ...")
hdf5_to_csv("deep-image-96-angular.hdf5", "deep")
print(f'Converted deep-image-96-angular.hdf5 to {NEAREST_NEIGHBOR_DATA_DIR}/deep_query.csv and {NEAREST_NEIGHBOR_DATA_DIR}/deep_train.csv')

def fvec_to_csv(fvec_file, csv_file):
    file = np.fromfile(fvec_file, dtype='int32')
    dim = file[0]
    vectors = file.reshape(-1, dim + 1)[:, 1:].view('float32')
    df = pd.DataFrame(vectors)
    df.to_csv(csv_file, index=False, header=False, float_format="%.18e")

print("Converting siftsmall_learn.fvecs and siftsmall_query.fvecs to csv ...")
fvec_to_csv(f"{DOWNLOAD_CACHE}/siftsmall/siftsmall_learn.fvecs", f"{NEAREST_NEIGHBOR_DATA_DIR}/sift_train.csv")
fvec_to_csv(f"{DOWNLOAD_CACHE}/siftsmall/siftsmall_query.fvecs", f"{NEAREST_NEIGHBOR_DATA_DIR}/sift_query.csv")
print(f'Converted siftsmall_learn.fvecs to {NEAREST_NEIGHBOR_DATA_DIR}/sift_train.csv and siftsmall_query.fvecs to {NEAREST_NEIGHBOR_DATA_DIR}/sift_query.csv')

print("Converting sift_learn.fvecs and sift_query.fvecs to csv ...")
fvec_to_csv(f"{DOWNLOAD_CACHE}/sift/sift_learn.fvecs", f"{NEAREST_NEIGHBOR_DATA_DIR}/sift_large_train.csv")
fvec_to_csv(f"{DOWNLOAD_CACHE}/sift/sift_query.fvecs", f"{NEAREST_NEIGHBOR_DATA_DIR}/sift_large_query.csv")
print(f'Converted sift_learn.fvecs to {NEAREST_NEIGHBOR_DATA_DIR}/sift_large_train.csv and sift_query.fvecs to {NEAREST_NEIGHBOR_DATA_DIR}/sift_large_query.csv')

print("=== Finished converting nearest neighbor files to csv ===")
