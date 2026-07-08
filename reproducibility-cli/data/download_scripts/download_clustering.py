# This script is based on the loader code from https://github.com/nla-group/snn/

import h5py # type: ignore
import pandas as pd
import os
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris, load_wine, load_digits, fetch_openml

if not os.path.basename(os.getcwd()) == "reproducibility-cli":
    raise Exception("Please run this script from the reproducibility-cli directory root.")

if not os.path.exists("data/realworld/clustering_data"):
    os.makedirs("data/realworld/clustering_data")

print("=== Downloading clustering files ... ===")

CLUSTERING_DATA = "data/realworld/clustering_data"

banknote_data = fetch_openml(name='banknote-authentication', version=1, as_frame=True)
banknote_train = banknote_data.data.values
banknote_train = (banknote_train - banknote_train.mean(axis=0)) / banknote_train.std(axis=0) 
pd.DataFrame(banknote_train).to_csv(f"{CLUSTERING_DATA}/banknote.csv", index=False, header=False, float_format="%.18e")
print(f'Downloaded banknote data to {CLUSTERING_DATA}/banknote.csv')

dermatology_data = fetch_openml(name='dermatology', version=1, as_frame=True)
dermatology_train = dermatology_data.data.values
dermatology_train = dermatology_train.astype(float)
dermatology_train = dermatology_train[np.isnan(dermatology_train).sum(1) == 0,:]
dermatology_train = (dermatology_train - dermatology_train.mean(axis=0)) / dermatology_train.std(axis=0) 
pd.DataFrame(dermatology_train).to_csv(f"{CLUSTERING_DATA}/dermatology.csv", index=False, header=False, float_format="%.18e")
print(f'Downloaded dermatology data to {CLUSTERING_DATA}/dermatology.csv')

ecoli_data = fetch_openml(name='ecoli', version=1, as_frame=True)
ecoli_train = ecoli_data.data.values
ecoli_train = (ecoli_train - ecoli_train.mean(axis=0)) / ecoli_train.std(axis=0) 
pd.DataFrame(ecoli_train).to_csv(f"{CLUSTERING_DATA}/ecoli.csv", index=False, header=False, float_format="%.18e")
print(f'Downloaded ecoli data to {CLUSTERING_DATA}/ecoli.csv')

phoneme_data = fetch_openml(name='phoneme', version=1, as_frame=True)
phoneme_train = phoneme_data.data.values
phoneme_train = phoneme_train.astype(float)
phoneme_train = phoneme_train[~np.isnan(phoneme_train).any(axis=1), :]
phoneme_train = (phoneme_train - phoneme_train.mean(axis=0)) / phoneme_train.std(axis=0) 
pd.DataFrame(phoneme_train).to_csv(f"{CLUSTERING_DATA}/phoneme.csv", index=False, header=False, float_format="%.18e")
print(f'Downloaded phoneme data to {CLUSTERING_DATA}/phoneme.csv')

wine_data = fetch_openml(name='wine', version=1, as_frame=True)
wine_train = wine_data.data.values
wine_train = (wine_train - wine_train.mean(axis=0)) / wine_train.std(axis=0) 
pd.DataFrame(wine_train).to_csv(f"{CLUSTERING_DATA}/wine.csv", index=False, header=False, float_format="%.18e")
print(f'Downloaded wine data to {CLUSTERING_DATA}/wine.csv')

print("=== Finished downloading clustering files ===")

