if [ ! -d "data" ]; then
    echo "Please run this script from the root directory reproducibility-cli."
    exit 1
fi

DOWNLOAD_CACHE="${PWD}/data/.download_cache"

if [ ! -d "$DOWNLOAD_CACHE" ]; then
        echo "Creating download cache directory."
        mkdir $DOWNLOAD_CACHE
fi

DISTRIBUTIONS_DATA_DIR="${PWD}/data/distributions/"

if [ ! -d "$DISTRIBUTIONS_DATA_DIR" ]; then
        echo "Creating directory for distributions data."
        mkdir -p $DISTRIBUTIONS_DATA_DIR
fi

echo "=== Downloading compressed distributions datasets to ${DOWNLOAD_CACHE}... ==="

# wget https://zenodo.org/records/21243483/files/distributions.zip?download=1 -O $DOWNLOAD_CACHE/distributions.zip
# unzip $DOWNLOAD_CACHE/distributions.zip -d $DISTRIBUTIONS_DATA_DIR
mv $DISTRIBUTIONS_DATA_DIR/distribution_data $DISTRIBUTIONS_DATA_DIR/distributions_data

echo "=== Finished unpacking distributions datasets to ${DISTRIBUTIONS_DATA_DIR} ==="