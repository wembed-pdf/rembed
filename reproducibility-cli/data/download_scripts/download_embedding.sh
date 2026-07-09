if [ ! -d "data" ]; then
    echo "Please run this script from the root directory reproducibility-cli."
    exit 1
fi

DOWNLOAD_CACHE="${PWD}/data/.download_cache"

if [ ! -d "$DOWNLOAD_CACHE" ]; then
        echo "Creating download cache directory."
        mkdir $DOWNLOAD_CACHE
fi

EMBEDDING_DATA_DIR="${PWD}/data/embedding/embedding_data"

if [ ! -d "$EMBEDDING_DATA_DIR" ]; then
        echo "Creating directory for embedding data."
        mkdir -p $EMBEDDING_DATA_DIR
fi

echo "=== Downloading compressed embedding datasets to ${DOWNLOAD_CACHE}... ==="

# wget https://zenodo.org/records/21243483/files/embedding_data.zip?download=1 -O $DOWNLOAD_CACHE/embedding_data.zip
unzip $DOWNLOAD_CACHE/embedding_data.zip -d $EMBEDDING_DATA_DIR
mv $EMBEDDING_DATA_DIR/embedding_export/* $EMBEDDING_DATA_DIR
rm -rf $EMBEDDING_DATA_DIRy/embedding_export
mv $EMBEDDING_DATA_DIR/metadata.csv $EMBEDDING_DATA_DIR/../embedding_metadata.csv

echo "=== Finished unpacking embedding datasets to ${EMBEDDING_DATA_DIR} and ${EMBEDDING_DATA_DIR}/../embedding_metadata.csv ==="