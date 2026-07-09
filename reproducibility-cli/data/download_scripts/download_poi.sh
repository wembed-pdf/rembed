if [ ! -d "data" ]; then
    echo "Please run this script from the root directory reproducibility-cli."
    exit 1
fi

REALWORLD_DIR="${PWD}/data/realworld/"

if [ ! -d "$REALWORLD_DIR" ]; then
        echo "Creating directory of realworld data."
        mkdir $REALWORLD_DIR
fi

DOWNLOAD_CACHE="${PWD}/data/.download_cache"

if [ ! -d "$DOWNLOAD_CACHE" ]; then
        echo "Creating download cache directory."
        mkdir $DOWNLOAD_CACHE
fi

POI_DATA_DIR="${REALWORLD_DIR}/poi_data"

if [ ! -d "$POI_DATA_DIR" ]; then
        echo "Creating directory for poi data."
        mkdir -p $POI_DATA_DIR
fi

echo "=== Downloading compressed poi datasets to ${DOWNLOAD_CACHE}... ==="

wget https://zenodo.org/records/TODO/files/poi.zip?download=1 -O $DOWNLOAD_CACHE/poi.zip
unzip $DOWNLOAD_CACHE/poi.zip -d $POI_DATA_DIR

echo "=== Finished unpacking poi datasets to ${POI_DATA_DIR} ==="