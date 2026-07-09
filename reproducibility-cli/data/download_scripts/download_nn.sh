# This script is based on the loader code from https://github.com/nla-group/snn/

if [ ! -d "data" ]; then
    echo "Please run this script from the root directory reproducibility-cli."
    exit 1
fi

REALWORLD_DIR="${PWD}/data/realworld/"

if [ ! -d "$REALWORLD_DIR" ]; then
        echo "Creating directory of realworld data."
        mkdir $REALWORLD_DIR
fi

DOWNLOAD_CACHE="${REALWORLD_DIR}/.download_cache"

if [ ! -d "$DOWNLOAD_CACHE" ]; then
        echo "Creating download cache directory."
        mkdir $DOWNLOAD_CACHE
fi

echo "=== Downloading compressed nearest neighbor datasets to ${DOWNLOAD_CACHE}... ==="

echo "Download Euclidean Nearest Neighbour datasets."

wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
wget http://ann-benchmarks.com/gist-960-euclidean.hdf5

tar -xf siftsmall.tar.gz
tar -xf sift.tar.gz
rm -rf siftsmall.tar.gz sift.tar.gz
mv fashion-mnist-784-euclidean.hdf5 $DOWNLOAD_CACHE/fashion-mnist-784-euclidean.hdf5
mv siftsmall $DOWNLOAD_CACHE/siftsmall
mv sift $DOWNLOAD_CACHE/sift
mv gist-960-euclidean.hdf5 $DOWNLOAD_CACHE/gist-960-euclidean.hdf5

echo "Download Angular Nearest Neighbour datasets."

wget http://ann-benchmarks.com/glove-100-angular.hdf5
wget http://ann-benchmarks.com/deep-image-96-angular.hdf5
mv glove-100-angular.hdf5 $DOWNLOAD_CACHE/glove-100-angular.hdf5
mv deep-image-96-angular.hdf5 $DOWNLOAD_CACHE/deep-image-96-angular.hdf5

python data/download_scripts/download_to_csv.py