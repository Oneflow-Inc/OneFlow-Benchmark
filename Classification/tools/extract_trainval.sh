# usage: sh extract_trainval.sh your_path_to/imagenet
# 参数指定存放imagenet元素数据的文件夹路径

set -e
ROOT_DIR=$1  # your path to imagenet dataset root dir
echo "Imagenet dataset in dir:${ROOT_DIR}"

SYNSETS_FILE="imagenet_lsvrc_2015_synsets.txt"
TRAIN_TARBALL="${ROOT_DIR}/ILSVRC2012_img_train.tar"
TRAIN_OUTPUT_PATH="${ROOT_DIR}/train/"
VALIDATION_TARBALL="${ROOT_DIR}/ILSVRC2012_img_val.tar"
VALIDATION_OUTPUT_PATH="${ROOT_DIR}/validation/"

mkdir -p "${TRAIN_OUTPUT_PATH}"
mkdir -p "${VALIDATION_OUTPUT_PATH}"

# extract .tar file of validation
tar xf "${VALIDATION_TARBALL}" -C "${VALIDATION_OUTPUT_PATH}"

# extract .tar file of train
echo "Uncompressing individual train tar-balls in the training data."

while read SYNSET; do
  # Uncompress into the directory.
  tar xf "${TRAIN_TARBALL}" "${SYNSET}.tar"
  if [ "$?" = "0" ];then
    # Create a directory and delete anything there.
    mkdir -p "${TRAIN_OUTPUT_PATH}/${SYNSET}"
    rm -rf "${TRAIN_OUTPUT_PATH}/${SYNSET}/*"
    echo "Processing: ${SYNSET}"
    tar xf "${SYNSET}.tar" -C "${TRAIN_OUTPUT_PATH}/${SYNSET}/"
    rm -f "${SYNSET}.tar"
    echo "Finished processing: ${SYNSET}"
  else
    echo "${SYNSET}.tar doesn't exist!"
  fi
done < "${SYNSETS_FILE}"