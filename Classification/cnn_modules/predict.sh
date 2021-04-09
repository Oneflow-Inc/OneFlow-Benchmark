# pytorch model download link: https://download.pytorch.org/models/resnet50-19c8e357.pth

PRETRAIN_MODEL_PATH="resnet50-19c8e357.pth"
PREDICT_IMAGE_PATH="data/fish.jpg"


if [ ! -f "$PRETRAIN_MODEL_PATH" ]; then
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth
fi

python3 predict.py --model_path $PRETRAIN_MODEL_PATH --image_path $PREDICT_IMAGE_PATH
