rm -rf core.* 

# Set up model path, e.g. :  vgg16_of_best_model_val_top1_721  alexnet_of_best_model_val_top1_54762 
MODEL_LOAD_DIR="resnet_v15_of_best_model_val_top1_77318"

python3 of_cnn_inference.py \
    --model="resnet50" \
    --image_path="data/fish.jpg" \
    --model_load_dir=$MODEL_LOAD_DIR