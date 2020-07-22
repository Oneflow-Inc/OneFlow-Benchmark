rm -rf core.* 

# Set dir path of your model
MODEL_LOAD_DIR="../../../model/resnet_v15_of_best_model_val_top1_77318"

python3 of_cnn_inference.py \
    --image_path="test_img/tiger.jpg" \
    --log_dir="inference_output" \
    --model_load_dir=$MODEL_LOAD_DIR