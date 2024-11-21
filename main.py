from utils.helper import set_seed
set_seed(42)

from data.dataloader import get_dataloaders
from models.unet import UNet
from utils.trainer import UNetTrainer
from utils.helper import infer_test_set_color
from utils.mask2csv import convert_infer_to_csv
from config import *

from huggingface_hub import HfApi, login
import argparse
parser = argparse.ArgumentParser(description="Machine Translation Training Script")
parser.add_argument("--huggingface_token", type=str, required=True, help="Hugging Face token for authentication")
args = parser.parse_args()

def main():
    # Set name of experiment
    experiment_name = EXPERIMENT_NAME
    
    # Get dataloaders
    dataloaders = get_dataloaders(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, 
                                  batch_size=BATCH_SIZE, img_size=IMG_SIZE, val_split=VAL_SPLIT, augmentations=AUGMENTATIONS)
    
    # Initialize model
    model = UNet(in_channels=IN_CHANNELS,
                 out_channels=OUT_CHANNELS,
                 num_layers=NUM_LAYERS,
                 base_channels=BASE_CHANNELS,
                 kernel_size=KERNEL_SIZE,
                 padding=PADDING,
                 dropout_rate=DROPOUT_RATE,
                 upsampling_method=UPSAMPLING_METHOD)
    
    # Initialize trainer
    trainer = UNetTrainer(model=model,
                          name=experiment_name,
                          learning_rate=LEARNING_RATE,
                          device=DEVICE)
    
    # Train model
    trainer.train(dataloaders, num_epochs=NUM_EPOCHS)
    
    # Infer test set
    infer_test_set_color(model, TEST_IMAGE_DIR, INFER_IMAGE_DIR, DEVICE, input_size=IMG_SIZE, threshold=0.5)
    
    # Change result
    convert_infer_to_csv(INFER_IMAGE_DIR, RESULT_DIR)
    
    
if __name__ == '__main__':
    main()