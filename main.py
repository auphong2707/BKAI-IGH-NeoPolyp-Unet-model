from utils.helper import set_seed, infer_and_save
set_seed(464562)

from data.dataloader import get_dataloaders
from models.unet import UNet
from utils.trainer import UNetTrainer
from utils.mask2csv import convert_infer_to_csv
from config import *
import segmentation_models_pytorch as smp

from huggingface_hub import HfApi, login
import argparse
parser = argparse.ArgumentParser(description="Machine Translation Training Script")
parser.add_argument("--huggingface_token", type=str, required=True, help="Hugging Face token for authentication")
parser.add_argument("--wandb_key", type=str, required=True, help="Wandb key for logging")
args = parser.parse_args()

def main():
    # Set name of experiment
    experiment_name = EXPERIMENT_NAME
    
    # Get dataloaders
    dataloaders = get_dataloaders(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, 
                                  batch_size=BATCH_SIZE, img_size=IMG_SIZE, val_split=VAL_SPLIT, augmentations=AUGMENTATIONS)
    
    # Initialize model
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=3     
    )
    
    # Initialize trainer
    trainer = UNetTrainer(model=model,
                          name=experiment_name,
                          learning_rate=LEARNING_RATE,
                          wandb_key=args.wandb_key,
                          device=DEVICE,)
    
    # Train model
    trainer.train(dataloaders['train'], dataloaders['val'], n_epochs=NUM_EPOCHS)
    
    # Infer and save masks
    infer_and_save(model, TEST_IMAGE_DIR, f'results/{EXPERIMENT_NAME}/infer_image/', DEVICE)
    
    # Change result
    convert_infer_to_csv(f'results/{EXPERIMENT_NAME}/infer_image/', 'results/submission.csv')
    
    
    # Upload results to Hugging Face
    login(token=args.huggingface_token)
    
    api = HfApi()
    api.upload_large_folder(
        folder_path='results',
        repo_type='model',
        repo_id='auphong2707/BKAI-IGH-NeoPolyp-Unet-model',
        private=False
    )
    
    
if __name__ == '__main__':
    main()