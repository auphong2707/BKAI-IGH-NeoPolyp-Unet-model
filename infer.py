import argparse

import torch
from config import DEVICE, LEARNING_RATE
from utils.helper import load_checkpoint, infer_and_save_single
import segmentation_models_pytorch as smp
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script for NeoPolyp Unet model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    
    image_path = args.image_path
    
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,
        classes=3     
    ).to(DEVICE)
    optimzer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    load_checkpoint(model, optimzer, 'model.pth')
    
    infer_and_save_single(model, image_path, image_path.split('/')[-1] + '_pred.jpeg', device=DEVICE)
