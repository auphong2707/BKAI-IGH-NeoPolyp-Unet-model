import os
import random
import cv2
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    loss: float,
                    filepath='checkpoint.pth',
                    best=False):
    """Save model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): Optmizer state to save. 
        epoch (int): Current epoch.
        filepath (str): Path to save the checkpoint. Defaults to 'checkpoint.pth'.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filepath)
    if best:
        print(f"Best checkpoint saved at {filepath}")

def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    filepath='checkpoint.pth'):
    """Load model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        filepath (str): Path to load the checkpoint from. Defaults to 'checkpoint.pth'.
        
    Returns:
        int: The epoch to resume training from.
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file not found at {filepath}")
        return 0
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filepath}, trained for {epoch} epochs")
    return epoch

def infer_test_set_color(model, test_dir, output_dir, device, input_size=(256, 256), threshold=0.5):
    """
    Infer segmentation masks for a test set, apply color coding, and save the results to a folder.
    
    Args:
        model (torch.nn.Module): The trained U-Net model.
        test_dir (str): Path to the folder containing test images.
        output_dir (str): Path to the folder where output masks will be saved.
        device (str): Device to run the inference ('cuda' or 'cpu').
        input_size (tuple): The size (width, height) to resize input images.
        threshold (float): Threshold for converting probabilities to binary masks.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Loop through all images in the test directory
    for filename in os.listdir(test_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            image_path = os.path.join(test_dir, filename)
            
            # Load and preprocess the image
            original_image = cv2.imread(image_path)
            original_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
            resized_image = cv2.resize(original_image, input_size)  # Resize to model input size
            input_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_tensor = input_tensor.to(device)
            
            # Infer using the model
            with torch.no_grad():
                output = model(input_tensor)  # Get raw logits or probabilities
                output = torch.softmax(output, dim=1)  # Convert logits to class probabilities
            
            # Get class predictions
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Class labels (0, 1, 2)

            # Resize the class prediction to the original size
            resized_prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Create a color-coded mask
            color_mask = np.zeros((*resized_prediction.shape, 3), dtype=np.uint8)
            color_mask[resized_prediction == 1] = [0, 0, 255]  # Red for neoplastic polyps
            color_mask[resized_prediction == 2] = [0, 255, 0]  # Green for non-neoplastic polyps
            color_mask[resized_prediction == 0] = [0, 0, 0]    # Black for background
            
            # Save the color-coded mask as an image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, color_mask)

            print(f"Processed and saved: {output_path}")
