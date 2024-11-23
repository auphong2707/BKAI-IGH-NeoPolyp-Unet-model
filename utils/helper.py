import math
import os
import random
import time

from matplotlib import pyplot as plt, ticker
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

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

def as_minutes(s: int):
    """Converts seconds to minutes and seconds.
    
    Args:
        s (int): The number of seconds.
        
    Returns:
        str: The formatted string in minutes and seconds.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    """Calculate the time since a given time and percentage of completion.
    
    Args:
        since (float): The time since a given event.
        percent (float): The percentage of completion.
    
    Returns:
        str: The formatted string of time since and time remaining.
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def save_loss(epoch, train_loss, val_loss, filename='losses.csv'):
    # Check if the CSV file already exists
    file_exists = os.path.isfile(filename)
    
    # Create a DataFrame to hold the loss values
    loss_data = pd.DataFrame({'Epoch': [epoch], 'Train Loss': [train_loss], 'Validation Loss': [val_loss]})
    
    # Append the loss data to the CSV file, creating it if it doesn't exist
    loss_data.to_csv(filename, mode='a', index=False, header=not file_exists)

def save_plot(csv_directory, filename='loss_plot.png'):
    """Read training and validation losses from a CSV file, plot them, and save the plot.
    
    Args:
        csv_directory (str): Path to the CSV file containing losses.
        filename (str): The name of the file to save the plot as.
    """
    # Read the CSV file
    df = pd.read_csv(csv_directory)
    
    # Extract train and val losses
    train_losses = df['Train Loss'].tolist()  # Change 'train_loss' to your actual column name
    val_losses = df['Validation Loss'].tolist()      # Change 'val_loss' to your actual column name
    
    plt.figure()
    fig, ax = plt.subplots()
    
    # Set tick locator for the y-axis at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    
    # Plot training and validation losses
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    
    # Add labels and legend
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    
    # Save the plot to a file
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory

def mask_to_rgb(mask, 
                color_dict={0: (0, 0, 0),
                            1: (255, 0, 0),
                            2: (0, 255, 0)}
                ):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)

def infer_and_save(model, test_dir, output_dir, device):
    """
    Perform inference on a model and save the output masks as images.
    
    Args:
        model: The trained PyTorch model.
        test_dir: Path to the directory containing test images.
        output_dir: Path to the directory to save output masks.
        device: The device to perform inference on (e.g., 'cpu' or 'cuda').
        val_transformation: Transformation to apply to the input image.
    """
    val_transformation = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        ori_img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = ori_img.shape[:2]

        # Preprocess the image
        img = cv2.resize(ori_img, (256, 256))
        transformed = val_transformation(image=img)
        input_img = transformed["image"].unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # Resize and process the output mask
        mask = cv2.resize(output_mask, (ori_w, ori_h))
        mask = np.argmax(mask, axis=2)
        mask_rgb = mask_to_rgb(mask)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

        # Save the resulting mask
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, mask_rgb)

def infer_and_save_single(model, img_path, output_path, device):
    """
    Perform inference on a single image and save the output mask as an image.
    
    Args:
        model: The trained PyTorch model.
        img_path: Path to the input image.
        output_dir: Path to the directory to save the output mask.
        device: The device to perform inference on (e.g., 'cpu' or 'cuda').
    """
    val_transformation = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    model.eval()

    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]

    # Preprocess the image
    img = cv2.resize(ori_img, (256, 256))
    transformed = val_transformation(image=img)
    input_img = transformed["image"].unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
    # Resize and process the output mask
    mask = cv2.resize(output_mask, (ori_w, ori_h))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    
    # Save the resulting mask
    cv2.imwrite(output_path, mask_rgb)
    print(f"Your output mask has been saved at {output_path}")