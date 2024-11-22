import math
import os
import random
import time

from matplotlib import pyplot as plt, ticker
import pandas as pd
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