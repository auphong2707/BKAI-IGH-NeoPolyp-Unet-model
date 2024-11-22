import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

from utils.helper import save_checkpoint
from utils.logger import setup_logger
from tqdm import tqdm

class UNetTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 name: str, 
                 learning_rate: float, 
                 device: str, 
                 criterion=nn.CrossEntropyLoss(), 
                 max_norm=1.0):
        """
        Initializes the trainer with model, optimizer, and loss criterion.
        
        Args:
            model (torch.nn.Module): The U-Net model.
            name (str): Name of the experiment (for saving checkpoints/logs).
            learning_rate (float): Learning rate for the optimizer.
            device (str): Device to train on ("cuda" or "cpu").
            criterion (callable): Loss function.
            max_norm (float): Maximum norm for gradient clipping.
        """
        self.name = name
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.max_norm = max_norm

        # Checkpoint directory
        self.checkpoint_directory = f'results/{self.name}'
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        self.best_loss = float('inf')
        
        # Logging
        self.logger = setup_logger(f"{self.checkpoint_directory}/training.log")
        
        # Load best checkpoint if available
        best_checkpoint_path = os.path.join(self.checkpoint_directory, "best.pth")
        if os.path.exists(best_checkpoint_path):
            checkpoint = torch.load(best_checkpoint_path)
            self.best_loss = checkpoint["loss"]

    def train_epoch(self, dataloader):
        total_loss = 0
        self.model.train()  # Set model to training mode

        for images, masks in tqdm(dataloader, desc="Training Epoch", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, val_dataloader):
        total_loss = 0
        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(self.device), masks.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                loss = self.criterion(outputs, masks)

                # Accumulate loss
                total_loss += loss.item()

        return total_loss / len(val_dataloader)

    def train(self, train_dataloader, val_dataloader, n_epochs, print_every=1):
        start_time = time.time()
        train_losses, val_losses = [], []

        # Training loop
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.validate(val_dataloader)

            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, train_loss, self.checkpoint_directory + '/checkpoint.pth')

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, train_loss, self.checkpoint_directory + '/best.pth', best=True)
                
            # Log progress
            if epoch % print_every == 0:
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                self.logger.info(
                    f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time Elapsed: {elapsed}"
                )
                print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time Elapsed: {elapsed}")

            # Track losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
