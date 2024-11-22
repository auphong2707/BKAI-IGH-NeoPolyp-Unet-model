import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb

from utils.helper import save_checkpoint, load_checkpoint, save_loss, save_plot, time_since
from utils.logger import setup_logger
from tqdm import tqdm

class UNetTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 name: str,
                 learning_rate: float,
                 wandb_key: str,
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
            
        wandb.login(
            # set the wandb project where this run will be logged
            # project= "PolypSegment", 
            key = wandb_key,
        )
        wandb.init(
            project = "PolypSegment"
        )

    def train_epoch(self, dataloader):
        total_loss = 0
        self.model.train()  # Set model to training mode

        for images, masks in tqdm(dataloader, desc="Training Epoch", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            
            masks = masks.squeeze(dim=1).long()

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Compute loss
            loss = self.criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

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
                masks = masks.squeeze(dim=1).long()

                # Forward pass
                outputs = self.model(images)

                # Compute loss
                loss = self.criterion(outputs, masks)

                # Accumulate loss
                total_loss += loss.item()

        return total_loss / len(val_dataloader)

    def train(self, train_dataloader, val_dataloader, n_epochs, print_every=1, plot_every=1):
        # Load checkpoint if available
        load_checkpoint_path = self.checkpoint_directory + '/checkpoint.pth'
        start_epoch = load_checkpoint(self.model, self.optimizer, load_checkpoint_path) + 1
        
        print(f"\nStart training from epoch {start_epoch}")
        print('-' * 80)
        
        # Initialize variables
        start = time.time()

        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(start_epoch, n_epochs + 1):
            # Training step
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation step
            val_loss = self.validate(val_dataloader)
                
            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, train_loss, self.checkpoint_directory + '/checkpoint.pth')

            # Save the best checkpoint
            if epoch == 1 or val_loss < self.best_loss:
                self.best_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, train_loss, self.checkpoint_directory + '/best.pth', best=True)
                
            # Print loss every 'print_every' epochs
            if epoch % print_every == 0:
                self.logger.info('%s (%d %d%%) Train loss: %.4f | Val loss: %.4f' % 
                                 (time_since(start, epoch / n_epochs), 
                                  epoch, epoch / n_epochs * 100, 
                                  train_loss, val_loss))
                print('-' * 80)
            
            # Collect losses for plotting
            if epoch % plot_every == 0:
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
            save_loss(epoch, train_loss, val_loss, self.checkpoint_directory + '/losses.csv')
            wandb.log({'Val_loss': val_loss,'Train_loss': train_loss})

        # Save the plot
        save_plot(self.checkpoint_directory + '/losses.csv', self.checkpoint_directory + '/losses.png')