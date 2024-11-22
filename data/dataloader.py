import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class FirstServeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, resize=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resize = resize
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)
    
    def read_mask(self, mask_path):
        image = cv2.imread(mask_path)
        image = cv2.resize(image, self.resize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 20])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160,100,20])
        upper_red2 = np.array([179,255,255])
        
        lower_mask_red = cv2.inRange(image, lower_red1, upper_red1)
        upper_mask_red = cv2.inRange(image, lower_red2, upper_red2)
        
        red_mask = lower_mask_red + upper_mask_red
        red_mask[red_mask != 0] = 1

        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255, 255))
        green_mask[green_mask != 0] = 2

        full_mask = cv2.bitwise_or(red_mask, green_mask)
        full_mask = np.expand_dims(full_mask, axis=-1) 
        full_mask = full_mask.astype(np.uint8)
        
        return full_mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = cv2.imread(img_path)  #  BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
        image = cv2.resize(image, self.resize)
        
        label = self.read_mask(mask_path)  
        if self.transform:
            image = self.transform(image)
            
        return image, label

class UnetDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        # Get image and label
        image = self.data[index]
        label = self.targets[index]
        
        # Ensure image and label have the same dimensions
        assert image.shape[:2] == label.shape[:2]
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image'].float()
            label = transformed['mask'].float()
            label = label.permute(2, 0, 1)
        
        # Normalize image
        return image, label
    
    def __len__(self):
        return len(self.data)

def get_dataloaders(image_dir, mask_dir, batch_size=16, img_size=(256, 256), augmentations=True, val_split=0.2):
    """
    Creates DataLoaders for training and validation for U-Net models.

    Args:
        image_dir (str): Path to the directory containing input images.
        mask_dir (str): Path to the directory containing masks.
        batch_size (int): Number of samples per batch.
        img_size (tuple): Size of images for resizing (width, height).
        augmentations (bool): Whether to include data augmentations.
        val_split (float): Fraction of data to use for validation.

    Returns:
        dict: A dictionary containing 'train' and 'val' DataLoaders.
    """
    dataset = FirstServeDataset(image_dir=image_dir,
                                mask_dir=mask_dir,
                                resize=img_size,
                                transform = None)

    images_data = []
    labels_data = []
    for x,y in dataset:
        images_data.append(x)
        labels_data.append(y)
        
    train_transformation = A.Compose([
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
        A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transformation = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    train_dataset = UnetDataset(data=images_data, targets=labels_data, 
                                       transform=train_transformation)

    val_dataset = UnetDataset(data=images_data, targets=labels_data, 
                                     transform=val_transformation)
    
    train_size = int((1 - val_split) * len(images_data))
    
    train_dataset = UnetDataset(images_data[:train_size], labels_data[:train_size], transform=train_transformation)
    val_dataset = UnetDataset(images_data[train_size:], labels_data[train_size:], transform=val_transformation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return {'train': train_loader, 'val': val_loader}