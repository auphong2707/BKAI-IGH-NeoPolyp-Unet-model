import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class UNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        """
        Custom dataset for U-Net models.

        Args:
            image_dir (str): Path to the directory containing input images.
            mask_dir (str): Path to the directory containing masks.
            transform (callable, optional): Transformation for input images.
            target_transform (callable, optional): Transformation for masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


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
    # Define transformations
    base_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),  # Masks become [0, 1] tensors
    ])

    # Augmentations (if enabled)
    if augmentations:
        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
        ])
        base_transform = transforms.Compose([augmentation_transform, *base_transform.transforms])

    # Full dataset
    dataset = UNetDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=base_transform,
        target_transform=mask_transform,
    )

    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader}


if __name__ == "__main__":
    image_dir = "./data/train/train"
    mask_dir = "./data/train_gt/train_gt"
    dataloaders = get_dataloaders(image_dir, mask_dir, batch_size=16, img_size=(256, 256))

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    for images, masks in train_loader:
        import matplotlib.pyplot as plt

        image = images[0].permute(1, 2, 0).numpy()
        mask = masks[0].squeeze().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title('Image')
        ax[0].axis('off')

        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('Mask')
        ax[1].axis('off')

        plt.show()
        break

