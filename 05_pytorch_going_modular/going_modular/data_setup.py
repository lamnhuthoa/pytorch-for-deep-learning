"""
Contains functionality by creating PyTorch DataLoader's for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = NUM_WORKERS):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns them ino PyTorch DataLoaders.

    Args:
        train_dir (str): Path to training data directory.
        test_dir (str): Path to testing data directory.
        transform (transforms.Compose): torchvision transforms to perform on the data.
        batch_size (int): Number of samples per batch in each DataLoader.
        num_workers (int, optional): An integer for number of workers per DataLoader.

    Returns:
        A Tuple of (train_dataloader, test_dataloader).
        Where class_names is a list of the target clases

        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(
                train_dir="data/pizza_steak_sushi/train",
                test_dir="data/pizza_steak_sushi/test",
                transform=transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                ]),
                batch_size=32,
                num_workers=2,
            )
    """

    # Use ImageFolder to create datasets(s)
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True, 
    )

    return train_dataloader, test_dataloader, class_names

