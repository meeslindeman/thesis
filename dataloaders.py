from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

def create_data_loaders(dataset, batch_size, train_split=0.8):
    """
    Create data loaders for training and validation datasets.

    Args:
        dataset (Dataset): The dataset to be split into training and validation sets.
        batch_size (int): The batch size for the data loaders.
        train_split (float, optional): The ratio of the dataset to be used for training. Defaults to 0.8.

    Returns:
        train_loader (DataLoader): The data loader for the training dataset.
        valid_loader (DataLoader): The data loader for the validation dataset.
    """
    train_size = int(train_split * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader