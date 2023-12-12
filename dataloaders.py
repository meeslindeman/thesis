from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
#from torch.utils.data import DataLoader

def create_data_loaders(dataset, batch_size, train_split=0.8):

    train_size = int(train_split * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader