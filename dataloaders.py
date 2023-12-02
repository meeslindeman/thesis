from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from graph.build_dataset import CustomGraphDataset

def create_data_loaders(root_path, batch_size, train_split=0.8):
    dataset = CustomGraphDataset(root=root_path)

    # Adjust split sizes based on your dataset
    train_size = int(train_split * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total dataset size: {len(dataset)}")
    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {valid_size}")
    print("======")
    print(train_loader)
    # Optional: Print details from the first batch
    first_batch = next(iter(train_loader))
    print("First batch - Graphs:", first_batch[0])  # Modify depending on your dataset structure
    print("First batch - Labels:", first_batch[1])

    return train_loader, valid_loader

create_data_loaders('/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/families', batch_size=3)