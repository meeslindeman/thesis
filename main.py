# Import necessary modules
from sklearn.model_selection import train_test_split
from dataloader import DataLoader
from game import get_game
from train import perform_training
from options import Options
from graph.dataset import FamilyGraphDataset
from analysis.plot import plot_acc


# Initialize options
opts = Options()

# Load dataset
dataset = FamilyGraphDataset(root='/Users/meeslindeman/Library/Mobile Documents/com~apple~CloudDocs/Thesis/Code/data', number_of_graphs=100, generations=1)

train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(game_size=1, dataset=train_data, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(game_size=1, dataset=val_data, batch_size=opts.batch_size, shuffle=True)

# Get the game setup based on the options
game = get_game(opts)

# Perform training
results, trainer = perform_training(opts, train_loader, val_loader, game)

plot_acc(results, dataset[0].num_nodes)