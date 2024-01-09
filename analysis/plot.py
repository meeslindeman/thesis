import matplotlib.pyplot as plt
import json

def plot_acc(results, n):
    # Extracting the data from each line
    results_lines = results.strip().split('\n')
    results = [json.loads(line) for line in results_lines]

    # Separating train and test accuracies
    train_acc = [r['acc'] for r in results if r['mode'] == 'train']
    test_acc = [r['acc'] for r in results if r['mode'] == 'test']
    epochs = range(1, len(train_acc) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.gca().set_facecolor('#E8F0F2')  # Light blue-grey color for plot area

    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o', linestyle='-.', linewidth=3, alpha=0.7)
    plt.plot(epochs, test_acc, label='Test Accuracy', marker='o', linestyle=':', linewidth=3, alpha=0.7)
    plt.axhline(y=(1/n), color='grey', linestyle='--', label=f'Chance Level (1/{n})')

    plt.title(f'Training and Test Accuracy per Epoch (n={n})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='-', linewidth=0.5, alpha=0.5, color='grey')
    plt.show()