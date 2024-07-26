import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the training history
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)


# Function to annotate the plot with selected values
def annotate_selected(values, ax, label, selected_epochs):
    for epoch in selected_epochs:
        value = values[epoch - 1]  # epochs are 1-based, indexes are 0-based
        ax.annotate(f'{label}\nEpoch: {epoch}\nValue: {value:.4f}',
                    xy=(epoch - 1, value),
                    xytext=(epoch - 1, value + 0.01),
                    arrowprops=dict(facecolor='black', shrink=0.07),
                    fontsize=12,
                    ha='center')


# Plot the accuracy history
def plot_accuracy(history, train_epochs, val_epochs):
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(history['accuracy'], 'b-', label='Train Accuracy', markersize=5)
    ax1.plot(history['val_accuracy'], 'r-', label='Validation Accuracy', markersize=5)
    annotate_selected(history['accuracy'], ax1, 'Train Accuracy', train_epochs)
    annotate_selected(history['val_accuracy'], ax1, 'Validation Accuracy', val_epochs)
    ax1.set_title('Model Accuracy', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch 100/ Batch 128', fontsize=14)
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True)
    ax1.set_yticks(np.arange(0.850, 0.965, 0.008))
    ax1.set_xticks(np.arange(0, 110, 10))
    plt.show()


# Plot the loss history
def plot_loss(history, train_epochs, val_epochs):
    plt.figure(figsize=(12, 6))
    ax2 = plt.gca()
    ax2.plot(history['loss'], 'b-', label='Train Loss', markersize=5)
    ax2.plot(history['val_loss'], 'r-', label='Validation Loss', markersize=5)
    annotate_selected(history['loss'], ax2, 'Train Loss', train_epochs)
    annotate_selected(history['val_loss'], ax2, 'Validation Loss', val_epochs)
    ax2.set_title('Model Loss', fontsize=16)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_xlabel('Epoch 100/ Batch 128', fontsize=14)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True)
    ax2.set_yticks(np.arange(0.10, 0.40, 0.02))
    ax2.set_xticks(np.arange(0, 110, 10))
    plt.show()


# Select specific epochs to annotate
train_epochs = [40]
val_epochs = [80]

plot_accuracy(history, train_epochs, val_epochs)
plot_loss(history, train_epochs, val_epochs)
