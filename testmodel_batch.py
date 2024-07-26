import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function to load labels and group them by sequence
def load_labels(label_folder):
    label_groups = []
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith(
        '.txt')])
    print(f"Number of label files: {len(label_files)}")
    for label_file in label_files:
        with open(os.path.join(label_folder, label_file), 'r') as file:
            labels = [int(line.strip()) for line in file if line.strip().isdigit()]
            label_groups.append(np.array(labels))
    return label_groups


# Pad labels to ensure they match the sequence length
def pad_labels(labels, maxlen):
    return pad_sequences(labels, maxlen=maxlen, padding='post',
                         truncating='post', value=0)


# Load features and adjust labels to match feature length
def load_features_and_labels(features_path, labels_folder):
    features = np.load(features_path)
    print(f"Number of feature files: {features.shape[0]}")
    labels = load_labels(labels_folder)
    assert len(features) == len(labels), "Mismatch in number of feature and label files"
    labels = pad_labels(labels, maxlen=features.shape[1])
    labels = to_categorical(labels, num_classes=2)
    # Convert labels to one-hot encoding
    return features, labels


# Function to evaluate the model and print results
def evaluate_model(model, X_test, y_test, batch_size):
    test_loss, test_accuracy = model.evaluate(X_test, y_test,
                                              batch_size=batch_size)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss * 100:.2f}%")

    # Predict on test data
    predictions = model.predict(X_test, batch_size=batch_size)
    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(y_test, axis=-1)

    # Calculate additional metrics
    precision = precision_score(true_classes, predicted_classes,
                                average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Print some of the predictions for review
    for i in range(5):  # Adjust this number to see more or fewer predictions
        print(f"Sample {i+1} - True: {true_classes[i]}, Predicted: {predicted_classes[i]}")

    return test_loss, test_accuracy, precision, recall, f1, predicted_classes


# Paths to the test data
test_features_path = ('C:\\Users\\HP\\AppData\\Local\\Programs\\Python'
                      '\\Python312\\d1\\audio\\Splitdata\\features_testing.npy'
                      )
test_labels_folder = ('C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\'
                      'Python312\\d1\\audio\\Splitdata\\label\\Testing_label\\'
                      )

# Load test data
X_test, y_test = load_features_and_labels(test_features_path,
                                          test_labels_folder)


# Load the trained model
model = load_model('final_model.keras')

# Evaluate with different batch sizes
batch_sizes = [16, 32, 64, 128]
results = {}

for batch_size in batch_sizes:
    print(f"\nEvaluating with batch size {batch_size}")
    test_loss, test_accuracy, precision, recall, f1, predicted_classes = evaluate_model(
        model, X_test, y_test, batch_size)
    results[batch_size] = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_classes': predicted_classes
    }

# Optionally: Save the evaluation results
with open('evaluation_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save the predictions to a file for further analysis
for batch_size in batch_sizes:
    np.save(f'predictions_batch_size_{batch_size}.npy', results[batch_size]['predicted_classes'])

# Plot the results
metrics = ['test_loss', 'test_accuracy', 'precision', 'recall', 'f1']
metric_names = ['Test Loss', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score']
colors = ['blue', 'green', 'red', 'purple', 'orange']
bar_width = 0.4

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for i, batch_size in enumerate(batch_sizes):
    row = i // 2
    col = i % 2
    values = [results[batch_size][metric] * 100 for metric in metrics]
    axs[row, col].bar(metric_names, values, color=colors, width=bar_width)
    axs[row, col].set_title(f'Batch Size {batch_size}', fontsize=12)
    axs[row, col].set_xlabel('Metrics', fontsize=10)
    axs[row, col].set_ylabel('Value (%)', fontsize=10)
    axs[row, col].set_ylim(0, 100)
    for j, v in enumerate(values):
        axs[row, col].text(j, v + 1, f'{v:.2f}%', ha='center', color='black',
                           fontsize=12)
    axs[row, col].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()
