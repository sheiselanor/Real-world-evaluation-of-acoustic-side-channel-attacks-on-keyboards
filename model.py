import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Masking, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tabulate import tabulate


# Function to load labels and group them by sequence
def load_labels(label_folder):
    label_groups = []
    label_files = sorted([f for f in os.listdir(label_folder) if
                          f.endswith('.txt')])
    print(f"Number of label files: {len(label_files)}")
    for label_file in label_files:
        with open(os.path.join(label_folder, label_file), 'r') as file:
            labels = [int(line.strip()) for line in file if
                      line.strip().isdigit()]
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
    assert len(features) == len(
        labels), "Mismatch in number of feature and label files"
    labels = pad_labels(labels, maxlen=features.shape[1])
    labels = to_categorical(labels, num_classes=2)
    # Convert labels to one-hot encoding
    return features, labels


# Define the LSTM Model with masking
def build_model(input_shape, num_classes=2):
    model = Sequential([
        Masking(mask_value=0., input_shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),
        # Keep return_sequences=True for sequence output
        TimeDistributed(Dense(128, activation='relu')),
        TimeDistributed(Dense(num_classes, activation='softmax'))
        # Apply Dense layer to each time step
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Paths to the data
train_features_path = (
    'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\'
    'Python312\\d1\\audio\\Splitdata\\features_training.npy'
)
val_features_path = (
    'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\'
    'Python312\\d1\\audio\\Splitdata\\features_validation.npy'
)
train_labels_folder = (
    'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\'
    'Python312\\d1\\audio\\Splitdata\\label\\Training_label\\'
)
val_labels_folder = (
    'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\'
    'Python312\\d1\\audio\\Splitdata\\label\\Validation_label\\'
)

# Load data
X_train, y_train = load_features_and_labels(
    train_features_path, train_labels_folder)
X_val, y_val = load_features_and_labels(val_features_path, val_labels_folder)

# Check if all features have corresponding labels
assert X_train.shape[0] == y_train.shape[
    0], "Training features and labels count mismatch"
assert X_val.shape[0] == y_val.shape[
    0], "Validation features and labels count mismatch"

# Build the model
model = build_model(input_shape=X_train.shape[1:])
model.summary()  # Print the model architecture

# Training the model
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss',
                             save_best_only=True, mode='min')
history = model.fit(X_train, y_train, epochs=100, batch_size=128,
                    validation_data=(X_val, y_val), callbacks=[checkpoint])

# Save the final model
model.save('final_model.keras')

# Optionally: Save the training history for later analysis
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)


# Calculate and print the accuracy and loss percentages
train_accuracy = history.history['accuracy'][-1] * 100
train_loss = history.history['loss'][-1] * 100
val_accuracy = history.history['val_accuracy'][-1] * 100
val_loss = history.history['val_loss'][-1] * 100

# Print the results in a table format
table = [["Training", f"{train_accuracy:.2f}%", f"{train_loss:.2f}%"],
         ["Validation", f"{val_accuracy:.2f}%", f"{val_loss:.2f}%"]]
headers = ["Set", "Accuracy", "Loss"]
print(tabulate(table, headers, tablefmt="pretty"))
