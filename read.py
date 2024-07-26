import numpy as np
import os

# Specify the directory where the .npy files are saved
base_path = "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata"

# Load the features and labels from the .npy files
features_path = os.path.join(base_path, 'features_training.npy')
#labels_path = os.path.join(base_path, 'labels.npy')

features = np.load(features_path)
#labels = np.load(labels_path)

# Print some basic information about the arrays
print("Features Array Shape:", features.shape)
#print("Labels Array Shape:", labels.shape)

# Print the first few rows of the features to see what they look like
print("First 1 feature entries:")
print(features[:1])

# Print the first few labels to see their distribution
#print("1st 20 label entries:")
#print(labels[:20])
