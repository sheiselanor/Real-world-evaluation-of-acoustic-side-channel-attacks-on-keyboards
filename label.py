import os
import numpy as np
import librosa

# Constants
DATA_DIR = "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\testsubset"  # Update with your actual data directory
OUTPUT_LABEL_DIR = "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\testsubset_label"
SR = 44100  # Sample rate

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SR)
    energy = np.array([sum(abs(audio[i:i+1024]**2)) for i in range(0, len(audio), 1024)])
    return energy

def detect_key_clicks(energy, threshold=0.3):
    # Simple threshold-based detection logic
    clicks = np.where(energy > threshold, 1, 0)
    return clicks

def process_files(data_dir, output_dir):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    for file_path in file_paths:
        print(f"Processing {file_path}")
        energy = extract_features(file_path)
        clicks = detect_key_clicks(energy)
        label_file = os.path.basename(file_path).replace('.wav', '.txt')
        label_path = os.path.join(output_dir, label_file)
        np.savetxt(label_path, clicks, fmt='%d')

# Ensure the output directory exists
if not os.path.exists(OUTPUT_LABEL_DIR):
    os.makedirs(OUTPUT_LABEL_DIR)

# Process all files
process_files(DATA_DIR, OUTPUT_LABEL_DIR)

print("Labeling completed. Check the output directory for labels.")
