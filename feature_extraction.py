import librosa
import numpy as np
import os
import unittest

def extract_features(file_path, n_mfcc=13, max_pad_length=86):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        pad_width = max_pad_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)),
                       mode='constant') if pad_width > 0 else mfccs[:, :max_pad_length]
        return mfccs.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Test cases for the feature extraction
class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features(self):
        # You need to replace 'test_audio.wav' with a path to a real test WAV file
        test_file = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\test_audio.wav'  
        features = extract_features(test_file)
        self.assertIsNotNone(features)  # Ensure features are not None
        self.assertEqual(features.shape, (86, 13))  # Adjust 86 and 13 according to your `max_pad_length` and `n_mfcc`

# Generating file paths
base_path = "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\testsubset"
file_paths = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.wav')]

# Extracting features
features = []
for file_path in file_paths:
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)

# Convert features to NumPy array
features = np.array(features)

# Save features to file
np.save(os.path.join(base_path, 'features_testsubset.npy'), features)

print("Features saved successfully for total dataset.")

# Run tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
