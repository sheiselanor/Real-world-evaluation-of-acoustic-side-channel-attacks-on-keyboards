import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# Function to record audio for 15 seconds
def record_audio(duration=15, sample_rate=44100):
    print("Recording audio for 15 seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished
    return audio, sample_rate


# Function to save the recorded audio to a file
def save_audio(audio, sample_rate, filename="recorded_audio.wav"):
    wav.write(filename, sample_rate, (audio * 32767).astype(np.int16))
    # Convert to int16 for wav file
    print(f"Audio saved as {filename}")


# Function to preprocess the audio (e.g., noise reduction)
def preprocess_audio(filename):
    print("Preprocessing audio...")
    audio, sample_rate = librosa.load(filename, sr=None)
    audio = librosa.effects.trim(audio)[0]  # Remove silence
    return audio, sample_rate


# Function to add noise to the audio
def add_noise(audio, noise_factor=0.001):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    augmented_audio = augmented_audio / np.max(np.abs(augmented_audio))
    return augmented_audio


# Function to extract features from the audio
def extract_features(audio, sample_rate, n_mfcc=13, max_pad_length=86):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    pad_width = max_pad_length - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)),
                       mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_length]

    # Plot the MFCCs with more precise settings
    plt.figure(figsize=(12, 6))
    plt.imshow(mfccs, interpolation='nearest', cmap='coolwarm', origin='lower', aspect='auto')
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    return mfccs.T


# Function to display audio characteristics
def display_audio_characteristics(audio, sample_rate, title):
    print(f"Audio Characteristics for {title}:")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Audio Duration: {len(audio) / sample_rate} seconds")
    print(f"Audio Max Amplitude: {np.max(audio)}")
    print(f"Audio Min Amplitude: {np.min(audio)}")
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title(f"Audio Waveform - {title}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


# Load the trained model
model = load_model('final_model.keras')


# Main function to record, preprocess, and predict audio
def main():
    # Record audio
    audio, sample_rate = record_audio()

    # Save the recorded audio
    save_audio(audio, sample_rate)

    # Display recorded audio characteristics
    display_audio_characteristics(audio, sample_rate, "Recorded Audio")

    # Preprocess the audio
    audio, sample_rate = preprocess_audio("recorded_audio.wav")

    # Display processed audio characteristics
    display_audio_characteristics(audio, sample_rate, "Processed Audio")

    # Augment the audio with noise
    audio_with_noise = add_noise(audio)

    # Display noise-augmented audio characteristics
    display_audio_characteristics(audio_with_noise, sample_rate,
                                  "Noise-Augmented Audio")

    # Extract features
    features = extract_features(audio_with_noise, sample_rate)
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    # Ensure features have the correct shape
    print(f"Feature shape before prediction: {features.shape}")

    # Predict using the model
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=-1)

    # Display the predictions
    print("Predictions:")
    print(predicted_classes[0])

    # Calculate accuracy (example)
    true_labels = np.array([0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
    # Replace with actual true labels
    true_labels = np.pad(true_labels, (0, features.shape[1] - len(true_labels)), 'constant', constant_values=0)
    accuracy = accuracy_score(true_labels, predicted_classes[0])
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
