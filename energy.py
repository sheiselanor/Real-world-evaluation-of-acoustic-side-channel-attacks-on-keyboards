import librosa
import numpy as np
import matplotlib.pyplot as plt


def calculate_energy(audio, frame_length=1024, hop_length=512):
    """ Calculate the energy of an audio signal. """
    energy = np.array([
        np.sum(np.abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio), hop_length)
    ])
    return energy


def plot_energy_distribution(energy):
    """ Plot the distribution of energy. """
    plt.figure(figsize=(10, 4))
    plt.hist(energy, bins=50)
    plt.title('Energy Distribution in Audio Signal')
    plt.xlabel('Energy')
    plt.ylabel('Frequency')
    plt.show()


# Load audio
file_path = (
    'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\'
    'totaldataset\\click_10.wav'
)
audio, sr = librosa.load(file_path, sr=None)

# Replace with your file path
audio, sr = librosa.load(file_path, sr=None)

# Calculate and plot energy
energy = calculate_energy(audio)
plot_energy_distribution(energy)

# Inspect the plot to determine an appropriate threshold
