from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio = AudioSegment.from_file(
    "E:\\University\\FYP\\FYP_A\\dataset\\PREPROCESSING_DATA\\audio\\phase1_005_s1\\session_1.mp3")

# Calculate the number of samples per second (sample rate)
sample_rate = audio.frame_rate

# Create a time axis for the samples, this generates a time for each sample
samples = np.array(audio.get_array_of_samples())
duration_seconds = len(samples) / sample_rate
time_axis = np.linspace(0, duration_seconds, num=len(samples))

# Plotting the waveform with the time axis
plt.figure(figsize=(10, 4))
plt.plot(time_axis, samples)
plt.title("Audio Waveform of session_1.mp3")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()
