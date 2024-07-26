from pydub import AudioSegment
import numpy as np

# Load your MP3 file
audio = AudioSegment.from_file("C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\phase1_025_s1\\session_1.mp3")
samples = np.array(audio.get_array_of_samples())

# Analyze the amplitude values to determine a threshold for click detection
peak_amplitude = np.max(np.abs(samples))
average_amplitude = np.mean(np.abs(samples))

# Set a threshold for click detection
threshold = peak_amplitude * 0.5  # For example, 50% of the peak amplitude

print(f"Peak amplitude: {peak_amplitude}")
print(f"Average amplitude: {average_amplitude}")
print(f"Suggested threshold: {threshold}")
