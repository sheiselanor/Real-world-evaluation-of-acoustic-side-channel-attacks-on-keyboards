import os
from pydub import AudioSegment
import numpy as np

# Specify the path to the FFmpeg executable
ffmpeg_path = r'C:\Users\HP\AppData\Local\Programs\Python\Python312\d1\audio\ffmpeg-6.1-full_build\bin\ffmpeg.exe'
AudioSegment.converter = ffmpeg_path

# Input audio file path
audio_path = r'E:\University\FYP\FYP_A\dataset\PREPROCESSING_DATA\audio\phase1_005_s1\session_1.mp3'

# Output folder for click audio files
output_folder = r'E:\University\FYP\FYP_A\dataset\PREPROCESSING_DATA\audio\phase1_005_s1\new_click'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the audio file using pydub
audio = AudioSegment.from_mp3(audio_path)

# Convert AudioSegment to numpy array
samples = np.array(audio.get_array_of_samples())

# Initialize an empty list to store clicks
clicks = []

# Detect clicks based on threshold (adjust as needed)
threshold = 1200  # Adjust this threshold as per your requirement
for i in range(len(samples)):
    if abs(samples[i]) > threshold:
        # Extract the click sound (you may want to refine this)
        start = max(0, i - 800)
        end = min(len(samples), i + 800)
        click = audio[start:end]
        clicks.append(click)

# Save each click as a separate file
for i, click in enumerate(clicks):
    output_audio_file = os.path.join(output_folder, f'click_{i+1}.wav')
    click.export(output_audio_file, format="wav")

print("Key click detection and splitting completed.")