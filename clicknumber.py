import os
from pydub import AudioSegment
import numpy as np
import re

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def count_clicks_in_file(file_path, threshold=1000, min_gap=800):  # Adjust these values
    audio = AudioSegment.from_file(file_path, format="wav")
    samples = np.array(audio.get_array_of_samples())
    last_click_idx = -min_gap
    count = 0

    for i in range(len(samples)):
        if abs(samples[i]) > threshold and i - last_click_idx > min_gap:
            count += 1
            last_click_idx = i

    return count

input_folder = r'E:\University\FYP\FYP_A\dataset\PREPROCESSING_DATA\audio\phase1_005_s1\new_click'
output_folder = r'E:\University\FYP\FYP_A\dataset\PREPROCESSING_DATA\audio\phase1_005_s1\clickdata'
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, 'phase1_005_s1_clicknumber1.txt')

sorted_filenames = sorted(os.listdir(input_folder), key=numerical_sort_key)

with open(output_file_path, 'w') as output_file:
    for filename in sorted_filenames:
        if filename.endswith('.wav'):
            file_path = os.path.join(input_folder, filename)
            num_clicks = count_clicks_in_file(file_path)
            output_file.write(f'{filename}: {num_clicks}\n')
print("Analysis completed and saved to phase1_053_s1_clicknumber1.txt.")
