from pydub.utils import mediainfo

# File path
file_path = r"C:\Users\HP\AppData\Local\Programs\Python\Python312\d1\audio\phase1_008_s1\session_1.mp3"

# Get media info
info = mediainfo(file_path)

# Extract sampling rate and frame rate
sampling_rate = info.get('sample_rate')

print(f"Sampling Rate: {sampling_rate} Hz")
