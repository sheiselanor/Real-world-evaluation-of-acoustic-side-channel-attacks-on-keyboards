import os
import shutil

# Set the directory containing the click sound files
clicks_directory = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Testing'
temp_directory = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\temp'

# Create a temporary directory if it doesn't exist
os.makedirs(temp_directory, exist_ok=True)

# Get a list of file names in the directory
click_files = sorted([f for f in os.listdir(clicks_directory) if f.endswith('.wav')])

# Rename the files sequentially and move to temporary directory
for i, filename in enumerate(click_files, start=1):
    old_path = os.path.join(clicks_directory, filename)
    new_filename = f'click_{i}.wav'
    temp_path = os.path.join(temp_directory, new_filename)
    os.rename(old_path, temp_path)

# Move the files back from the temporary directory to the original directory
for filename in os.listdir(temp_directory):
    temp_path = os.path.join(temp_directory, filename)
    final_path = os.path.join(clicks_directory, filename)
    shutil.move(temp_path, final_path)

# Clean up the temporary directory
os.rmdir(temp_directory)

print("All files have been renamed sequentially.")
