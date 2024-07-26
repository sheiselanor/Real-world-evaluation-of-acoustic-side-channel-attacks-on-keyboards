import os
import shutil

# Replace these paths with your specific paths
source_folder = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\phase1_001_s1\\4'
destination_folder = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\phase1_001_s1\\session_4_clicks'
file_list_txt = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\phase1_001_s1\\clickdata\\4.txt'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read the list of file names and corresponding numbers from the text file
with open(file_list_txt, 'r') as file:
    lines = file.readlines()

# Process each line to get the file name
for line in lines:
    parts = line.strip().split(':')
    if len(parts) == 2:  # Make sure the line is correctly formatted
        file_name = parts[0].strip()
        # Full paths for source and destination files
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)

        # Copy the file to the destination folder if it exists in the source folder
        if os.path.exists(source_file_path):
            shutil.copy2(source_file_path, destination_file_path)

print("Files mentioned in the text file have been copied to the destination folder.")