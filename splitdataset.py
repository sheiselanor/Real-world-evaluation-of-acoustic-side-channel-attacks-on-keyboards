import os
import shutil
import numpy as np

def split_data(source_folder, train_size=0.7, val_size=0.15, test_size=0.15):
    # Ensure the sizes approximately sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "The proportions must sum to 1"

    # Create directories for the train, validation and test datasets if they don't exist
    if not os.path.exists(os.path.join(source_folder, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Training')):
        os.makedirs(os.path.join(source_folder, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Training'))
    if not os.path.exists(os.path.join(source_folder, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Validation')):
        os.makedirs(os.path.join(source_folder, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Validation'))
    if not os.path.exists(os.path.join(source_folder, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Testing')):
        os.makedirs(os.path.join(source_folder, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Testing'))

    # Get a list of all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    np.random.shuffle(all_files)  # Shuffle the list to ensure random distribution

    # Split files according to the given ratios
    total_files = len(all_files)
    train_end = int(total_files * train_size)
    val_end = train_end + int(total_files * val_size)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # Function to copy files to a target directory
    def copy_files(files, target_dir):
        for file in files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(source_folder, target_dir, file))

    # Move files to respective folders
    copy_files(train_files, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Training')
    copy_files(val_files, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Validation')
    copy_files(test_files, 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\Testing')

# Use the function
source_folder = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\totaldataset'  # You need to replace this with the path to your dataset
split_data(source_folder)
