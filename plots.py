import matplotlib.pyplot as plt
import os
from PIL import Image

# Specify the folder where the images are saved
image_folder = 'E:\\University\\FYP\\FYP B\\result\\train_validate\\acc-epoch 500'

# List all image files in the specified folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'png'))]

# Sort or select the first 4 images (adjust as needed)
selected_images = image_files[:4]

# Create a 2x2 grid of subplots with increased figure size
fig, axs = plt.subplots(2, 2, figsize=(18, 18))

# Flatten the axes array for easy iteration
axs = axs.flatten()

# Loop over the images and the axes to display each image
for ax, image_file in zip(axs, selected_images):
    img_path = os.path.join(image_folder, image_file)
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title("")  # Clear the title at the top
    ax.axis('off')  # Hide axes
    ax.set_xlabel(image_file, fontsize=14)  # Set label below each subplot

# Adjust layout to remove empty spaces
plt.subplots_adjust(wspace=0.0001, hspace=0.0001)
plt.suptitle("Comparison of Model Accuracies", fontsize=16)
plt.show()
