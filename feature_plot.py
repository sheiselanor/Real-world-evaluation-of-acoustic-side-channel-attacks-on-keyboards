import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize, LinearSegmentedColormap

# Define the starting index for sample numbering
starting_index = 117  # You can change this value each time you run the script

# Create a custom colormap
colors = ["purple", "yellow", "black"]  # Black for low, purple for mid, yellow for high
nodes = [0.0, 0.5, 1.0]
cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))


# Load the features
features_path = 'C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\d1\\audio\\Splitdata\\features_training.npy'  # Adjust this to the path of your .npy file
features = np.load(features_path)
sample_features = features[116]  # Example: Just plotting the first sample

# Optional: Smooth the features to reduce noise and improve visual clarity
#smoothed_features = gaussian_filter(sample_features, sigma=1)

# Set up the figure with a higher resolution
plt.figure(figsize=(9, 10), dpi=130)

# Iterate over the first four samples
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)  # Creates 2x2 grid of subplots

     # Optional: Smooth the features to reduce noise and improve visual clarity
    smoothed_features = gaussian_filter(features[i + starting_index - 1], sigma=1)

    # Adjusting the normalization to emphasize spikes
    vmin, vmax = np.percentile(smoothed_features, [28, 92])  # Tightening the percentile for more contrast

    # Plot the smoothed features with adjusted color scale and interpolation
    im = ax.imshow(smoothed_features.T, aspect='auto', origin='lower', cmap=cmap,
                   interpolation='bilinear', norm=Normalize(vmin=vmin, vmax=vmax))
    
    # Adding titles and labels
    ax.set_title(f'Feature Extraction Visualization - Sample {i + starting_index}')
    ax.set_ylabel('MFCC Coefficients')
    ax.set_xlabel('Time')

# Adding a single colorbar for the entire figure
fig = plt.gcf()
cbar_ax = fig.add_axes([0.954, 0.15, 0.01, 0.7])  # Fine-tune these dimensions as necessary
fig.colorbar(im, cax=cbar_ax, label='Scale')

plt.tight_layout(pad=4.0)  # Adjust padding between plots
plt.show()