import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('0.npz')

# Extract images and labels
images = data['x']
labels = data['y']

# Define the number of images per batch and total batches
images_per_batch = 100
total_batches = len(images) // images_per_batch

for batch in range(total_batches):
    # Create a new figure
    plt.figure(figsize=(10, 10))

    for i in range(images_per_batch):
        # Calculate index of the image in the dataset
        idx = batch * images_per_batch + i
        if idx < len(images):
            plt.subplot(10, 10, i + 1)  # Assuming images_per_batch is 100
            plt.imshow(images[idx])
            plt.title(f'Label: {labels[idx]}')
            plt.axis('off')

    # Display the batch of images
    plt.show()

    # Optional: Pause or wait for user input between batches
    input("Press Enter to continue to the next batch...")
