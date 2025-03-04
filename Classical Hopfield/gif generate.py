import os
import imageio.v2 as imageio
from natsort import natsorted  # Import natsorted for natural sorting

# Specify the directory containing the images and the output video filename
image_directory = r'C:\\Users\\scfro\\OneDrive\\Documents\\tempFiles3'
output_video = 'hopfield_50_steps.mp4'

# Create a list to hold the image filenames
images = []

# Loop through the directory and collect image filenames
for filename in os.listdir(image_directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
        images.append(os.path.join(image_directory, filename))

# Use natsorted to sort the filenames naturally
images = natsorted(images)

# Create a video writer object
writer = imageio.get_writer(output_video, fps=200)  # Adjust fps as needed

# Loop through the images and add them to the video
for image_file in images:
    img = imageio.imread(image_file)
    writer.append_data(img)

# Close the writer
writer.close()

print(f'MP4 video saved as {output_video}')
