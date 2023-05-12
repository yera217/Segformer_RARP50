import os
import numpy as np
from PIL import Image

# initialize variables to store the mean and standard deviation of each channel
r_mean, g_mean, b_mean = 0, 0, 0
r_std, g_std, b_std = 0, 0, 0

# initialize a counter to keep track of the number of pixels processed
count = 0

# loop through all images in the current directory
for filename in os.listdir('.'):
    if filename.endswith('.png'):
        print(filename)
        # open the image and convert it to RGB
        img = Image.open(filename).convert('RGB')
        
        # convert the image to a numpy array
        img_arr = np.array(img)
        
        # calculate the mean and standard deviation of each channel
        r_mean += np.mean(img_arr[:, :, 0])
        g_mean += np.mean(img_arr[:, :, 1])
        b_mean += np.mean(img_arr[:, :, 2])
        r_std += np.std(img_arr[:, :, 0])
        g_std += np.std(img_arr[:, :, 1])
        b_std += np.std(img_arr[:, :, 2])

        # update the pixel count
        count += 1

# divide the sum of means and standard deviations by the total number of pixels to get the average values

r_mean /= count
g_mean /= count
b_mean /= count
r_std /= count
g_std /= count
b_std /= count

# print the results
print(f"Red channel: mean = {r_mean}, std = {r_std}")
print(f"Green channel: mean = {g_mean}, std = {g_std}")
print(f"Blue channel: mean = {b_mean}, std = {b_std}")

