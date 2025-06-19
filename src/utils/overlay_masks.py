#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script creates images that includes the masks overlayes with a certain
degree of transparency
"""


##########################
# Imports and libraries
##########################

import cv2
import numpy as np
import os


#########################
# Global variables
#########################

# Define your directories
img_directory = '/home/azken/datasets/cystoseira/all/img'
msk_directory = '/home/azken/datasets/cystoseira/all/msk'
out_directory = '/home/azken/datasets/cystoseira/all/overlaid'
alpha = 0.5


###########################
# Helper functions
###########################

def overlay_mask(image_path, mask_path, output_path, alpha=0.5):
    """
    Overlays a mask on an image with a specified transparency.

    Args:
        image_path (str): Path to the image file.
        mask_path (str): Path to the mask file.
        output_path (str): Path to save the overlaid image.
        alpha (float): Transparency of the mask (0 to 1).
    """
    try:
        # Read the image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if the images were successfully loaded
        if image is None:
            raise ValueError(f"Could not open or find the image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not open or find the mask: {mask_path}")

        # Convert the mask to a color image
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Resize the mask to match the image dimensions
        colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]))

        # Overlay the mask on the image with transparency
        overlaid_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

        # Save the result
        cv2.imwrite(output_path, overlaid_image)

        print(f"Successfully overlaid mask on image. Output saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def process_images(image_dir, mask_dir, output_dir, alpha=0.2):
    """
    Processes all images and masks in the given directories.

    Args:
        image_dir (str): Directory containing the images.
        mask_dir (str): Directory containing the masks.
        output_dir (str): Directory to save the overlaid images.
        alpha (float): Transparency of the mask (0 to 1).
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the image directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # or any other image format
            image_name = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, image_name + ".png")  # assuming masks have the same name
            output_path = os.path.join(output_dir,
                filename[:-4] + f'_ov{int(10*alpha)}_' + filename[-4:])

            # Check if the mask file exists
            if os.path.exists(mask_path):
                overlay_mask(image_path, mask_path, output_path, alpha)
            else:
                print(f"Mask not found for image: {filename}")


#################
# Main
#################

# Process the images
process_images(img_directory, msk_directory, out_directory, alpha)
