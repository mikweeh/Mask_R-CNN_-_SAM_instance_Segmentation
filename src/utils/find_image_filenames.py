#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"This script is used to create the files train.txt, test.txt and valid.txt"

import argparse
import os
from typing import List

# Global configuration constants
TARGET_FOLDER = './dataset/original_yolo/test/images'
SORT_ALPHABETICALLY = False  # Set to True to enable alphabetical sorting by default


def find_image_filenames(folder_path: str, sort_alphabetically: bool = False) -> List[str]:
    """Finds all image filenames in a given folder (non-recursively).

    Args:
        folder_path (str): The path to the folder to search.
        sort_alphabetically (bool): If True, returns filenames sorted alphabetically.
                                   If False, returns filenames in arbitrary order.
                                   Defaults to False.

    Returns:
        List[str]: A list of filenames corresponding to the images found.
                   Returns an empty list if the folder does not exist or
                   contains no images.
    """
    # Define a set of common image file extensions for efficient lookup
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'
    }

    image_filenames = []
    
    # Ensure the provided path is a valid directory
    if not os.path.isdir(folder_path):
        print(f"Error: Directory not found at '{folder_path}'")
        return image_filenames

    # Iterate over all entries in the given directory
    for filename in os.listdir(folder_path):
        # Build the full path to check if it's a file
        full_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(full_path):
            # Get the file extension and convert it to lowercase
            _, ext = os.path.splitext(filename)
            
            # Check if the extension is in our set of image extensions
            if ext.lower() in image_extensions:
                image_filenames.append(filename)
    
    # Sort alphabetically if requested
    if sort_alphabetically:
        image_filenames.sort()
                
    return image_filenames


# Example of how to use the function
if __name__ == '__main__':
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(
        description='Find image files in a directory and optionally sort them alphabetically.'
    )
    parser.add_argument(
        '--sort',
        dest='sort_alphabetically',
        action='store_true',
        help='Sort filenames alphabetically'
    )
    parser.add_argument(
        '--no-sort',
        dest='sort_alphabetically',
        action='store_false',
        help='Do not sort filenames'
    )
    # Use the global constant as the default value
    parser.set_defaults(sort_alphabetically=SORT_ALPHABETICALLY)
    
    args = parser.parse_args()
    
    target_folder = TARGET_FOLDER

    print(f"Searching for image files in: '{os.path.abspath(target_folder)}'")
    
    # Call the function to get the list of images
    images_found = find_image_filenames(target_folder, args.sort_alphabetically)

    if images_found:
        sort_status = "sorted alphabetically" if args.sort_alphabetically else "unsorted"
        print(f"\nFound {len(images_found)} image files ({sort_status}):")
        for image_file in images_found:
            print(f"{os.path.splitext(image_file)[0]},")
    else:
        print("\nNo image files were found in this directory.")
    
    pas
