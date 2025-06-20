#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run this script to check the image shapes of the dataset. You run it over the
masks in .png format, i.e. over the adapted dataset
example:
DATASET_PATH = '/home/azken/datasets/cystoseira/all/msk'
"""


##########################
# Imports and libraries
##########################

import os
from PIL import Image
from collections import defaultdict


####################################
# Global variables for control
####################################

DATASET_PATH = '/home/azken/datasets/cystoseira/all/msk'


###########################
# Helper functions
###########################

def analyze_image_shapes(root_path):
    """Quantifies unique image dimensions in hierarchical dataset structure.
    
    Args:
        root_path (str): Path to main dataset directory
        
    Returns:
        dict: Mapping of (width, height) tuples to occurrence counts
    """
    dimension_counts = defaultdict(int)
    
    for root, _, files in os.walk(root_path):
        for filename in files:
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
                try:
                    img_path = os.path.join(root, filename)
                    with Image.open(img_path) as img:
                        dimensions = img.size  # (width, height)
                        dimension_counts[dimensions] += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    
    return dimension_counts


#############################
# Main
#############################

shape_distribution = analyze_image_shapes(DATASET_PATH)

print("Dimensional Analysis Results:")
for dims, count in shape_distribution.items():
    print(f"Dimensions {dims}: {count} images")
