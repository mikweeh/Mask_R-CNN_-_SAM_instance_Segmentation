#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polygon Simplification Script
=============================

This script reduces the number of points in polygon annotations while maintaining
their shape, useful for optimizing machine learning datasets with YOLO format
annotations.

Usage:
------
From the command line:
    python polygon_simplifier.py -p PATH_TO_DATASET [options]

Required Arguments:
------------------
    --path, -p : Path to dataset containing 'images' and 'labels' folders

Optional Arguments:
------------------
    --tolerance, -t : Simplification tolerance (default: 2.0)
                      Higher values result in more simplification
    
    --show, -s : Show visualization of before and after simplification
    
    --figure-width, -fw : Width of display window (default: 800)
    
    --figure-height, -fh : Height of display window (default: 600)

Examples:
---------
    # Process dataset with default settings (no visualization)
    python polygon_simplifier.py -p /path/to/dataset
    
    # Process with visualization and higher tolerance
    python polygon_simplifier.py -p /path/to/dataset -s -t 4.0
    
    # Process with custom visualization size
    python polygon_simplifier.py -p /path/to/dataset -s -fw 1024 -fh 768

Visualization Controls:
---------------------
    Press ENTER key to proceed to the next image when visualization is enabled.
    
Input Format:
-----------
    Expects YOLO format polygon annotations in 'labels' folder with the format:
    class_id x1 y1 x2 y2 x3 y3 ...
    
    Where coordinates are normalized (0-1) relative to image dimensions.

Output:
------
    Simplified polygons are written back to the original label files.
    The script provides information about the number of points before and after
    simplification.
"""

########################
# Libraries
########################

import os
import argparse
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from shapely import Polygon


#################################
# Helper functions
#################################

def reduce_polygon_points(
    labels_path: str,
    image_path: str, 
    show: bool = False,
    tolerance: float = 2,
    figure_size: Optional[Tuple[int, int]] = None
) -> None:
    """
    Simplify polygon annotations in a label file by reducing the number of points.

    The function reads polygon annotations from a label file, simplifies them
    using the Douglas-Peucker algorithm, and writes the simplified polygons back
    to the file. It can optionally display the original and simplified polygons.

    Args:
        labels_path: Path to the label file containing polygon annotations.
        image_path: Path to the corresponding image file.
        show: Whether to display the original and simplified polygons.
        tolerance: The maximum distance between original and simplified lines.
                  Higher values result in more simplification.
        figure_size: Size of the display window when show=True as (width, height).
                    If None, uses the original image size.

    Returns:
        None
    """
    # Read the image to get dimensions and for visualization
    image = cv2.imread(image_path)
    image_dimensions = image.shape[0:2][::-1]  # Convert to (width, height)
    
    # Prepare display image if showing results
    if show and figure_size:
        display_image = cv2.resize(image, figure_size)
    else:
        display_image = image.copy()
        figure_size = image_dimensions
    
    # Read label file
    with open(labels_path, 'r') as label_file:
        info = label_file.readlines()
    
    new_data = []
    
    # Process each line in the label file
    for line in info:
        line = line.split()
        cls = line.pop(0)  # First value is the class ID
        
        try:
            # Convert string values to float coordinates
            mask_points = [float(value) for value in line]
            # Create (x,y) point pairs from the flat list
            points = [(mask_points[i], mask_points[i + 1]) 
                     for i in range(0, len(mask_points), 2)]
            print("Number of initial points: ", len(points))

            # Scale normalized coordinates (0-1) to image dimensions
            scaled_points = [
                (x * image_dimensions[0], y * image_dimensions[1]) 
                for x, y in points
            ]
            
            # Create Shapely polygon and convert to numpy array for OpenCV
            polygon = Polygon(scaled_points)
            polygon_np = np.array(
                polygon.exterior.coords.xy
            ).T.reshape((-1, 1, 2)).astype(np.int32)

            # Prepare for visualization
            before_polygon = display_image.copy()
            before_points = display_image.copy()
            
            # Scale polygon coordinates to match display size
            display_polygon_np = np.array([
                [[int(p[0][0] * figure_size[0] / image_dimensions[0]), 
                  int(p[0][1] * figure_size[1] / image_dimensions[1])]] 
                for p in polygon_np
            ])
            
            # Draw before polygon and points
            cv2.polylines(
                before_polygon, 
                [display_polygon_np], 
                isClosed=True, 
                color=(255, 0, 0), 
                thickness=2
            )

            for point in display_polygon_np:
                cv2.circle(
                    before_points, 
                    center=(point[0][0], point[0][1]), 
                    radius=1, 
                    color=[0, 0, 255], 
                    thickness=2
                )

            # Simplify the polygon
            polygon = polygon.simplify(
                tolerance=tolerance, 
                preserve_topology=True
            )
            final_points = list(polygon.exterior.coords)[0:-1]  # Remove duplicate

            # Safety check: if simplified too much, keep original
            if len(final_points) < 3:
                print(
                    f"Warning: Polygon simplified to {len(final_points)} points, "
                    f"keeping original polygon"
                )
                # Restore original points
                new_points = [cls] + [str(value) for value in mask_points]
                new_data.append(' '.join(new_points) + '\n')
                continue

            # Convert back to normalized coordinates (0-1)
            new_points = []
            for x, y in final_points:
                new_points.append(str(x / image_dimensions[0]))
                new_points.append(str(y / image_dimensions[1]))
            
            # Add class ID back at the beginning
            new_points.insert(0, cls)

            # Prepare after visualizations
            polygon_np = np.array(
                polygon.exterior.coords.xy
            ).T.reshape((-1, 1, 2)).astype(np.int32)
            
            after_polygon = display_image.copy()
            after_points = display_image.copy()
            
            # Scale polygon coordinates to match display size
            display_polygon_np = np.array([
                [[int(p[0][0] * figure_size[0] / image_dimensions[0]), 
                  int(p[0][1] * figure_size[1] / image_dimensions[1])]] 
                for p in polygon_np
            ])
            
            # Draw after polygon and points
            cv2.polylines(
                after_polygon, 
                [display_polygon_np], 
                isClosed=True, 
                color=(255, 0, 0), 
                thickness=2
            )

            for point in display_polygon_np:
                cv2.circle(
                    after_points, 
                    center=(point[0][0], point[0][1]), 
                    radius=1, 
                    color=[0, 0, 255], 
                    thickness=2
                )

            # Display all four images in a single figure with subplots
            if show:
                # Convert BGR to RGB for matplotlib
                before_polygon_rgb = cv2.cvtColor(before_polygon, cv2.COLOR_BGR2RGB)
                before_points_rgb = cv2.cvtColor(before_points, cv2.COLOR_BGR2RGB)
                after_polygon_rgb = cv2.cvtColor(after_polygon, cv2.COLOR_BGR2RGB)
                after_points_rgb = cv2.cvtColor(after_points, cv2.COLOR_BGR2RGB)
                
                # Create figure with 2x2 subplots
                plt.figure(figsize=(12, 10))
                
                # Before polygon
                plt.subplot(2, 2, 1)
                plt.imshow(before_polygon_rgb)
                plt.title('Before: polygon')
                plt.axis('off')
                
                # Before points
                plt.subplot(2, 2, 2)
                plt.imshow(before_points_rgb)
                plt.title('Before: points')
                plt.axis('off')
                
                # After polygon
                plt.subplot(2, 2, 3)
                plt.imshow(after_polygon_rgb)
                plt.title('After: polygon')
                plt.axis('off')
                
                # After points
                plt.subplot(2, 2, 4)
                plt.imshow(after_points_rgb)
                plt.title('After: points')
                plt.axis('off')
                
                plt.tight_layout()
                
                def press(event):
                    if event.key == 'enter':
                        plt.close()
                        
                fig = plt.gcf()  # Get current figure
                fig.canvas.mpl_connect('key_press_event', press)
                plt.show(block=True)  # Block execution until figure is closed
            
            # Add the simplified polygon data to the output
            new_data.append(' '.join(new_points) + '\n')
            print("Final number of points:", len(final_points))
            
        except Exception as e:
            # If there's an error, keep the original line
            print(f"Error processing polygon: {e}")
            line.insert(0, cls)
            new_data.append(' '.join(line) + '\n')

    # Write the updated data back to the label file
    with open(labels_path, 'w') as label_file:
        for data in new_data:
            label_file.write(data)


def process_dataset(
    dataset_path: str,
    show: bool = False,
    tolerance: float = 2,
    figure_size: Optional[Tuple[int, int]] = None
) -> None:
    """
    Process all images and label files in a dataset.

    Args:
        dataset_path: Path to the dataset containing 'images' and 'labels' folders
        show: Whether to display the original and simplified polygons
        tolerance: Simplification tolerance parameter
        figure_size: Size of the display window when show=True as (width, height)
                     If None, uses the original image size

    Returns:
        None
    """
    # Get sorted lists of image and label files
    image_paths = natsorted(os.listdir(os.path.join(dataset_path, 'images')))
    label_paths = natsorted(os.listdir(os.path.join(dataset_path, 'labels')))

    # Process each image-label pair
    for image_path, labels_path in zip(image_paths, label_paths):
        image_path = os.path.join(dataset_path, 'images', image_path)
        labels_path = os.path.join(dataset_path, 'labels', labels_path)
        
        print(f"Processing: {os.path.basename(image_path)}")
        
        reduce_polygon_points(
            labels_path=labels_path, 
            image_path=image_path, 
            show=show, 
            tolerance=tolerance,
            figure_size=figure_size
        )


def main() -> None:
    """
    Main function to parse command line arguments and run the script.
    """
    parser = argparse.ArgumentParser(
        description='Polygon Simplification for YOLO Format Annotations'
    )
    
    parser.add_argument(
        '--path', '-p', 
        required=True,
        help='Path to dataset directory containing "images" and "labels" folders'
    )
    
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=2.0,
        help='Simplification tolerance (default: 2.0)'
    )
    
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Show visualization of before and after simplification'
    )
    
    parser.add_argument(
        '--figure-width', '-fw',
        type=int,
        default=800,
        help='Width of the display window when show=True (default: 800)'
    )
    
    parser.add_argument(
        '--figure-height', '-fh',
        type=int,
        default=600,
        help='Height of the display window when show=True (default: 600)'
    )
    
    args = parser.parse_args()
    
    # Determine figure size if showing visualization
    figure_size = None
    if args.show:
        figure_size = (args.figure_width, args.figure_height)
    
    # Process the dataset
    process_dataset(
        dataset_path=args.path,
        show=args.show,
        tolerance=args.tolerance,
        figure_size=figure_size
    )


if __name__ == "__main__":
    main()
    # Example (debugging):
    # -p ObZea_improvement.v1i.yolov11/train -s