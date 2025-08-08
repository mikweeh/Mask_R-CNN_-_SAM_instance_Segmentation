#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename Matcher Script
This script reads two text files containing comma-separated filenames.
It finds entries in the second list that start with entries from the first list
and outputs the matched filenames with their full roboflow names.
"""

# File paths
FIRST_FILE = "dataset/list1.txt"
SECOND_FILE = "dataset/test.txt" 

def read_filename_list(file_path):
    """
    Read a text file and return a list of filenames.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of cleaned filenames
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    # Split by comma and clean whitespace
    filenames = [filename.strip() for filename in content.split(',') 
                 if filename.strip()]
    return filenames


def match_filenames(list1, list2):
    """
    Match filenames from list1 with corresponding entries in list2.
    
    Args:
        list1 (list): List of base filenames (without extensions)
        list2 (list): List of roboflow filenames (with .rf. suffix)
        
    Returns:
        list: List of matched filenames from list2 in order of list1
    """
    matched_filenames = []
    
    for base_filename in list1:
        matched = None
        # Search for a filename in list2 that starts with base_filename
        for rf_filename in list2:
            if rf_filename.startswith(base_filename):
                matched = rf_filename
                break
        
        # If no match found, keep the original filename
        if matched:
            matched_filenames.append(matched)
        else:
            print(f"Warning: No match found for '{base_filename}'")
            matched_filenames.append(base_filename)
    
    return matched_filenames


def write_output_list(output_list, output_file_path):
    """
    Write the matched filenames to an output file.
    
    Args:
        output_list (list): List of matched filenames
        output_file_path (str): Path to the output file
    """
    with open(output_file_path, 'w', encoding='utf-8') as file:
        # Write as comma-separated values with trailing comma
        for filename in output_list:
            file.write(f"{filename},\n")


def main():
    """
    Main function to execute the filename matching process.
    """
    # File paths
    first_file = FIRST_FILE
    second_file = SECOND_FILE
    output_file = "matched_list.txt"
    
    try:
        # Read both input files
        print("Reading first file...")
        first_list = read_filename_list(first_file)
        print(f"Found {len(first_list)} filenames in first list")
        
        print("Reading second file...")
        second_list = read_filename_list(second_file)
        print(f"Found {len(second_list)} filenames in second list")
        
        # Match filenames
        print("Matching filenames...")
        matched_list = match_filenames(first_list, second_list)
        
        # Write output
        print(f"Writing matched list to {output_file}...")
        write_output_list(matched_list, output_file)
        
        # Print summary
        matches_found = sum(1 for i, filename in enumerate(matched_list) 
                           if filename != first_list[i])
        print(f"Matching complete! Found {matches_found} matches out of "
              f"{len(first_list)} entries.")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
