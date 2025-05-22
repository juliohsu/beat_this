#!/usr/bin/env python3

import os
import sys
import numpy as np
from pathlib import Path
from itertools import chain
import argparse
import re

def validate_and_normalize_beats_file(file_path):
    """
    Validate a beats file and normalize its format.
    
    Args:
        file_path (Path): Path to the beats file
        
    Returns:
        tuple: (beat_times, beat_values) if valid, None otherwise
    """
    try:
        # Load the file content
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
        
        if not lines:
            print(f"Warning: Empty file {file_path}")
            return None
        
        # Parse the file content
        parsed_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    # Time should be convertible to float
                    time = float(parts[0])
                    
                    # Beat value (if exists) should be a number
                    beat_value = int(float(parts[1])) if len(parts) >= 2 else 0
                    
                    parsed_lines.append((time, beat_value))
                except (ValueError, IndexError):
                    print(f"Warning: Invalid line format in {file_path}: {line}")
        
        if not parsed_lines:
            print(f"Warning: No valid lines in {file_path}")
            return None
        
        # Sort by time
        parsed_lines.sort(key=lambda x: x[0])
        
        # Convert to numpy arrays
        times = np.array([x[0] for x in parsed_lines])
        values = np.array([x[1] for x in parsed_lines])
        
        return times, values
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def normalize_beat_values(beat_times, beat_values):
    """
    Normalize beat values to ensure they follow the pattern:
    1 for downbeats, 2,3,4,... for other beats in each measure
    
    Args:
        beat_times (np.ndarray): Array of beat times
        beat_values (np.ndarray): Array of beat values
        
    Returns:
        np.ndarray: Normalized beat values
    """
    # If there are no explicit beat values (empty or all zeros), 
    # create a default pattern of 1,2,3,4,1,2,3,4,...
    if len(beat_values) == 0 or (beat_values == 0).all():
        beat_values = np.ones_like(beat_times, dtype=np.int32)
        # Default to 4/4 time signature if we can't determine otherwise
        for i in range(1, len(beat_values)):
            beat_values[i] = (i % 4) + 1
        return beat_values
    
    # If there are explicit values but none are 1 (downbeats),
    # assume the first beat is a downbeat and every 4th beat after that
    if 1 not in beat_values:
        new_values = np.ones_like(beat_values)
        for i in range(len(new_values)):
            new_values[i] = (i % 4) + 1
        return new_values
    
    # Ensure all beat values are integers
    normalized = np.array(beat_values, dtype=np.int32)
    
    # In some datasets, the values might use different conventions
    # e.g., 1.0 for downbeat, or 0 for regular beats
    if np.any(normalized <= 0):
        # Replace 0s with appropriate values
        counter = 1
        for i in range(len(normalized)):
            if normalized[i] == 1:
                counter = 1
            else:
                if normalized[i] <= 0:
                    counter += 1
                    normalized[i] = counter
                    if counter >= 4:  # Assume 4/4 time signature
                        counter = 0
    
    return normalized

def save_formatted_beats_file(output_path, beat_times, beat_values):
    """
    Save beat information to a properly formatted .beats file
    
    Args:
        output_path (Path): Path to save the formatted file
        beat_times (np.ndarray): Array of beat times
        beat_values (np.ndarray): Array of beat values
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for time, value in zip(beat_times, beat_values):
            f.write(f"{time:.6f}\t{value}\n")
            
def process_dataset(dataset_dir, remove_invalid=False):
    """
    Process all .beats files in a dataset directory
    
    Args:
        dataset_dir (Path): Path to the dataset directory containing annotations/beats
        remove_invalid (bool): If True, remove invalid .beats files
    """
    beats_dir = dataset_dir / 'annotations' / 'beats'
    if not beats_dir.exists():
        print(f"Beats directory not found: {beats_dir}")
        return 0, 0, 0
    
    # Create a backup directory
    backup_dir = beats_dir.parent / 'beats_backup'
    backup_dir.mkdir(exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    removed_count = 0
    
    for beats_file in beats_dir.glob('*.beats'):
        # First create a backup
        backup_file = backup_dir / beats_file.name
        if not backup_file.exists():
            try:
                with open(beats_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
            except Exception as e:
                print(f"Error creating backup of {beats_file}: {str(e)}")
                continue
        
        # Process the file
        result = validate_and_normalize_beats_file(beats_file)
        if result:
            beat_times, beat_values = result
            
            # Check if file has actual beats
            if len(beat_times) == 0:
                if remove_invalid:
                    print(f"Removing empty beats file: {beats_file}")
                    beats_file.unlink()
                    removed_count += 1
                else:
                    skipped_count += 1
                continue
            
            # Normalize beat values
            normalized_values = normalize_beat_values(beat_times, beat_values)
            
            # Save the formatted file
            save_formatted_beats_file(beats_file, beat_times, normalized_values)
            processed_count += 1
        else:
            if remove_invalid:
                print(f"Removing invalid beats file: {beats_file}")
                beats_file.unlink()
                removed_count += 1
            else:
                skipped_count += 1
    
    print(f"Dataset {dataset_dir.name}: Processed {processed_count} files, skipped {skipped_count} files, removed {removed_count} files")
    return processed_count, skipped_count, removed_count

def main():
    parser = argparse.ArgumentParser(description='Format .beats files in each dataset')
    parser.add_argument('--data-dir', default='beat_this/data/annotations', 
                        help='Path to annotations directory (default: beat_this/data/annotations)')
    parser.add_argument('--remove-invalid', action='store_true',
                        help='Remove invalid or empty .beats files')
    args = parser.parse_args()
    
    annotations_path = Path(args.data_dir)
    if not annotations_path.exists():
        print(f"Error: Annotations directory not found: {annotations_path}")
        return 1
    
    total_processed = 0
    total_skipped = 0
    total_removed = 0
    dataset_results = []
    
    # Process each dataset directory
    for dataset_dir in annotations_path.iterdir():
        if dataset_dir.is_dir():
            print(f"Processing dataset: {dataset_dir.name}")
            processed, skipped, removed = process_dataset(dataset_dir, args.remove_invalid)
            total_processed += processed
            total_skipped += skipped
            total_removed += removed
            dataset_results.append((dataset_dir.name, processed, skipped, removed))
    
    # Print summary
    print("\nSummary:")
    print(f"Total files processed: {total_processed}")
    print(f"Total files skipped: {total_skipped}")
    print(f"Total files removed: {total_removed}")
    print("\nBy dataset:")
    for dataset, processed, skipped, removed in dataset_results:
        if processed > 0 or skipped > 0 or removed > 0:
            print(f"  {dataset}: {processed} processed, {skipped} skipped, {removed} removed")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
