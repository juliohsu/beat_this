#!/usr/bin/env python3
import os
import collections

# Path to annotations directory
annotations_dir = '/home/julio.hsu/beat_this/beat_this/data/annotations'

# Counters for statistics
total_beats_files = 0
files_with_beats = 0
files_with_downbeats = 0
files_with_both = 0
total_minutes_with_both = 0  # Total minutes counter

# Collection to store counts per dataset
dataset_stats = collections.defaultdict(lambda: {
    'total': 0,
    'with_beats': 0,
    'with_downbeats': 0,
    'with_both': 0,
    'minutes_with_both': 0  # Minutes counter for each dataset
})

# List of all datasets for verification
all_datasets = []
for item in os.listdir(annotations_dir):
    if os.path.isdir(os.path.join(annotations_dir, item)) and item != 'annotations':
        all_datasets.append(item)
print(f"Datasets found: {all_datasets}")

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(annotations_dir):
    # Find all .beats files in the current directory
    beats_files = [f for f in files if f.endswith('.beats')]
    
    if beats_files:
        # Get dataset name from the path
        root_parts = root.split(os.sep)
        
        # Determine dataset by looking at the path
        dataset = None
        for part in root_parts:
            if part in all_datasets:
                dataset = part
                break
        
        if dataset is None:
            # Try to get the dataset from path structure
            annotations_index = root_parts.index('annotations') if 'annotations' in root_parts else -1
            if annotations_index > 0 and annotations_index - 1 < len(root_parts):
                dataset = root_parts[annotations_index - 1]
            else:
                dataset = 'unknown'
                
        print(f"Processing {len(beats_files)} files in dataset: {dataset} (path: {root})")
        
        # Process each .beats file
        for beats_file in beats_files:
            total_beats_files += 1
            dataset_stats[dataset]['total'] += 1
            
            file_path = os.path.join(root, beats_file)
            
            has_beats = False
            has_downbeats = False
            last_beat_time = 0  # Track the last beat time to calculate duration
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Print sample for first file in each dataset to debug
                    if dataset_stats[dataset]['total'] == 1 and len(lines) > 0:
                        print(f"\nSample lines from first file in {dataset} ({beats_file}):")
                        for i, line in enumerate(lines[:5]):
                            print(f"  {i+1}: {line.strip()}")
                    
                    for line in lines:
                        # Skip empty lines or comments
                        if line.strip() and not line.strip().startswith('#'):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                # Get the timestamp (first column)
                                try:
                                    time_sec = float(parts[0])
                                    last_beat_time = max(last_beat_time, time_sec)
                                except ValueError:
                                    pass
                                    
                                beat_type = parts[1]
                                
                                # Check for downbeat (1, 1.0, etc.)
                                if beat_type == '1' or beat_type == '1.0' or beat_type.startswith('1.'):
                                    has_downbeats = True
                                # Check for non-downbeat beats (2, 3, 4, etc.)
                                elif beat_type in ['2', '2.0', '3', '3.0', '4', '4.0'] or beat_type.startswith(('2.', '3.', '4.')):
                                    has_beats = True
                                
                            # If we've found both beats and downbeats, we can stop checking for those
                            # but continue to get the last timestamp
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
            
            # Update statistics
            if has_beats:
                files_with_beats += 1
                dataset_stats[dataset]['with_beats'] += 1
            
            if has_downbeats:
                files_with_downbeats += 1
                dataset_stats[dataset]['with_downbeats'] += 1
            
            if has_beats and has_downbeats:
                files_with_both += 1
                dataset_stats[dataset]['with_both'] += 1
                
                # Calculate minutes and add to counters
                minutes = last_beat_time / 60
                total_minutes_with_both += minutes
                dataset_stats[dataset]['minutes_with_both'] += minutes

# Print overall results
print("\n=== OVERALL STATISTICS ===")
print(f"Total .beats files: {total_beats_files}")
print(f"Files with beats: {files_with_beats}")
print(f"Files with downbeats: {files_with_downbeats}")
print(f"Files with both beats and downbeats: {files_with_both}")
print(f"Total minutes of audio with both beats and downbeats: {total_minutes_with_both:.2f}")

# Print per-dataset statistics
print("\n=== STATISTICS BY DATASET ===")
for dataset, stats in sorted(dataset_stats.items()):
    print(f"\n{dataset}:")
    print(f"  Total .beats files: {stats['total']}")
    print(f"  Files with beats: {stats['with_beats']}")
    print(f"  Files with downbeats: {stats['with_downbeats']}")
    print(f"  Files with both beats and downbeats: {stats['with_both']}")
    print(f"  Minutes of audio with both beats and downbeats: {stats['minutes_with_both']:.2f}") 