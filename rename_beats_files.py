#!/usr/bin/env python3
import os
import re
import shutil

def rename_beats_files():
    beats_dir = "beat_this/data/annotations/groovemidi/annotations/beats"
    audio_dir = "beat_this/data/audio/groovemidi"
    
    # Get list of audio files to understand the correct naming pattern
    audio_files = []
    if os.path.exists(audio_dir):
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    # Get list of beats files
    beats_files = []
    if os.path.exists(beats_dir):
        beats_files = [f for f in os.listdir(beats_dir) if f.endswith('.beats')]
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Found {len(beats_files)} beats files")
    
    renamed_count = 0
    
    for beats_file in beats_files:
        # Extract the base name without extension
        base_name = beats_file.replace('.beats', '')
        
        # Check if this is an eval session file
        if '_eval_session' in base_name:
            # Pattern: groovemidi_drummer1_eval_session10_soul-groove10_102_beat_4-4
            # Should be: groovemidi_drummer1_eval_session_10_soul-groove10_102_beat_4-4
            pattern = r'(_eval_session)(\d+)(_.*)'
            match = re.search(pattern, base_name)
            if match:
                new_name = base_name.replace(match.group(0), f"{match.group(1)}_{match.group(2)}{match.group(3)}")
                new_beats_file = new_name + '.beats'
                
                # Check if corresponding audio file exists
                corresponding_audio = new_name + '.wav'
                if corresponding_audio in audio_files:
                    old_path = os.path.join(beats_dir, beats_file)
                    new_path = os.path.join(beats_dir, new_beats_file)
                    
                    if not os.path.exists(new_path):
                        print(f"Renaming: {beats_file} -> {new_beats_file}")
                        shutil.move(old_path, new_path)
                        renamed_count += 1
                    else:
                        print(f"Target already exists: {new_beats_file}")
                else:
                    print(f"No corresponding audio file found for: {corresponding_audio}")
        
        # Check if this is a regular session file
        elif '_session' in base_name and '_eval_' not in base_name:
            # Pattern: groovemidi_drummer10_session110_jazz-swing_110_beat_4-4
            # Should be: groovemidi_drummer10_session1_10_jazz-swing_110_beat_4-4
            pattern = r'(_session)(\d+)(_.*)'
            match = re.search(pattern, base_name)
            if match:
                session_num = match.group(2)
                # Split session number: first digit is session, rest is track
                if len(session_num) >= 2:
                    session_part = session_num[0]
                    track_part = session_num[1:]
                    new_name = base_name.replace(match.group(0), f"{match.group(1)}{session_part}_{track_part}{match.group(3)}")
                    new_beats_file = new_name + '.beats'
                    
                    # Check if corresponding audio file exists
                    corresponding_audio = new_name + '.wav'
                    if corresponding_audio in audio_files:
                        old_path = os.path.join(beats_dir, beats_file)
                        new_path = os.path.join(beats_dir, new_beats_file)
                        
                        if not os.path.exists(new_path):
                            print(f"Renaming: {beats_file} -> {new_beats_file}")
                            shutil.move(old_path, new_path)
                            renamed_count += 1
                        else:
                            print(f"Target already exists: {new_beats_file}")
                    else:
                        print(f"No corresponding audio file found for: {corresponding_audio}")
    
    print(f"\nTotal files renamed: {renamed_count}")

if __name__ == "__main__":
    rename_beats_files() 