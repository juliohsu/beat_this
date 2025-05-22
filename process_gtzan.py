import os
import glob
import torch
from tqdm import tqdm
import argparse
from beat_this.custom_cqt.custom_inference import File2File

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process GTZAN dataset using multiple GPUs')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (0 or 1)')
    parser.add_argument('--num-gpus', type=int, default=2, help='Total number of GPUs to use')
    args = parser.parse_args()
    
    # Path to checkpoint
    checkpoint_path = "/home/julio.hsu/beat_this/launch_scripts/checkpoints/cqt-train S0 shift_tolerant_weighted_bce-h512-augTrueTrueTrue.ckpt"
    
    # Input GTZAN directory
    input_dir = "/home/julio.hsu/beat_this/beat_this/data/audio/gtzan"
    
    # Output directory
    output_dir = "/home/julio.hsu/beat_this/moises_results/train_cqt_beats"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # Initialize the beat tracking model with custom CQT
    beat_tracker = File2File(checkpoint_path=checkpoint_path, device=device)
    
    # Get all .wav files in the GTZAN directory
    all_wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    
    # Split files based on GPU ID
    total_files = len(all_wav_files)
    files_per_gpu = total_files // args.num_gpus
    start_idx = args.gpu_id * files_per_gpu
    end_idx = start_idx + files_per_gpu if args.gpu_id < args.num_gpus - 1 else total_files
    
    # Get subset of files for this GPU
    wav_files = all_wav_files[start_idx:end_idx]
    
    print(f"GPU {args.gpu_id}: Processing {len(wav_files)} files out of {total_files} total")
    
    # Process each audio file
    for audio_path in tqdm(wav_files, desc=f"GPU {args.gpu_id} processing GTZAN files"):
        # Get the filename without extension
        filename = os.path.basename(audio_path)[:-4]
        
        # Define output path
        output_path = os.path.join(output_dir, f"{filename}.beats")
        
        try:
            # Process the audio file and save the beats using custom CQT
            beat_tracker(audio_path, output_path)
            print(f"Processed {filename} -> {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"GPU {args.gpu_id} complete! Processed {len(wav_files)} files.")

if __name__ == "__main__":
    main() 