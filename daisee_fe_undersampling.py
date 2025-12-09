import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

DAISEE_ROOT = "DAiSEE" 
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet")
LABELS_DIR = os.path.join(DAISEE_ROOT, "Labels")

OUTPUT_ROOT = "DAiSEE_confusion_undersampling" 

class_names = {
    0: "0_not_confused",
    1: "1_slightly_confused",
    2: "2_confused",
    3: "3_very_confused"
}

splits_info = {
    "Train": os.path.join(LABELS_DIR, "TrainLabels.csv"),
    "Validation": os.path.join(LABELS_DIR, "ValidationLabels.csv"),
    "Test": os.path.join(LABELS_DIR, "TestLabels.csv")
}
def find_video_file(base_path):
    """Find video file with any supported extension"""
    supported_extensions = ['.avi', '.mp4', '.mov', '.mkv']  # Add more if needed
    
    for ext in supported_extensions:
        video_path = base_path + ext
        if os.path.exists(video_path):
            return video_path
    
    return None

def get_video_path(clip_id, split):
    """Get video path handling different extensions"""
    clip_base = os.path.splitext(clip_id)[0]  # Remove any extension
    folder = clip_base[:6]
    
    # Base path without extension
    base_path = os.path.join(DATASET_DIR, split, folder, clip_base, clip_base)
    
    # Try to find the actual file with any extension
    video_path = find_video_file(base_path)
    
    if video_path:
        return video_path
    
    # Fallback: try with .avi (original behavior)
    fallback_path = base_path + '.avi'
    return fallback_path

def calculate_proportional_sampling_rates(class_counts, minority_classes):
    """
    Calculate sampling rates using proportional undersampling
    r_c = N_minority / N_c
    """
    # Find the smallest minority class count
    minority_count = min([class_counts[c] for c in minority_classes])
    
    sampling_rates = {}
    for class_id, count in class_counts.items():
        if class_id in minority_classes:
            sampling_rates[class_id] = 1.0
        else:
            sampling_rates[class_id] = minority_count / count
    
    return sampling_rates

def extract_frames_proportional(video_path, output_dir, clip_base, split_name, sampling_rate, min_frames=5):
    """
    Extract frames with proportional sampling rate
    """
    if not os.path.exists(video_path):
        return 0
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return 0
    
    if sampling_rate >= 0.99:  # Extract all frames
        frames_to_extract = total_frames
        frame_indices = range(total_frames)
    else:
        # Calculate how many frames to sample
        frames_to_extract = max(min_frames, int(total_frames * sampling_rate))
        frames_to_extract = min(frames_to_extract, total_frames)
        
        # Randomly sample frames
        frame_indices = sorted(random.sample(range(total_frames), frames_to_extract))
    
    saved_count = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        
        if success:
            try:
                out_path = os.path.join(output_dir, f"{split_name}_{clip_base}_frame{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1
            except Exception:
                continue
    
    cap.release()
    return saved_count

def process_clip_proportional(args):
    """
    Process clip with proportional undersampling
    """
    clip_id, label, split_name, video_path, output_dir, sampling_rate = args

    if not os.path.exists(video_path):
        return "missing", 0

    clip_base = os.path.splitext(clip_id)[0]
    
    try:
        saved_count = extract_frames_proportional(video_path, output_dir, clip_base, split_name, sampling_rate)
        if saved_count > 0:
            return "saved", saved_count
        else:
            return "error", 0
    except Exception as e:
        return "error", 0

# In your main function:
def main():
    # Load your data
    splits = {}
    class_distributions = {}
    
    for split, csv_path in splits_info.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=["ClipID", "Confusion"])
            splits[split] = df
            class_distributions[split] = dict(df['Confusion'].value_counts().sort_index())
            print(f"{split} class distribution: {class_distributions[split]}")
    
    # Calculate sampling rates using YOUR proportional formula
    train_counts = class_distributions['Train']
    MINORITY_CLASSES = {3}  # Only class 3 as minority
    
    sampling_rates = calculate_proportional_sampling_rates(train_counts, MINORITY_CLASSES)
    
    print("\n🎯 Proportional Undersampling Rates (Your Formula):")
    for class_id, rate in sorted(sampling_rates.items()):
        class_name = class_names[class_id]
        print(f"  {class_name}: {rate:.4f} ({rate*100:.2f}%)")
    
    # Prepare tasks
    tasks = []
    for split_name, df in splits.items():
        for _, row in df.iterrows():
            try:
                label = int(row["Confusion"])
            except (ValueError, TypeError):
                continue
            if label not in class_names:
                continue

            clip_id = row["ClipID"]
            video_path = get_video_path(clip_id, split_name)
            output_dir = os.path.join(OUTPUT_ROOT, split_name, class_names[label])
            
            sample_rate = sampling_rates.get(label, 1.0)
            tasks.append((clip_id, label, split_name, video_path, output_dir, sample_rate))
    
    # Continue with processing...
    print(f"\nTotal clips to process: {len(tasks)}")
    
    # Create directories
    for split in splits.keys():
        for cname in class_names.values():
            os.makedirs(os.path.join(OUTPUT_ROOT, split, cname), exist_ok=True)
    
    # Process with multiprocessing
    num_workers = max(1, int(cpu_count() * 0.75))
    
    stats = {'saved': 0, 'missing': 0, 'error': 0}
    total_frames = 0
    
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Proportional undersampling") as pbar:
            for status, frame_count in pool.imap_unordered(process_clip_proportional, tasks, chunksize=10):
                stats[status] += 1
                total_frames += frame_count
                pbar.update(1)
    
    print(f"\nExtracted {total_frames} total frames")
    print(f"Final class balance will be much closer to equal!")

if __name__ == "__main__":
    random.seed(42)
    main()
