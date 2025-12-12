import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

DAISEE_ROOT = "DAiSEE" 
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet")
LABELS_DIR = os.path.join(DAISEE_ROOT, "Labels")

OUTPUT_ROOT = "DAiSEE_confusion_interval5" 

# Frame extraction settings
FRAME_INTERVAL = 5  # Extract every 5th frame

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
    supported_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    
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

def extract_frames_with_interval(video_path, output_dir, clip_base, split_name, interval=5):
    """
    Extract frames at fixed intervals
    For 10s video at 30fps (300 frames), interval=5 gives ~60 frames
    """
    if not os.path.exists(video_path):
        return 0
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        cap.release()
        return 0
    
    # Extract frames at intervals
    saved_count = 0
    frame_idx = 0
    
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        
        if success:
            try:
                out_path = os.path.join(output_dir, f"{split_name}_{clip_base}_frame{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1
            except Exception:
                pass
        
        frame_idx += interval
    
    cap.release()
    return saved_count

def process_clip_interval(args):
    """
    Process clip with interval-based frame extraction
    """
    clip_id, label, split_name, video_path, output_dir, interval = args

    if not os.path.exists(video_path):
        return "missing", 0

    clip_base = os.path.splitext(clip_id)[0]
    
    try:
        saved_count = extract_frames_with_interval(video_path, output_dir, clip_base, split_name, interval)
        if saved_count > 0:
            return "saved", saved_count
        else:
            return "error", 0
    except Exception as e:
        return "error", 0

def main():
    # Load data and show class distributions
    splits = {}
    class_distributions = {}
    
    for split, csv_path in splits_info.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=["ClipID", "Confusion"])
            splits[split] = df
            class_distributions[split] = dict(df['Confusion'].value_counts().sort_index())
            print(f"\n{split} class distribution:")
            for class_id in sorted(class_distributions[split].keys()):
                count = class_distributions[split][class_id]
                class_name = class_names.get(class_id, f"Unknown_{class_id}")
                print(f"  {class_name}: {count} videos")
    
    print(f"\n{'='*60}")
    print(f"Frame Extraction Settings:")
    print(f"  Interval: Every {FRAME_INTERVAL} frames")
    print(f"  Expected frames per 10s@30fps video: ~{300//FRAME_INTERVAL} frames")
    print(f"{'='*60}\n")
    
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
            
            tasks.append((clip_id, label, split_name, video_path, output_dir, FRAME_INTERVAL))
    
    print(f"Total clips to process: {len(tasks)}\n")
    
    # Create output directories
    for split in splits.keys():
        for cname in class_names.values():
            os.makedirs(os.path.join(OUTPUT_ROOT, split, cname), exist_ok=True)
    
    # Process with multiprocessing
    num_workers = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_workers} worker processes\n")
    
    stats = {'saved': 0, 'missing': 0, 'error': 0}
    total_frames = 0
    frames_per_split = {split: 0 for split in splits.keys()}
    
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Extracting frames") as pbar:
            for (status, frame_count), (clip_id, label, split_name, _, _, _) in zip(
                pool.imap_unordered(process_clip_interval, tasks, chunksize=10),
                tasks
            ):
                stats[status] += 1
                total_frames += frame_count
                if status == "saved":
                    frames_per_split[split_name] += frame_count
                pbar.update(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Extraction Summary:")
    print(f"{'='*60}")
    print(f"Total videos processed: {sum(stats.values())}")
    print(f"  ✓ Successfully extracted: {stats['saved']}")
    print(f"  ✗ Missing videos: {stats['missing']}")
    print(f"  ✗ Errors: {stats['error']}")
    print(f"\nTotal frames extracted: {total_frames:,}")
    print(f"Average frames per video: {total_frames/max(stats['saved'], 1):.1f}")
    
    print(f"\nFrames per split:")
    for split_name, count in frames_per_split.items():
        print(f"  {split_name}: {count:,} frames")
    
    print(f"\nOutput directory: {OUTPUT_ROOT}/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()