import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

# ----------------------------
# Configuration 
# ----------------------------
DAISEE_ROOT = "DAiSEE" 
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet")
LABELS_DIR = os.path.join(DAISEE_ROOT, "Labels")

OUTPUT_ROOT = "DAiSEE_confusion_oversampling" 

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

# Temporal sampling parameters
AVERAGE_VIDEO_DURATION = 10  # seconds
FPS = 30
MIN_INTERVAL = 15  # frames (0.5 seconds)
MAX_INTERVAL = 60  # frames (2.0 seconds)

def calculate_log_scaled_intervals(class_counts):
    """
    Calculate log-scaled inverse frequency temporal sampling intervals
    """
    N_min = min(class_counts.values())  # Minority class count (class 3)
    
    # Calculate relative weights
    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = count / N_min
    
    # Find maximum weight for normalization
    max_weight = max(weights.values())
    
    # Calculate log-scaled intervals
    intervals = {}
    for class_id, weight in weights.items():
        if weight == 1.0:  # Minority class
            intervals[class_id] = MIN_INTERVAL
        else:
            # Log scaling formula: interval = MIN + (log(weight)/log(max_weight)) * (MAX - MIN)
            log_scaled = MIN_INTERVAL + (np.log(weight) / np.log(max_weight)) * (MAX_INTERVAL - MIN_INTERVAL)
            intervals[class_id] = int(np.round(log_scaled))
    
    return intervals

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
    clip_base = os.path.splitext(clip_id)[0]
    folder = clip_base[:6]
    
    base_path = os.path.join(DATASET_DIR, split, folder, clip_base, clip_base)
    video_path = find_video_file(base_path)
    
    if video_path:
        return video_path
    
    fallback_path = base_path + '.avi'
    return fallback_path

def extract_frames_fixed_interval(video_path, output_dir, clip_base, split_name, sampling_interval):
    """
    Extract frames at fixed temporal intervals
    """
    if not os.path.exists(video_path):
        return 0
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return 0
    
    # Calculate frame indices to extract
    frame_indices = list(range(0, total_frames, sampling_interval))
    
    # Ensure we get at least one frame even from very short videos
    if not frame_indices and total_frames > 0:
        frame_indices = [0]
    
    saved_count = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        
        if success:
            try:
                out_path = os.path.join(output_dir, f"{split_name}_{clip_base}_frame{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1
            except Exception as e:
                continue
    
    cap.release()
    return saved_count

def process_clip_temporal(args):
    """
    Process clip with temporal oversampling
    """
    clip_id, label, split_name, video_path, output_dir, sampling_interval = args

    if not os.path.exists(video_path):
        return "missing", 0

    clip_base = os.path.splitext(clip_id)[0]
    
    try:
        saved_count = extract_frames_fixed_interval(video_path, output_dir, clip_base, split_name, sampling_interval)
        
        if saved_count > 0:
            return "saved", saved_count
        else:
            return "no_frames", 0
    except Exception as e:
        return "error", 0

def main():
    # Load label CSVs
    splits = {}
    class_distributions = {}
    
    for split, csv_path in splits_info.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=["ClipID", "Confusion"])
            splits[split] = df
            class_distributions[split] = dict(df['Confusion'].value_counts().sort_index())
            print(f"{split} class distribution: {class_distributions[split]}")
        else:
            print(f"[WARN] Labels CSV not found for {split}: {csv_path}")

    if not splits:
        raise SystemExit("No label CSVs found. Check LABELS_DIR path.")

    # Calculate log-scaled sampling intervals using training set distribution
    train_counts = class_distributions['Train']
    sampling_intervals = calculate_log_scaled_intervals(train_counts)
    
    print(f"\n🎯 Log-Scaled Inverse Frequency Temporal Sampling Intervals:")
    print(f"Minimum interval: {MIN_INTERVAL} frames ({MIN_INTERVAL/FPS:.1f}s)")
    print(f"Maximum interval: {MAX_INTERVAL} frames ({MAX_INTERVAL/FPS:.1f}s)")
    print(f"\nSampling intervals per class:")
    for class_id, interval in sorted(sampling_intervals.items()):
        interval_seconds = interval / FPS
        class_name = class_names[class_id]
        print(f"  {class_name}: {interval} frames ({interval_seconds:.1f}s)")
    
    # Calculate expected frame distribution
    print(f"\n�� Expected Frame Distribution:")
    for split_name, class_counts in class_distributions.items():
        total_frames = 0
        frame_distribution = {}
        
        for class_id, count in class_counts.items():
            interval = sampling_intervals[class_id]
            # Estimate frames per video (assuming 300 frames average)
            frames_per_video = 300 // interval
            class_frames = count * frames_per_video
            frame_distribution[class_id] = class_frames
            total_frames += class_frames
        
        print(f"\n{split_name}:")
        for class_id in sorted(frame_distribution.keys()):
            frames = frame_distribution[class_id]
            percentage = (frames / total_frames) * 100
            interval = sampling_intervals[class_id]
            print(f"  Class {class_id}: {frames:,} frames ({percentage:.1f}%) - interval: {interval}frames")

    # Prepare task list
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
            
            sampling_interval = sampling_intervals[label]
            tasks.append((clip_id, label, split_name, video_path, output_dir, sampling_interval))

    print(f"\nTotal clips to process: {len(tasks)}")

    # Create base output dirs
    for split in splits.keys():
        for cname in class_names.values():
            os.makedirs(os.path.join(OUTPUT_ROOT, split, cname), exist_ok=True)

    # Process with multiprocessing
    num_workers = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_workers} worker processes...")

    stats = {'saved': 0, 'missing': 0, 'no_frames': 0, 'error': 0}
    total_frames_extracted = 0
    frames_per_class = {0: 0, 1: 0, 2: 0, 3: 0}

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Temporal oversampling") as pbar:
            for status, frame_count in pool.imap_unordered(process_clip_temporal, tasks, chunksize=10):
                stats[status] += 1
                total_frames_extracted += frame_count
                
                # Track frames per class (approximate)
                if status == 'saved' and frame_count > 0:
                    # We don't have class info here, but we can estimate from task order
                    pass
                
                pbar.update(1)

    print("\n==== Temporal Oversampling Complete ====")
    print(f"Successfully processed: {stats['saved']}")
    print(f"Missing videos: {stats['missing']}")
    print(f"Videos with no frames extracted: {stats['no_frames']}")
    print(f"Errors: {stats['error']}")
    print(f"Total frames extracted: {total_frames_extracted}")

    # Save sampling configuration for reproducibility
    config_path = os.path.join(OUTPUT_ROOT, "sampling_configuration.txt")
    with open(config_path, 'w') as f:
        f.write("Log-Scaled Inverse Frequency Temporal Sampling Configuration\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: DAiSEE\n")
        f.write(f"Average video duration: {AVERAGE_VIDEO_DURATION}s\n")
        f.write(f"FPS: {FPS}\n")
        f.write(f"Minimum interval: {MIN_INTERVAL} frames ({MIN_INTERVAL/FPS:.1f}s)\n")
        f.write(f"Maximum interval: {MAX_INTERVAL} frames ({MAX_INTERVAL/FPS:.1f}s)\n")
        f.write(f"Random seed: 42\n\n")
        f.write("Class distribution (Train set):\n")
        for class_id, count in sorted(train_counts.items()):
            f.write(f"  Class {class_id}: {count} videos\n")
        f.write("\nSampling intervals:\n")
        for class_id, interval in sorted(sampling_intervals.items()):
            interval_seconds = interval / FPS
            f.write(f"  Class {class_id}: {interval} frames ({interval_seconds:.1f}s)\n")
        f.write(f"\nTotal frames extracted: {total_frames_extracted}\n")

    print(f"\nConfiguration saved to: {config_path}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
