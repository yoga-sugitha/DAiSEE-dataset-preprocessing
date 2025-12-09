import os
import cv2
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import shutil

# ----------------------------
# Configuration 
# ----------------------------
DAISEE_ROOT = "DAiSEE" 
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet")
LABELS_DIR = os.path.join(DAISEE_ROOT, "Labels")

OUTPUT_ROOT = "DAiSEE_confusion_normal" 

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

def process_clip(args):
    """
    Process one clip independently.
    Args: (clip_id, label, split_name, video_path, out_path)
    Returns: status string ('saved', 'missing', 'error')
    """
    clip_id, label, split_name, video_path, out_path = args

    if not os.path.exists(video_path):
        return "missing"

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        return "missing"

    try:
        # Ensure output directory exists (in case of race condition)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, frame)
        return "saved"
    except Exception:
        return "error"

def main():
    # Load label CSVs
    splits = {}
    for split, csv_path in splits_info.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=["ClipID", "Confusion"])
            splits[split] = df
        else:
            print(f"[WARN] Labels CSV not found for {split}: {csv_path}")

    if not splits:
        raise SystemExit("No label CSVs found. Check LABELS_DIR path.")

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
            
            # Use original clip_id for output filename to preserve extension info
            out_fname = f"{split_name}_{os.path.splitext(clip_id)[0]}.jpg"
            out_path = os.path.join(OUTPUT_ROOT, split_name, class_names[label], out_fname)

            tasks.append((clip_id, label, split_name, video_path, out_path))

    print(f"Total clips to process: {len(tasks)}")

    # Create base output dirs (safe for multiprocessing)
    for split in splits.keys():
        for cname in class_names.values():
            os.makedirs(os.path.join(OUTPUT_ROOT, split, cname), exist_ok=True)

    # But cap at 8 to avoid overloading I/O
    num_workers = max(1, int(cpu_count() * 0.75))

    print(f"Using {num_workers} worker processes...")

    saved = missing = errors = 0

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Extracting frames") as pbar:
            for status in pool.imap_unordered(process_clip, tasks, chunksize=20):
                if status == "saved":
                    saved += 1
                elif status == "missing":
                    missing += 1
                else:
                    errors += 1
                pbar.update(1)

    print("\n==== Done ====")
    print(f"Saved frames: {saved}")
    print(f"Missing/unreadable videos: {missing}")
    print(f"Other errors/skips: {errors}")

    # Optional: zip output for Kaggle
    # print("Zipping output...")
    # shutil.make_archive(OUTPUT_ROOT, 'zip', OUTPUT_ROOT)

if __name__ == "__main__":
    main()