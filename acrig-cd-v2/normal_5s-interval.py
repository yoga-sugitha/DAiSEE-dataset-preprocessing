import os
import cv2
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ===== CONFIGURATION =====
DAISEE_ROOT = "DAiSEE"
DATASET_DIR = os.path.join(DAISEE_ROOT, "DataSet")
LABELS_DIR = os.path.join(DAISEE_ROOT, "Labels")

OUTPUT_ROOT = "DAiSEE_confusion_interval5"
FRAME_INTERVAL = 5  # Extract every Nth frame

CLASS_NAMES = {
    0: "0_not_confused",
    1: "1_slightly_confused",
    2: "2_confused",
    3: "3_very_confused"
}

SPLITS_INFO = {
    "Train": os.path.join(LABELS_DIR, "TrainLabels.csv"),
    "Validation": os.path.join(LABELS_DIR, "ValidationLabels.csv"),
    "Test": os.path.join(LABELS_DIR, "TestLabels.csv")
}

SUPPORTED_EXTENSIONS = ['.avi', '.mp4', '.mov', '.mkv']


# ===== UTILITIES =====
def find_video_file(base_path: str) -> str | None:
    """Find a video file with any supported extension."""
    for ext in SUPPORTED_EXTENSIONS:
        path = base_path + ext
        if os.path.exists(path):
            return path
    return None


def get_video_path(clip_id: str, split: str) -> str | None:
    """Construct and locate the actual video path."""
    clip_base = os.path.splitext(clip_id)[0]
    folder = clip_base[:6]
    base_path = os.path.join(DATASET_DIR, split, folder, clip_base, clip_base)
    return find_video_file(base_path)


def extract_frames_sequential(video_path: str, output_dir: str, clip_base: str, split_name: str, interval: int) -> int:
    """
    Efficiently extract every `interval`-th frame by reading sequentially.
    Returns number of frames successfully saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    frame_idx = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f"{split_name}_{clip_base}_frame{frame_idx:06d}.jpg")
            try:
                cv2.imwrite(out_path, frame)
                saved_count += 1
            except Exception:
                pass  # Silently skip save errors (or log if needed)

        frame_idx += 1

    cap.release()
    return saved_count


def process_clip(args):
    """
    Process a single clip.
    Returns: (status, frame_count, split_name)
    Status: 'saved', 'missing', or 'error'
    """
    clip_id, label, split_name, video_path, output_dir, interval = args

    if not video_path or not os.path.exists(video_path):
        return "missing", 0, split_name

    try:
        frame_count = extract_frames_sequential(video_path, output_dir, clip_id.split('.')[0], split_name, interval)
        status = "saved" if frame_count > 0 else "error"
        return status, frame_count, split_name
    except Exception:
        return "error", 0, split_name


# ===== MAIN =====
def main():
    # Load label CSVs and analyze class distribution
    splits = {}
    for split, csv_path in SPLITS_INFO.items():
        if not os.path.exists(csv_path):
            print(f"⚠️  Missing label file: {csv_path}")
            continue
        df = pd.read_csv(csv_path, usecols=["ClipID", "Confusion"])
        # Filter valid labels
        df = df[df["Confusion"].isin(CLASS_NAMES.keys())]
        splits[split] = df

        print(f"\n{split} class distribution:")
        for class_id in sorted(CLASS_NAMES):
            count = (df["Confusion"] == class_id).sum()
            print(f"  {CLASS_NAMES[class_id]}: {count} videos")

    if not splits:
        print("❌ No valid splits found. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"Frame Extraction Settings:")
    print(f"  Interval: Every {FRAME_INTERVAL} frames")
    print(f"  Output root: {OUTPUT_ROOT}")
    print(f"{'='*60}\n")

    # Build task list
    tasks = []
    for split_name, df in splits.items():
        for _, row in df.iterrows():
            clip_id = str(row["ClipID"])
            label = int(row["Confusion"])

            video_path = get_video_path(clip_id, split_name)
            output_dir = os.path.join(OUTPUT_ROOT, split_name, CLASS_NAMES[label])
            tasks.append((clip_id, label, split_name, video_path, output_dir, FRAME_INTERVAL))

    print(f"Total clips to process: {len(tasks)}\n")

    # Ensure output directories exist
    for split_name in splits:
        for class_dir in CLASS_NAMES.values():
            os.makedirs(os.path.join(OUTPUT_ROOT, split_name, class_dir), exist_ok=True)

    # Multiprocessing
    num_workers = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_workers} worker processes\n")

    stats = {"saved": 0, "missing": 0, "error": 0}
    total_frames = 0
    frames_per_split = {split: 0 for split in splits}

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Extracting frames") as pbar:
            for status, frame_count, split_name in pool.imap_unordered(process_clip, tasks, chunksize=20):
                stats[status] += 1
                total_frames += frame_count
                if status == "saved":
                    frames_per_split[split_name] += frame_count
                pbar.update(1)

    # Final report
    print(f"\n{'='*60}")
    print("Extraction Summary")
    print(f"{'='*60}")
    print(f"Total clips processed: {sum(stats.values())}")
    print(f"  ✓ Saved: {stats['saved']}")
    print(f"  ⚠️ Missing: {stats['missing']}")
    print(f"  ❌ Errors: {stats['error']}")
    print(f"\nTotal frames extracted: {total_frames:,}")
    print(f"Average frames per saved video: {total_frames / max(stats['saved'], 1):.1f}")

    print("\nFrames per split:")
    for split, count in frames_per_split.items():
        print(f"  {split}: {count:,}")

    print(f"\nOutput saved to: {OUTPUT_ROOT}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()