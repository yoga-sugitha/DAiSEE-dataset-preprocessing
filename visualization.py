import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# ============================================================================
# CONFIGURATION - Change this to match your extraction output
# ============================================================================
EXTRACTION_ROOT = "DAiSEE_class_balanced"  # Change to your output folder
# Options: "DAiSEE_class_balanced", "DAiSEE_adaptive_interval", 
#          "DAiSEE_proportional_undersampling", "DAiSEE_confusion_interval5"

# ============================================================================

class_names = {
    "0_not_confused": "Not Confused",
    "1_slightly_confused": "Slightly Confused",
    "2_confused": "Confused",
    "3_very_confused": "Very Confused"
}

def count_frames_per_class(root_dir):
    """Count frames for each class in each split"""
    data = defaultdict(lambda: defaultdict(int))
    clips_per_class = defaultdict(lambda: defaultdict(set))
    
    for split in ["Train", "Validation", "Test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            print(f"⚠️  Warning: {split_path} not found, skipping...")
            continue
        
        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            # Count image files
            frames = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            data[split][class_folder] = len(frames)
            
            # Extract unique clip IDs
            for frame in frames:
                # Frame format: Train_110001_frame000005.jpg
                parts = frame.split('_frame')
                if len(parts) > 0:
                    clip_id = parts[0]
                    clips_per_class[split][class_folder].add(clip_id)
    
    # Convert sets to counts
    clips_count = defaultdict(lambda: defaultdict(int))
    for split in clips_per_class:
        for class_folder in clips_per_class[split]:
            clips_count[split][class_folder] = len(clips_per_class[split][class_folder])
    
    return data, clips_count

def plot_frame_distribution(data, output_file="frame_distribution.png"):
    """Plot frame count per class for each split"""
    splits = ["Train", "Validation", "Test"]
    class_folders = sorted(class_names.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Frame Distribution - {os.path.basename(EXTRACTION_ROOT)}', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    for idx, split in enumerate(splits):
        ax = axes[idx]
        
        counts = [data[split].get(cf, 0) for cf in class_folders]
        labels = [class_names[cf] for cf in class_folders]
        
        bars = ax.bar(range(len(class_folders)), counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Number of Frames', fontsize=12)
        ax.set_title(f'{split}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_folders)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_frames_per_clip(data, clips_count, output_file="frames_per_clip.png"):
    """Plot average frames per clip for each class"""
    splits = ["Train", "Validation", "Test"]
    class_folders = sorted(class_names.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Average Frames per Clip - {os.path.basename(EXTRACTION_ROOT)}', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    for idx, split in enumerate(splits):
        ax = axes[idx]
        
        avg_frames = []
        for cf in class_folders:
            total_frames = data[split].get(cf, 0)
            num_clips = clips_count[split].get(cf, 1)  # Avoid division by zero
            avg_frames.append(total_frames / num_clips if num_clips > 0 else 0)
        
        labels = [class_names[cf] for cf in class_folders]
        bars = ax.bar(range(len(class_folders)), avg_frames, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, avg in zip(bars, avg_frames):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{avg:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Avg Frames per Clip', fontsize=12)
        ax.set_title(f'{split}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_folders)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Max (60 frames)')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_class_balance_ratio(data, output_file="class_balance_ratio.png"):
    """Plot class balance ratio (relative to smallest class)"""
    splits = ["Train", "Validation", "Test"]
    class_folders = sorted(class_names.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Class Balance Ratio - {os.path.basename(EXTRACTION_ROOT)}', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    for idx, split in enumerate(splits):
        ax = axes[idx]
        
        counts = [data[split].get(cf, 0) for cf in class_folders]
        min_count = min(counts) if min(counts) > 0 else 1
        ratios = [c / min_count for c in counts]
        
        labels = [class_names[cf] for cf in class_folders]
        bars = ax.bar(range(len(class_folders)), ratios, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.1f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Ratio (relative to smallest)', fontsize=12)
        ax.set_title(f'{split}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_folders)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Balance')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def plot_combined_summary(data, clips_count, output_file="combined_summary.png"):
    """Create a comprehensive summary visualization"""
    splits = ["Train", "Validation", "Test"]
    class_folders = sorted(class_names.keys())
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Extraction Summary - {os.path.basename(EXTRACTION_ROOT)}', 
                 fontsize=18, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    
    # 1. Total frames per split
    ax1 = plt.subplot(2, 3, 1)
    split_totals = [sum(data[split].values()) for split in splits]
    bars = ax1.bar(splits, split_totals, color='steelblue', alpha=0.8, edgecolor='black')
    for bar, total in zip(bars, split_totals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(total):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Frames', fontsize=11)
    ax1.set_title('Total Frames per Split', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Total clips per split
    ax2 = plt.subplot(2, 3, 2)
    clip_totals = [sum(clips_count[split].values()) for split in splits]
    bars = ax2.bar(splits, clip_totals, color='coral', alpha=0.8, edgecolor='black')
    for bar, total in zip(bars, clip_totals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(total):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Clips', fontsize=11)
    ax2.set_title('Total Clips per Split', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Average frames per clip per split
    ax3 = plt.subplot(2, 3, 3)
    avg_per_split = [split_totals[i] / clip_totals[i] if clip_totals[i] > 0 else 0 
                     for i in range(len(splits))]
    bars = ax3.bar(splits, avg_per_split, color='mediumseagreen', alpha=0.8, edgecolor='black')
    for bar, avg in zip(bars, avg_per_split):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Avg Frames per Clip', fontsize=11)
    ax3.set_title('Overall Avg Frames/Clip', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4-6. Stacked bar charts for each split
    for idx, split in enumerate(splits):
        ax = plt.subplot(2, 3, 4 + idx)
        
        counts = [data[split].get(cf, 0) for cf in class_folders]
        labels = [class_names[cf] for cf in class_folders]
        
        bars = ax.bar(range(len(class_folders)), counts, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{int(count):,}',
                       ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        ax.set_ylabel('Frames', fontsize=10)
        ax.set_title(f'{split} Distribution', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(class_folders)))
        ax.set_xticklabels([cn.split()[0] for cn in labels], rotation=0, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()

def print_statistics(data, clips_count):
    """Print detailed statistics"""
    print(f"\n{'='*70}")
    print(f"STATISTICS - {os.path.basename(EXTRACTION_ROOT)}")
    print(f"{'='*70}\n")
    
    splits = ["Train", "Validation", "Test"]
    class_folders = sorted(class_names.keys())
    
    for split in splits:
        print(f"📊 {split}:")
        print(f"   {'Class':<25} {'Frames':>12} {'Clips':>10} {'Frames/Clip':>12}")
        print(f"   {'-'*60}")
        
        split_total_frames = 0
        split_total_clips = 0
        
        for cf in class_folders:
            frames = data[split].get(cf, 0)
            clips = clips_count[split].get(cf, 0)
            avg = frames / clips if clips > 0 else 0
            
            print(f"   {class_names[cf]:<25} {frames:>12,} {clips:>10,} {avg:>12.1f}")
            split_total_frames += frames
            split_total_clips += clips
        
        print(f"   {'-'*60}")
        overall_avg = split_total_frames / split_total_clips if split_total_clips > 0 else 0
        print(f"   {'TOTAL':<25} {split_total_frames:>12,} {split_total_clips:>10,} {overall_avg:>12.1f}")
        
        # Calculate imbalance ratio
        counts = [data[split].get(cf, 0) for cf in class_folders]
        if min(counts) > 0:
            max_ratio = max(counts) / min(counts)
            print(f"   Imbalance Ratio (max/min): {max_ratio:.2f}x")
        print()
    
    print(f"{'='*70}\n")

def main():
    print(f"\n🔍 Analyzing extracted frames from: {EXTRACTION_ROOT}\n")
    
    if not os.path.exists(EXTRACTION_ROOT):
        print(f"❌ Error: Directory '{EXTRACTION_ROOT}' not found!")
        print(f"   Please update EXTRACTION_ROOT variable to match your extraction output.")
        return
    
    # Count frames
    print("Counting frames...")
    data, clips_count = count_frames_per_class(EXTRACTION_ROOT)
    
    if not data:
        print("❌ No data found! Check if extraction completed successfully.")
        return
    
    # Print statistics
    print_statistics(data, clips_count)
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    plot_frame_distribution(data, "1_frame_distribution.png")
    plot_frames_per_clip(data, clips_count, "2_frames_per_clip.png")
    plot_class_balance_ratio(data, "3_class_balance_ratio.png")
    plot_combined_summary(data, clips_count, "4_combined_summary.png")
    
    print("\n✅ All visualizations completed!")
    print("\nGenerated files:")
    print("  1. 1_frame_distribution.png - Total frames per class")
    print("  2. 2_frames_per_clip.png - Average frames extracted per video")
    print("  3. 3_class_balance_ratio.png - Balance ratio between classes")
    print("  4. 4_combined_summary.png - Comprehensive overview")
    print()

if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    main()