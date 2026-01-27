"""
Training Monitor - Real-time Training Progress Viewer
"""

import time
import os
from pathlib import Path


def print_header():
    """Print monitoring header"""
    print("=" * 80)
    print("FIRE AND SMOKE DETECTION - TRAINING MONITOR".center(80))
    print("=" * 80)
    print()


def get_latest_metrics():
    """Get latest training metrics from results.csv"""
    results_file = Path("artifacts/model_training/train/results.csv")
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            
            # Get last line (latest epoch)
            last_line = lines[-1].strip()
            values = last_line.split(',')
            
            # Parse metrics
            metrics = {
                'epoch': int(values[0].strip()),
                'train_box_loss': float(values[1].strip()),
                'train_cls_loss': float(values[2].strip()),
                'train_dfl_loss': float(values[3].strip()),
                'precision': float(values[4].strip()),
                'recall': float(values[5].strip()),
                'mAP50': float(values[6].strip()),
                'mAP50-95': float(values[7].strip()),
                'val_box_loss': float(values[8].strip()),
                'val_cls_loss': float(values[9].strip()),
                'val_dfl_loss': float(values[10].strip()),
            }
            
            return metrics
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return None


def display_metrics(metrics):
    """Display training metrics"""
    if metrics is None:
        print("Waiting for training to start...")
        return
    
    epoch = metrics['epoch']
    
    print(f"\n{'EPOCH ' + str(epoch):^80}")
    print("-" * 80)
    
    print("\nTRAINING LOSSES:")
    print(f"  Box Loss:   {metrics['train_box_loss']:.4f}")
    print(f"  Class Loss: {metrics['train_cls_loss']:.4f}")
    print(f"  DFL Loss:   {metrics['train_dfl_loss']:.4f}")
    
    print("\nVALIDATION METRICS:")
    print(f"  Precision:    {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:       {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f} ({metrics['mAP50']*100:.1f}%)")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f} ({metrics['mAP50-95']*100:.1f}%)")
    
    print("\nVALIDATION LOSSES:")
    print(f"  Box Loss:   {metrics['val_box_loss']:.4f}")
    print(f"  Class Loss: {metrics['val_cls_loss']:.4f}")
    print(f"  DFL Loss:   {metrics['val_dfl_loss']:.4f}")
    
    # Performance rating
    mAP = metrics['mAP50']
    if mAP >= 0.85:
        rating = "🟢 EXCELLENT"
    elif mAP >= 0.75:
        rating = "🟢 GOOD"
    elif mAP >= 0.65:
        rating = "🟡 FAIR"
    else:
        rating = "🔴 NEEDS IMPROVEMENT"
    
    print(f"\nPERFORMANCE: {rating}")
    print("-" * 80)


def monitor_training(refresh_interval=5):
    """
    Monitor training progress in real-time
    
    Args:
        refresh_interval: Seconds between updates
    """
    print_header()
    print(f"Monitoring training progress (updates every {refresh_interval}s)")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Clear screen (optional)
            # os.system('clear' if os.name == 'posix' else 'cls')
            
            # Get and display metrics
            metrics = get_latest_metrics()
            display_metrics(metrics)
            
            # Wait before next update
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Refresh interval in seconds (default: 5)'
    )
    
    args = parser.parse_args()
    monitor_training(refresh_interval=args.interval)
