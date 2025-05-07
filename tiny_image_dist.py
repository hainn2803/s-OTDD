import argparse
import torch
import numpy as np
import os
import time
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from otdd.pytorch.method5 import compute_pairwise_distance
import gc


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(f"Using device: {DEVICE}")


def get_task_numbers(parent_dir):
    """Get sorted list of task numbers from data files with validation"""
    task_numbers = []
    try:
        for fname in os.listdir(parent_dir):
            if fname.startswith("data_task_") and fname.endswith("_size_10000.pt"):
                parts = fname.split("_")
                if len(parts) >= 3 and parts[2].isdigit():
                    task_num = int(parts[2])
                    task_numbers.append(task_num)
        return sorted(task_numbers)
    except FileNotFoundError:
        raise ValueError(f"Directory {parent_dir} not found")
    except Exception as e:
        raise RuntimeError(f"Error scanning directory: {str(e)}")


def load_task_data(task_num, parent_dir, sample_size=400, seed=42):
    """Load task data with balanced class sampling"""

    np.random.seed(seed)
    data_path = f'{parent_dir}/data/trainset_{task_num}.pt'

    # Load full dataset
    task_data, task_labels = torch.load(data_path)
    
    labels_np = task_labels.numpy()
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    num_classes = len(unique_labels)
    
    samples_per_class = sample_size // num_classes
    
    selected_indices = []
    
    # Stratified sampling with balanced classes
    for idx, label in enumerate(unique_labels):
        # Get indices for this class
        class_indices = np.where(labels_np == label)[0]
        
        selected = np.random.permutation(class_indices)[:samples_per_class]
        selected_indices.extend(selected)
    
    selected_indices = np.random.permutation(selected_indices)[:sample_size]
    
    return TensorDataset(
        task_data[selected_indices],
        task_labels[selected_indices]
    )
        

def compute_pairwise_distances(parent_dir, output_file, num_samples=1000, num_projections=10000, unique_tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Compute distances between source task and multiple target tasks"""
    total_start = time.time()
    time_metrics = {}
    
    # Phase 1: Data Preparation and Validation
    phase_start = time.time()
    list_task_datasets = list()
    for task in unique_tasks:
        task_dataset = load_task_data(task_num=task, parent_dir=parent_dir, sample_size=num_samples)
        list_task_datasets.append(DataLoader(task_dataset, batch_size=256, shuffle=True))
    time_metrics['data_loading'] = time.time() - phase_start

    # Phase 2: Distance Computation
    kwargs = {
        "dimension": 224,
        "num_channels": 3,
        "num_moments": 5,
        "use_conv": True,
        "precision": "float",
        "p": 2,
        "chunk": 1000
    }

    # Compute distance
    distance_matrix, processing_time = compute_pairwise_distance(
        list_D=list_task_datasets,
        num_projections=num_projections,
        device=DEVICE,
        evaluate_time=True,
        **kwargs
    )

    time_metrics['computation'] = processing_time # 600 secs

    os.makedirs(output_file, exist_ok=True)
    torch.save(distance_matrix, output_file + f"/sotdd_distance.pt")

    time_metrics['total'] = time.time() - total_start

    print(f"Results saved to {output_file}")
    print("\nTime Breakdown:")
    for phase, t in time_metrics.items():
        print(f"- {phase.capitalize()}: {t:.2f} seconds")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute task distances using source-target pairs')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Source task number')
    parser.add_argument('--num_projections', type=int, default=500000,
                       help='Source task number')
    parser.add_argument('--parent_dir', default="saved_split_task",
                       help='Parent directory with task data')
    
    args = parser.parse_args()

    args.output = args.parent_dir + "/dist_pairwise"

    print(f"Starting computation on {DEVICE}...")
    results = compute_pairwise_distances(
        parent_dir=args.parent_dir,
        output_file=args.output,
        num_samples=args.num_samples,
        num_projections=args.num_projections
    )