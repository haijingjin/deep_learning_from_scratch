"""
Common utilities for neural network training and evaluation.
"""

import os
import sys
import csv
from datetime import datetime
import matplotlib.pyplot as plt 

# Setup paths, logging, and results

def setup_logging(script_path):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(script_path))
    
    # Create a "logs" folder under the same script directory
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(script_path).split(".")[0]
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout
    
    return log_file


def save_results_to_csv(results, header, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
    print(f"Results saved to: {path}")



def plot_train_eval_history(json_history, title=None, show=True, save_path=None):
    """
    Plot training/evaluation metrics (cost + accuracy) with color distinction.
    - Training: Blue
    - Evaluation: Red
    """
    # Check required keys
    required_keys = ["training_cost", "evaluation_cost", "training_accuracy", "evaluation_accuracy"]
    if not all(key in json_history for key in required_keys):
        raise ValueError(f"History must contain keys: {required_keys}")

    if len(json_history["training_cost"]) == 0:
        raise ValueError("History is empty. No data to plot.")

    epochs = range(1, len(json_history["training_cost"]) + 1)

    plt.figure(figsize=(12, 5))

    # === Cost Plot (Blue: Training, Red: Evaluation) ===
    plt.subplot(1, 2, 1)
    plt.plot(epochs, json_history["training_cost"], 'b-', label='Train Cost')  # Solid blue
    plt.plot(epochs, json_history["evaluation_cost"], 'C1-', label='Eval Cost')  # Solid red
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Cost Over Epochs")
    plt.legend()

    # === Accuracy Plot (Blue: Training, Red: Evaluation) ===
    plt.subplot(1, 2, 2)
    plt.plot(epochs, json_history["training_accuracy"], 'b-', label='Train Accuracy')  # Solid blue
    plt.plot(epochs, json_history["evaluation_accuracy"], 'C1-', label='Eval Accuracy')  # Solid red
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_eval_metrics_from_json(json_history, label_keys=None, show=True, save_path=None):
    """
    Plot evaluation cost and accuracy over epochs from a detailed tuning JSON file.
    
    Args:
        json_path (str): Path to the JSON file.
        label_keys (list or str): Keys to use in the label (e.g., ["reg_type", "lambda"]).
        show (bool): Whether to display the plot.
        save_path (str): If provided, save the plot to this path.
    """
    plt.figure(figsize=(12, 5))

    # === Plot Cost ===
    plt.subplot(1, 2, 1)
    for entry in json_history:
        label = make_label(entry, label_keys)
        epochs = range(1, len(entry["evaluation_cost"]) + 1)
        plt.plot(epochs, entry["evaluation_cost"], label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Evaluation Cost")
    plt.title("Evaluation Cost Over Epochs")
    plt.legend()

    # === Plot Accuracy ===
    plt.subplot(1, 2, 2)
    for entry in json_history:
        label = make_label(entry, label_keys)
        epochs = range(1, len(entry["evaluation_accuracy"]) + 1)
        plt.plot(epochs, entry["evaluation_accuracy"], label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Evaluation Accuracy")
    plt.title("Evaluation Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()


def make_label(entry, label_keys):
    if isinstance(label_keys, list):
        return ", ".join([f"{key}={entry[key]}" for key in label_keys])
    elif label_keys:
        return str(entry[label_keys])
    else:
        return "Run"
