
import numpy as np
import csv
import os
import sys
import json
from datetime import datetime

# Add local src/ folder to path
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(project_root, "src"))

# Import self-defined modules
from extended_nn import *
from data_loader import * 
from common_utils import *

# Save all stdout + stderr
log_file = setup_logging(__file__)
print(f"Logging to: {log_file}")

# Load the MNIST data
training_data, validation_data, test_data = mnist_load_data_wrapper()

# Training hyperparameters  
hidden_layer_sizes = [784, 128, 128, 10] # From previous tuning
eta = 0.1 # From previous tuning
batch_size = 20 # From previous tuning
lmbda = 1.0 # From previous tuning
regularization_type = 'l2' # From previous tuning
epochs = 30
early_stopping_n = 10 

# === Try both with and without LR scheduling ===
lr_schedule_options = [False, True]

# === Save results ===
summary_results = []
detailed_results = []
model_dir = os.path.join(project_root, "saved_models")
os.makedirs(model_dir, exist_ok=True)

for use_lr_schedule in lr_schedule_options:
    label = "with_lr_schedule" if use_lr_schedule else "no_lr_schedule"
    print(f"\n>>> Training with LR scheduling: {use_lr_schedule}")

    net = Network(sizes=hidden_layer_sizes, cost=CrossEntropyCost)

    history = net.SGD(
        training_data=training_data,
        epochs=epochs,
        mini_batch_size=batch_size,
        eta=eta,
        lmbda=lmbda,
        regularization_type=regularization_type,
        evaluation_data=validation_data,
        test_data=test_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True,
        early_stopping_n=early_stopping_n,
        use_lr_schedule=use_lr_schedule,
        verbose=False
    )

    final_eval_acc = history["evaluation_accuracy"][-1]
    final_eval_cost = history["evaluation_cost"][-1]
    final_test_acc = history["test_accuracy"]
    final_test_cost = history["test_cost"]

    # Save model
    model_path = os.path.join(model_dir, f"model_{label}.json")
    net.save(model_path)
    print(f"Model saved to: {model_path}")

    # Collect summary
    summary_results.append((label, final_eval_acc, final_eval_cost, final_test_acc, final_test_cost))

    # Collect detailed
    history_entry = history.copy()
    history_entry.update({
        "label": label,
        "architecture": hidden_layer_sizes,
        "eta": eta,
        "batch_size": batch_size,
        "lmbda": lmbda,
        "regularization_type": regularization_type,
        "use_lr_schedule": use_lr_schedule
    })
    detailed_results.append(history_entry)
    print(f"LR Schedule={use_lr_schedule}: Eval Accuracy={final_eval_acc:.4f}, Eval Cost={final_eval_cost:.4f}, Test Accuracy={final_test_acc:.4f}\n")

# === Print Summary Table ===
print("\nFinal Model Results:")
print(f"{'Setting':<20} {'Eval Acc':<10} {'Eval Cost':<10} {'Test Acc':<10} {'Test Cost'}")
for label, final_eval_acc, final_eval_cost, final_test_acc, final_test_cost in summary_results:
    print(f"{label:<20} {final_eval_acc:<10} {final_eval_cost:<10.4f} {final_test_acc:<10} {final_test_cost:.4f}")

# === Save summary CSV ===
csv_path = os.path.join(project_root, "results", f"{os.path.basename(__file__).split('.')[0]}_summary.csv")
save_results_to_csv(summary_results, 
                    header=["setting", "eval_accuracy", "eval_cost", "test_accuracy", "test_cost"],
                    path=csv_path)

# === Save detailed JSON ===
json_path = os.path.join(project_root, "results", f"{os.path.basename(__file__).split('.')[0]}_detailed.json")
with open(json_path, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"Detailed results saved to: {json_path}")
