
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
architectures = [
    [784, 32, 10],
    [784, 64, 10],
    [784, 128, 10],
    [784, 64, 64, 10],
    [784, 128, 64, 10],
    [784, 128, 128, 10]
]

eta = 0.1 # From previous tuning
batch_size = 20
lmbda = 1.0
regularization_type = 'l2'
epochs = 30
early_stopping_n = 10 

# Results will be collected here
summary_results = []
detailed_results = []

for arch in architectures:
    print(f"Testing architecture: {arch}")
    net = Network(sizes=arch, cost=CrossEntropyCost)
    
    history = net.SGD(
        training_data=training_data,
        epochs=epochs,
        mini_batch_size=batch_size,
        eta=eta,
        lmbda=lmbda,
        regularization_type=regularization_type,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True,
        early_stopping_n=early_stopping_n,
        verbose=False
    )

    final_eval_acc = history["evaluation_accuracy"][-1]
    final_eval_cost = history["evaluation_cost"][-1]
    summary_results.append((arch, final_eval_acc, final_eval_cost))
    detailed_results.append({
        "architecture": arch,
        "evaluation_accuracy": history["evaluation_accuracy"],
        "evaluation_cost": history["evaluation_cost"],
        "training_accuracy": history["training_accuracy"],
        "training_cost": history["training_cost"]
    })
    print(f"Architecture: {arch} → Accuracy={final_eval_acc}, Cost={final_eval_cost:.4f}\n")

# === Summary Table ===
print("\nArchitecture Sweep Results:")
print(f"{'Architecture':<20} {'Accuracy':<10} {'Eval Cost'}")
for arch, final_eval_acc, final_eval_cost in summary_results:
    print(f"{str(arch):<20} {final_eval_acc:<10} {final_eval_cost:.4f}")

# === Save results to CSV ===
csv_path = os.path.join(project_root, 
                        "results", 
                        f"{os.path.basename(__file__).split('.')[0]}_summary.csv")
save_results_to_csv(summary_results, 
                    header = ["architecture", "eval_accuracy", "eval_cost"], 
                    path = csv_path)

# === Save detailed results to JSON ===
json_path = os.path.join(project_root, 
                         "results", 
                         f"{os.path.basename(__file__).split('.')[0]}_detailed.json")

with open(json_path, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"Detailed results saved to: {json_path}")