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

# === Load data ===
training_data, validation_data, test_data = mnist_load_data_wrapper()

# === Config ===
eta = 0.1  # From previous tuning
batch_size = 20  # From previous tuning
architecture = [784, 30, 10]
epochs = 30
early_stopping_n = 10

# === Grid: Regularization Type & Lambda ===
regularizers = ['none', 'l1', 'l2']  # Include 'none' as an option
lambda_grid = {
    'none': [0.0],     # Only λ=0.0 makes sense for no regularization
    'l1': [0.01, 0.1, 1.0],
    'l2': [0.01, 0.1, 1.0]
}


# Results will be collected here
summary_results = []
detailed_results = []

print("\nStarting regularization sweep...\n")

regularizers = ['None', 'l1', 'l2']
lambda_grid = {
    'None': [0.0],
    'l1': [0.01, 0.1, 1.0],
    'l2': [0.01, 0.1, 1.0]
}

for reg_type in regularizers:
    for lmbda in lambda_grid[reg_type]:
        print(f"Training with {reg_type} regularization (λ = {lmbda})")

        # Initialize the network
        net = Network(sizes=architecture, cost=CrossEntropyCost)

        # Train the network
        history = net.SGD(
            training_data=training_data,
            epochs=epochs,
            mini_batch_size=batch_size,
            eta=eta,
            lmbda=lmbda,
            regularization_type=reg_type,
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
        summary_results.append((reg_type, lmbda, final_eval_acc, final_eval_cost))
        detailed_results.append({
            "regularization_type": reg_type,
            "lambda": lmbda,
            "evaluation_accuracy": history["evaluation_accuracy"],
            "evaluation_cost": history["evaluation_cost"],
            "training_accuracy": history["training_accuracy"],
            "training_cost": history["training_cost"]
        })
        print(f"{reg_type.upper()} (λ={lmbda}): Accuracy={final_eval_acc}, Cost={final_eval_cost:.4f}\n")

        
# === Summary Table ===
print("\nRegularization Sweep Results:")
print(f"{'Type':<8} {'Lambda':<8} {'Accuracy':<10} {'Eval Cost'}")
for reg, lmbda, final_eval_acc, final_eval_cost in summary_results:
    print(f"{reg:<8} {lmbda:<8.1e} {final_eval_acc:<10} {final_eval_cost:.4f}")

# === Save results to CSV ===
csv_path = os.path.join(project_root, 
                        "results", 
                        f"{os.path.basename(__file__).split('.')[0]}_summary.csv")
save_results_to_csv(summary_results, 
                    header = ["regularization_type", "lambda", "eval_accuracy", "eval_cost"], 
                    path = csv_path)

# === Save detailed results to JSON ===
json_path = os.path.join(project_root, 
                         "results", 
                         f"{os.path.basename(__file__).split('.')[0]}_detailed.json")
with open(json_path, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"Detailed results saved to: {json_path}")