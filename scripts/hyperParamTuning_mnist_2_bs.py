
# Libraries
import sys
import os
import numpy as np
import csv 
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

# === Config ===
hidden_layer_sizes = [784, 30, 10]
eta = 0.1
batch_sizes = [10, 20, 50, 100]
lambda_reg = 0.0
epochs = 30
early_stopping_n = 10

# === Load data ===
training_data, validation_data, test_data = mnist_load_data_wrapper()

# === Result storage ===
summary_results = []
detailed_results = []

print("\nStarting mini-batch size sweep...\n")

for batch_size in batch_sizes:
    print(f"Training with batch size = {batch_size}")
    net = Network(sizes=hidden_layer_sizes, cost=CrossEntropyCost)
    history = net.SGD(training_data = training_data, 
                      epochs = epochs, 
                      mini_batch_size = batch_size, 
                      eta = eta, 
                      lmbda = lambda_reg, 
                      evaluation_data = validation_data, 
                      early_stopping_n = early_stopping_n,
                      monitor_evaluation_accuracy=True,
                      monitor_evaluation_cost=True,
                      monitor_training_accuracy=True,
                      monitor_training_cost=True,
                      verbose=False)
    final_eval_acc = history["evaluation_accuracy"][-1]
    final_eval_cost = history["evaluation_cost"][-1]
    summary_results.append((batch_size, final_eval_acc, final_eval_cost))
    detailed_results.append({
        "batch_size": batch_size,
        "evaluation_accuracy": history["evaluation_accuracy"],
        "evaluation_cost": history["evaluation_cost"],
        "training_accuracy": history["training_accuracy"],
        "training_cost": history["training_cost"]
    })
    print(f"batch size {batch_size} finished: Eval Accuracy = {final_eval_acc}, Eval Cost = {final_eval_cost}\n")

# === Print Summary ===
print("\nLearning Rate Sweep Results:")
print("Batch_size\tAccuracy\tCost")
for batch_size, final_eval_acc, final_eval_cost in summary_results:
        print(f"{batch_size}\t{final_eval_acc}\t\t{final_eval_cost:.4f}")

# === Save results to CSV ===
csv_path = os.path.join(project_root, 
                        "results", 
                        f"{os.path.basename(__file__).split('.')[0]}_summary.csv")
save_results_to_csv(summary_results, 
                    header = ["batch_size", "eval_accuracy", "eval_cost"], 
                    path = csv_path)

# === Save detailed results to JSON ===
json_path = os.path.join(project_root, 
                         "results", 
                         f"{os.path.basename(__file__).split('.')[0]}_detailed.json")

with open(json_path, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"Detailed results saved to: {json_path}")