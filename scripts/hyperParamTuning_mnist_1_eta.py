# Libraries 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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

# Training hyperparameters  
hidden_layer_sizes = [784, 30, 10]
etas = [0.5, 0.1, 0.05, 0.01]
mini_batch_size = 20
lambda_reg = 0.0
epochs = 30
early_stopping_n = 10

# === Result storage ===
summary_results = []
detailed_results = []

print("\nStarting learning rate sweep...\n")

for eta in etas:
    print(f"Training with η = {eta}")
    net = Network(sizes=hidden_layer_sizes, cost=CrossEntropyCost)
    history = net.SGD(
        training_data=training_data,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        eta=eta,
        lmbda=lambda_reg,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True,
        early_stopping_n=early_stopping_n,
        verbose=False  # Set True if you want detailed logs
    )
    
    final_eval_acc = history["evaluation_accuracy"][-1]
    final_eval_cost = history["evaluation_cost"][-1]
    summary_results.append((eta, final_eval_acc, final_eval_cost))
    detailed_results.append({
        "eta": eta,
        "evaluation_accuracy": history["evaluation_accuracy"],
        "evaluation_cost": history["evaluation_cost"],
        "training_accuracy": history["training_accuracy"],
        "training_cost": history["training_cost"]
    })
    print(f"η = {eta} finished: Eval Accuracy = {final_eval_acc}, Eval Cost = {final_eval_cost}\n")

# === Print Summary ===
print("\nLearning Rate Sweep Results:")
print("Eta\tAccuracy\tCost")
for eta, final_eval_acc, final_eval_cost in summary_results:
    print(f"{eta:.4f}\t{final_eval_acc}\t\t{final_eval_cost:.4f}")

# === Save results to CSV ===
csv_path = os.path.join(project_root, 
                        "results", 
                        f"{os.path.basename(__file__).split('.')[0]}_summary.csv")
save_results_to_csv(summary_results, 
                    header = ["eta", "eval_accuracy", "eval_cost"], 
                    path = csv_path)
# === Save detailed results to JSON ===
json_path = os.path.join(project_root, 
                         "results", 
                         f"{os.path.basename(__file__).split('.')[0]}_detailed.json")

with open(json_path, "w") as f:
    json.dump(detailed_results, f, indent=2)
print(f"Detailed results saved to: {json_path}")
