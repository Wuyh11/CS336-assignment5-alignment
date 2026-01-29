import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(result_dir):
    metrics_path = os.path.join(result_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}")
        return
        
    with open(metrics_path, "r") as f:
        data = json.load(f)
        
    steps = data["steps"]
    acc = data["accuracy"]
    entropy = data["entropy"]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('EI Steps')
    ax1.set_ylabel('Validation Accuracy', color=color)
    ax1.plot(steps, acc, marker='o', color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Entropy on secondary axis
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Mean Training Entropy', color=color)
    # Entropy is recorded during training steps (step 1..N), step 0 has 0.0 placeholder
    ax2.plot(steps[1:], entropy[1:], marker='x', linestyle='--', color=color, label='Entropy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Expert Iteration: Accuracy & Entropy over Steps')
    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, "ei_metrics_plot.png"))
    print(f"Plot saved to {os.path.join(result_dir, 'ei_metrics_plot.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()
    
    plot_metrics(args.result_dir)