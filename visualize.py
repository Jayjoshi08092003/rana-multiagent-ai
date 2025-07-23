# visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_knowledge_map(agent, title="Agent Activation Map"):
    """
    Plots a heatmap of average neuron activations for an agent.
    """
    knowledge = agent.get_knowledge_map()
    if not knowledge:
        print(f"⚠️ No knowledge found for {title}")
        return

    for state, activations in knowledge.items():
        data = np.array(activations).reshape(1, -1)  # shape (1, N neurons)
        plt.figure(figsize=(10, 1))
        sns.heatmap(data, cmap="viridis", cbar=True, xticklabels=True, yticklabels=[state])
        plt.title(f"{title} — State: {state}")
        plt.xlabel("Neuron Index")
        plt.tight_layout()
        plt.show()

