import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict
import numpy as np


class RANABaseAgent(nn.Module):
    def __init__(self, input_size=10, hidden_size=16, output_size=1, state_name="default"):
        super(RANABaseAgent, self).__init__()

        self.state_name = state_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.last_output = None
        self.last_reward = None
        self.activation_log = []

    def forward(self, x):
        hidden = F.relu(self.fc1(x))

        # Track neuron activations for knowledge labeling
        activations = hidden.detach().cpu().numpy().tolist()
        self.activation_log.append({
            "state": self.state_name,
            "activations": activations,
            "timestamp": datetime.utcnow().isoformat()
        })

        out = self.fc2(hidden)
        self.last_output = out
        return out

    def receive_feedback(self, reward):
        self.last_reward = reward
        print(f"[{self.state_name}] Reward received: {reward}")

    def summary(self):
        print(f"Agent: {self.state_name}")
        print(f"Input: {self.input_size}, Hidden: {self.hidden_size}, Output: {self.output_size}")
        print(f"Last Reward: {self.last_reward}")

    def get_knowledge_map(self):
        """
        Aggregate activations by knowledge state.
        Returns dictionary: {state_name: [avg_activation_per_neuron]}
        """
        state_map = defaultdict(list)

        for entry in self.activation_log:
            state_map[entry["state"]].append(entry["activations"])

        knowledge_map = {}
        for state, activations in state_map.items():
            avg = np.mean(activations, axis=0).tolist()
            knowledge_map[state] = avg

        return knowledge_map

    def prune_low_activity_neurons(self, activity_threshold=0.05):
        """
        Prune neurons with average activation below the threshold.
        Returns: Number of neurons pruned
        """
        knowledge_map = self.get_knowledge_map()
        avg_activations = knowledge_map.get(self.state_name, [])
        if not avg_activations:
            print("⚠️ No activation data available. Skipping pruning.")
            return 0

        keep_indices = [i for i, act in enumerate(avg_activations) if act >= activity_threshold]
        if len(keep_indices) == len(avg_activations):
            print("✅ All neurons active enough. No pruning needed.")
            return 0

        print(f"✂️ Pruning {len(avg_activations) - len(keep_indices)} low-activity neurons...")

        new_hidden_size = len(keep_indices)
        new_agent = RANABaseAgent(
            input_size=self.input_size,
            hidden_size=new_hidden_size,
            output_size=self.output_size,
            state_name=self.state_name
        )

        with torch.no_grad():
            new_agent.fc1.weight.copy_(self.fc1.weight[keep_indices])
            new_agent.fc1.bias.copy_(self.fc1.bias[keep_indices])
            new_agent.fc2.weight.copy_(self.fc2.weight[:, keep_indices])
            new_agent.fc2.bias.copy_(self.fc2.bias)

        new_agent.activation_log = self.activation_log
        new_agent.last_reward = self.last_reward
        new_agent.last_output = self.last_output

        self.__dict__.update(new_agent.__dict__)
        self.fc1 = new_agent.fc1
        self.fc2 = new_agent.fc2
        self.hidden_size = new_hidden_size

        return len(avg_activations) - len(keep_indices)
