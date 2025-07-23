# models/expansion_controller.py

import torch
import torch.nn as nn
from models.base_agent import RANABaseAgent


class ExpansionController:
    def __init__(self, agent: RANABaseAgent, reward_threshold=0.3, max_expansions=3):
        self.agent = agent
        self.reward_threshold = reward_threshold
        self.max_expansions = max_expansions
        self.expansion_count = 0

    def should_expand(self):
        """Check if the agent's last reward was too low."""
        return (
            self.agent.last_reward is not None
            and self.agent.last_reward < self.reward_threshold
            and self.expansion_count < self.max_expansions
        )

    def expand_agent(self):
        """
        Expands the agent by adding hidden neurons.
        (Later: Can also add layers or memory heads.)
        """
        if not self.should_expand():
            return False

        print(f"[{self.agent.state_name}] Expansion triggered. Expanding agent...")

        # Step 1: Get current parameters
        input_size = self.agent.input_size
        old_hidden_size = self.agent.hidden_size
        new_hidden_size = old_hidden_size + 4  # Add 4 neurons
        output_size = self.agent.output_size

        # Step 2: Create new model with larger hidden layer
        new_agent = RANABaseAgent(
            input_size=input_size,
            hidden_size=new_hidden_size,
            output_size=output_size,
            state_name=self.agent.state_name,
        )

        # Step 3: Copy old weights into new model (partial copy)
        with torch.no_grad():
            new_agent.fc1.weight[:old_hidden_size] = self.agent.fc1.weight
            new_agent.fc1.bias[:old_hidden_size] = self.agent.fc1.bias

        # Step 4: Replace the output layer to match new hidden size
        new_agent.fc2 = nn.Linear(new_hidden_size, output_size)

        # Step 5: Replace the agent instance
        self.agent = new_agent
        self.expansion_count += 1
        print(f"[{self.agent.state_name}] Agent expanded to {new_hidden_size} hidden neurons.")

        return True

    def get_agent(self):
        return self.agent

