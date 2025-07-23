import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from models.base_agent import RANABaseAgent
from models.expansion_controller import ExpansionController
from models.buffer import ColdMemoryBuffer
from utils.feedback_simulator import FeedbackSimulator
from visualize import plot_knowledge_map
from dashboard import run_dashboard  # ✅ Add this

 # ✅ Make sure visualize.py exists


def generate_dummy_input(input_size):
    return torch.tensor(np.random.rand(input_size), dtype=torch.float32)


def create_agent(state_name, input_size, hidden_size, output_size, reward_threshold=0.4):
    agent = RANABaseAgent(input_size, hidden_size, output_size, state_name=state_name)
    controller = ExpansionController(agent, reward_threshold=reward_threshold)
    buffer = ColdMemoryBuffer(ttl_seconds=300)
    return {"agent": agent, "controller": controller, "buffer": buffer}


def main():
    # Config
    input_size = 10
    output_size = 1
    initial_hidden_size = 8
    max_cycles = 10
    reward_threshold = 0.4

    # Define multi-state agents
    states = ["self_healing", "emotion", "stress", "immunity"]
    agents = {
        state: create_agent(state, input_size, initial_hidden_size, output_size, reward_threshold)
        for state in states
    }

    feedback_engine = FeedbackSimulator(mode="fail-heavy")

    print("\n🧠 Multi-Agent RANA Simulation Start\n")

    for cycle in range(max_cycles):
        print(f"\n🔄 Cycle {cycle+1}")

        input_data = generate_dummy_input(input_size)

        for state, components in agents.items():
            print(f"\n🔎 State: {state}")

            agent = components["controller"].get_agent()
            controller = components["controller"]
            buffer = components["buffer"]

            # Reactivate from cold memory
            similar = buffer.find_similar_input(input_data, similarity_threshold=0.93)
            if similar:
                print(f"🔁 {len(similar)} similar memories found:")
                for sim, item in similar:
                    print(f"   - Similarity {round(sim, 2)} | Reward: {item['reward']}")

            # Run agent
            output = agent(input_data)
            reward = feedback_engine.get_reward(output)
            agent.receive_feedback(reward)

            # Cold memory if poor
            if reward < reward_threshold:
                buffer.store_failed_cycle(input_data, output, reward)

            # Expand if needed
            controller.expand_agent()

            # Prune expired memory
            buffer.prune_expired()

            # Show agent stats
            agent.summary()

            # ✅ Visualize neuron heatmap (per agent per cycle)
            plot_knowledge_map(agent, title=f"{state} Agent (Cycle {cycle+1})")

    # Final Summary
    print("\n📦 Final Knowledge Mapping:")

    for state, components in agents.items():
        agent = components["controller"].get_agent()
        knowledge = agent.get_knowledge_map()
        activations = knowledge.get(state, [])
        print(f"  [{state}] → {len(activations)} neurons | Snapshot: {list(map(lambda x: round(x, 3), activations))}")

        # Run pruning
        pruned = agent.prune_low_activity_neurons()
        print(f"  ✂️  {pruned} neurons pruned from '{state}' agent.\n")

    print("\n✅ Multi-Agent RANA Simulation Complete\n")
    # Launch dashboard
    run_dashboard(agents)



if __name__ == "__main__":
    main()
