import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.base_agent import RANABaseAgent
from models.buffer import ColdMemoryBuffer


def show_activation_heatmap(agent: RANABaseAgent):
    knowledge = agent.get_knowledge_map()
    for state, activations in knowledge.items():
        data = np.array(activations).reshape(1, -1)
        st.write(f"🔬 **{state.upper()} Activation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 1))
        sns.heatmap(data, cmap="viridis", ax=ax, cbar=True, xticklabels=True, yticklabels=[state])
        st.pyplot(fig)


def show_buffer_stats(buffer: ColdMemoryBuffer):
    buffer_data = buffer.get_buffer()
    st.write(f"🧊 **Cold Buffer Memory** — {len(buffer_data)} items")
    for i, item in enumerate(buffer_data[-5:]):
        st.json({
            "Input": item["input"],
            "Reward": item["reward"],
            "Timestamp": item["timestamp"]
        })


def show_agent_stats(agent: RANABaseAgent):
    st.metric("Input Neurons", agent.input_size)
    st.metric("Hidden Neurons", agent.hidden_size)
    st.metric("Output Neurons", agent.output_size)
    st.metric("Last Reward", round(agent.last_reward or 0.0, 3))


def run_dashboard(agents_dict):
    st.title("🧠 RANA Multi-Agent Dashboard")

    for state, components in agents_dict.items():
        with st.expander(f"🔎 {state.upper()} Agent"):
            agent = components["controller"].get_agent()
            buffer = components["buffer"]

            show_agent_stats(agent)
            show_activation_heatmap(agent)
            show_buffer_stats(buffer)
