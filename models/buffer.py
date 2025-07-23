import json
import os
import numpy as np
from datetime import datetime, timedelta


class ColdMemoryBuffer:
    def __init__(self, buffer_path="memory/cold_buffer.json", ttl_seconds=120):
        self.buffer_path = buffer_path
        self.ttl = timedelta(seconds=ttl_seconds)
        self._load()

    def _load(self):
        if os.path.exists(self.buffer_path):
            with open(self.buffer_path, "r") as f:
                try:
                    self.buffer = json.load(f)
                except json.JSONDecodeError:
                    self.buffer = []
        else:
            self.buffer = []

    def _save(self):
        with open(self.buffer_path, "w") as f:
            json.dump(self.buffer, f, indent=2)

    def store_failed_cycle(self, input_vector, output, reward):
        timestamp = datetime.utcnow().isoformat()
        item = {
            "input": input_vector.tolist(),
            "output": output.tolist(),
            "reward": reward,
            "timestamp": timestamp
        }
        self.buffer.append(item)
        self._save()
        print(f"❄️ Cold memory stored. Reward = {reward}")

    def prune_expired(self):
        now = datetime.utcnow()
        original_len = len(self.buffer)

        self.buffer = [
            item for item in self.buffer
            if now - datetime.fromisoformat(item["timestamp"]) < self.ttl
        ]

        if len(self.buffer) < original_len:
            print(f"🧹 Pruned {original_len - len(self.buffer)} expired memories.")
            self._save()

    def show_buffer(self):
        print("\n📦 Cold Memory Buffer:")
        for i, item in enumerate(self.buffer):
            print(f"  #{i+1}: reward={item['reward']}, input={item['input']}")

    def get_buffer(self):
        return self.buffer

    def find_similar_input(self, current_input, similarity_threshold=0.9):
        """
        Search cold buffer for similar input vectors using cosine similarity.
        """
        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

        matches = []
        for item in self.buffer:
            similarity = cosine_similarity(item["input"], current_input.tolist())
            if similarity >= similarity_threshold:
                matches.append((similarity, item))

        matches.sort(reverse=True)
        return matches[:3]  # Top 3 similar items

