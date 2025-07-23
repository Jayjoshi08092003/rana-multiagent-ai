# utils/feedback_simulator.py

import random

class FeedbackSimulator:
    """
    Simulates reward/stress feedback for the agent.
    In the real system, this would come from user response,
    sensors, vision, etc.
    """

    def __init__(self, mode="random", fail_threshold=0.3):
        self.mode = mode
        self.fail_threshold = fail_threshold

    def get_reward(self, output_tensor):
        """
        Return a pseudo-reward based on mode:
        - 'random': randomly between 0 and 1
        - 'fail-heavy': mostly low values (simulate tough input)
        - 'high': mostly good feedback
        """
        if self.mode == "random":
            return round(random.uniform(0.0, 1.0), 2)

        elif self.mode == "fail-heavy":
            return round(random.uniform(0.0, self.fail_threshold), 2)

        elif self.mode == "high":
            return round(random.uniform(0.6, 1.0), 2)

        else:
            return 0.0
