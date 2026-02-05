# src/gfn/reward.py
import math


def terminal_reward(best_score: float, beta: float = 5.0) -> float:
    """
    Terminal reward for a trajectory.
    Uses the best score achieved, not necessarily final score.
    """
    return math.exp(beta * best_score)


def dense_reward(prev_score: float, curr_score: float, cap: float = 0.2) -> float:
    """
    Optional dense shaping reward (can be disabled).
    """
    return max(min(curr_score - prev_score, cap), -cap)
