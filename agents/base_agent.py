"""
agents/base_agent.py
─────────────────────
Abstract base class defining the interface every agent must implement.

WHY an abstract base class?
  - ReinforceAgent, PPOAgent, MAPPOAgent, LLMAgent all share the same
    interaction pattern: select_action → store_transition → update
  - Defining this contract here means the training loop (train_reinforce.py,
    train_ppo.py) can be written against BaseAgent — swap the agent,
    the loop doesn't change.
  - Any agent that forgets to implement a required method gets an error
    at CLASS DEFINITION time, not at runtime during training.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class BaseAgent(ABC):

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Given observation, return action. Called before env.step()."""
        ...

    @abstractmethod
    def store_transition(self, action: np.ndarray, reward: float) -> None:
        """Record outcome. Called after env.step() returns reward."""
        ...

    @abstractmethod
    def update(self) -> Dict:
        """Improve policy using stored experience. Returns metrics dict."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent weights to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent weights from disk."""
        ...

    @abstractmethod
    def get_training_log(self) -> List[Dict]:
        """Return full history of per-episode training metrics."""
        ...