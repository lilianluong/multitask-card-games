"""Hardcoded Learner-Agent pair for an AI that always selects a random card in its hand to play."""

import random
from typing import List, Tuple

from agents.base import Learner, Agent
from environments.trick_taking_game import TrickTakingGame
from util import Card


class RandomAgent(Agent):
    def __init__(self, game: TrickTakingGame, player_number: int):
        super(RandomAgent).__init__(game, player_number)
        self._current_observation = None

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        self._current_observation = observation

    def act(self) -> Card:
        current_hand = self._get_hand(self._current_observation)
        return random.sample(current_hand, 1)[0]


class RandomLearner(Learner):
    def train(self, tasks: List[TrickTakingGame.__class__]):
        pass

    def initialize_agent(self, game: TrickTakingGame, player_number: int) -> Agent:
        return RandomAgent(game, player_number)
