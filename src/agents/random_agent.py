"""Hardcoded Learner-Agent pair for an AI that always selects a random valid card in its hand to play."""

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
        valid_cards = self._get_hand(self._current_observation, valid_only=True)
        return random.sample(valid_cards, 1)[0]


class RandomLearner(Learner):
    def train(self, tasks: List[TrickTakingGame.__class__]):
        pass

    def initialize_agent(self, game: TrickTakingGame, player_number: int) -> Agent:
        return RandomAgent(game, player_number)
