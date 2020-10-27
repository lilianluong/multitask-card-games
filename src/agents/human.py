# Created by Patrick Kao
from typing import Tuple, List

from agents.base import Agent
from environments.trick_taking_game import TrickTakingGame
from util import Card


class Human(Agent):
    def __init__(self, game: TrickTakingGame, player_number: int):
        super().__init__(game, player_number)
        self._current_observation = None

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        self._current_observation = observation

    def act(self) -> Card:
        raise ValueError("Human agents can't act")
