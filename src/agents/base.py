import abc
import multiprocessing
import sys
from typing import List, Set, Tuple

import torch

from environments.trick_taking_game import TrickTakingGame
from util import Card


class ExecutorManager:
    executor = None
    num_threads = None

    @staticmethod
    def get_executor(use_thread):
        if ExecutorManager.executor is None:
            is_linux = sys.platform == "linux" or sys.platform == "linux2"
            use_thread = use_thread if use_thread is not None else is_linux
            if use_thread:
                torch.multiprocessing.set_start_method('spawn')  # allow CUDA in multiprocessing

                num_cpus = multiprocessing.cpu_count()
                ExecutorManager.num_threads = int(num_cpus / 2)  # can use more or less CPUs
                ExecutorManager.executor = multiprocessing.Pool(ExecutorManager.num_threads)

        return ExecutorManager.num_threads, ExecutorManager.executor


class Agent:
    """Abstract base class for an AI agent that plays a trick taking game."""

    def __init__(self, game: TrickTakingGame, player_number: int, executor: bool = None):
        self._game = game
        self._player = player_number

    @abc.abstractmethod
    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        """
        Handle an observation from the environment, and update any personal records/current
        belief/etc.

        :param action: tuple of the player who moved and the index of the card they played
        :param observation: the observation corresponding to this player as returned by the env
        :param reward: an integral reward corresponding to this player as returned by the env
        :return: None
        """
        pass

    @abc.abstractmethod
    def act(self, epsilon: float = 0) -> int:
        """
        Based on the current observation/belief/known state, select a Card to play.
        :return: the card to play
        """
        pass

    def _get_hand(self, observation: List[int], valid_only: bool = False) -> Set[Card]:
        """
        Get the hand of an agent based on an observation.
        :param observation: observation corresponding to this player as returned by the env
        :param valid_only: True if only valid card plays should be returned, False if entire hand
        should be returned
        :return: the set of cards in the player's hand
        """
        cards = observation[:self._game.num_cards]
        return set(self._game.index_to_card(i) for i, in_hand in enumerate(cards)
                   if in_hand and (not valid_only or self._game.is_valid_play(self._player, i)))


class Learner:
    """Abstract base class for an AI that learns to play trick taking games."""

    @abc.abstractmethod
    def __init__(self, use_thread: bool = None):
        self._use_thread = use_thread
        self.num_threads, self.executor = ExecutorManager.get_executor(use_thread)
