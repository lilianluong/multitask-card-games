import abc
from typing import List, Set, Tuple

from environments.trick_taking_game import TrickTakingGame
from util import Card


class Agent:
    """Abstract base class for an AI agent that plays a trick taking game."""

    def __init__(self, game: TrickTakingGame, player_number: int):
        self._game = game
        self._player = player_number

    @abc.abstractmethod
    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        """
        Handle an observation from the environment, and update any personal records/current belief/etc.

        :param action: tuple of the player who moved and the index of the card they played
        :param observation: the observation corresponding to this player as returned by the env
        :param reward: an integral reward corresponding to this player as returned by the env
        :return: None
        """
        pass

    @abc.abstractmethod
    def act(self) -> Card:
        """
        Based on the current observation/belief/known state, select a Card to play.
        :return: the card to play
        """
        pass

    def _get_hand(self, observation: List[int], valid_only: bool = False) -> Set[Card]:
        """
        Get the hand of an agent based on an observation.
        :param observation: observation corresponding to this player as returned by the env
        :param valid_only: True if only valid card plays should be returned, False if entire hand should be returned
        :return: the set of cards in the player's hand
        """
        cards = observation[:self._game.num_cards]
        return set(self._game.index_to_card(i) for i, in_hand in enumerate(cards)
                   if in_hand and (not valid_only or self._game.is_valid_play(self._player, i)))


class Learner:
    """Abstract base class for an AI that learns to play trick taking games."""

    @abc.abstractmethod
    def train(self, tasks: List[TrickTakingGame.__class__]):
        """
        Given a list of trick taking game environment classes, train on them.

        :param tasks: List of environment classes, which inherit from TrickTakingGame
        :return: None
        """
        pass

    @abc.abstractmethod
    def initialize_agent(self, game: TrickTakingGame, player_number: int) -> Agent:
        """
        Given an instance of a TrickTakingGame, return an Agent that will play it.

        :param game: instance of TrickTakingGame to play
        :param player_number: the index of the player in the game
        :return: an instance of Agent that will play game
        """
        pass
