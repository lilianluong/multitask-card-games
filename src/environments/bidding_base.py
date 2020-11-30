# Created by Patrick Kao
# Created by Patrick Kao
import copy
from enum import Enum
from typing import List, Tuple, Dict

from environments.trick_taking_game import TrickTakingGame
from util import Suit, OutOfTurnException


class Phase(Enum):
    BIDDING = 0
    PLAY = 1


class BiddingBase(TrickTakingGame):
    """
    Oh Hell environment

    The first action the player gives

    Modifications:
    - trump suit is always spades
    - play is with a fixed # of cards instead of varying between rounds
    """
    name = "Bidding ADT"

    def __init__(self):
        super().__init__()
        self._player_bids = None
        self._phase = None
        self._tricks_won = None

    def reset(self, state: List[int] = None) -> Tuple[List[int], ...]:
        self._phase = Phase.BIDDING
        self._player_bids = [None] * self.num_players
        self._tricks_won = [0] * self.num_players
        return super().reset(state)

    def _get_observations(self) -> Tuple[List[int], ...]:
        """
        Same as in superclass, but with player bids and game phase appended to end of observation

        :return: [super observation] + [player bids: int x players] + [game phase: int]
        """
        new_observations = super()._get_observations()
        for i in range(self.num_players):
            new_observations[i].extend(self._player_bids)
            new_observations[i].append(self._phase)

        return new_observations

    def step(self, action: Tuple[int, int]) -> Tuple[
        Tuple[List[int], ...], Tuple[int, ...], bool, Dict]:
        """
        Execute action according to the rules defined in the docstring of TrickTakingGame.

        :param action: Tuple[int, int], (id, i) representing player id playing card i
        :return: Tuple of the following:
            - observation, Tuple[List[int], ...] of observations
            - reward, Tuple[int, ...] of rewards
            - done, bool that is True if the game has ended, otherwise False
            - info, Dict of diagnostic information, currently empty
        """
        if self._phase == Phase.BIDDING:
            num_players = self.num_players
            rewards = [0 for _ in range(num_players)]
            player, bid = action
            invalid_plays = {}

            # check for invalid
            if player != self.next_player:
                return self._get_observations(), tuple(rewards), False, {
                    "error": OutOfTurnException}

            new_player_bids = copy.deepcopy(self._player_bids)
            new_player_bids[player] = bid
            if None not in new_player_bids:
                # invalid
                if not self._valid_bids(new_player_bids):
                    rewards[player] = -100
                    invalid_plays[player] = "invalid"
                    new_player_bids[player] += 1

                self._phase = Phase.PLAY

            self._player_bids = new_player_bids

            self._state[-1] = self._get_next_player(player)
            return self._get_observations(), tuple(rewards), False, invalid_plays
        else:
            return super().step(action)

    def _valid_bids(self, proposed_bids: List) -> bool:
        return True

    def _get_next_player(self, player):
        return (player + 1) % self.num_players