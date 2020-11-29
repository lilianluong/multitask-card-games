# Created by Patrick Kao
import copy
from enum import Enum
from typing import List, Tuple, Dict

from environments.trick_taking_game import TrickTakingGame
from util import Suit, OutOfTurnException


class OhHellPhase(Enum):
    BIDDING = 0
    PLAY = 1


class OhHell(TrickTakingGame):
    """
    Oh Hell environment

    The first action the player gives

    Modifications:
    - trump suit is always spades
    - play is with a fixed # of cards instead of varying between rounds
    """
    name = "Oh Hell"

    def __init__(self):
        super().__init__()
        self._player_bids = None
        self._phase = None
        self._tricks_won = None

    def reset(self, state: List[int] = None) -> Tuple[List[int], ...]:
        self._phase = OhHellPhase.BIDDING
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

    # TODO: randomize trump suit at beginning of each game
    def _get_trump_suit(self) -> int:
        return Suit.SPADES

    def _end_trick(self) -> Tuple[List[int], int]:
        winning_player, winning_card = self._get_trick_winner()
        rewards = [0 for _ in range(self.num_players)]
        # TODO: reward intermediate as getting closer
        self._tricks_won[winning_player] += 1

        return rewards, winning_player

    def _end_game_bonuses(self) -> List[int]:
        scores = [0 for _ in range(self.num_players)]
        for player in range(self.num_players):
            scores[player] += self._tricks_won[player]
            if self._tricks_won[player] == self._player_bids[player]:
                scores[player] += 10

        return scores

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
        if self._phase == OhHellPhase.BIDDING:
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
                num_tricks = self.num_cards / self.num_players

                # invalid
                if sum(new_player_bids) == num_tricks:
                    rewards[player] = -100
                    invalid_plays[player] = "invalid"
                    new_player_bids[player] += 1

                self._phase = OhHellPhase.PLAY

            self._player_bids = new_player_bids

            self._state[-1] = (player + 1) % num_players
            return self._get_observations(), tuple(rewards), False, invalid_plays
        else:
            return super().step(action)

    @property
    # TODO: deal more or less cards depending  on round
    def cards_per_suit(self) -> Tuple[int, ...]:
        return 8, 8, 8, 8
