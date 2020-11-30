# Created by Patrick Kao
from typing import List, Tuple

from environments.bidding_base import BiddingBase
from util import Suit


class OhHell(BiddingBase):
    name = "Oh Hell"

    def _valid_bids(self, proposed_bids: List) -> bool:
        num_tricks = self.num_cards / self.num_players
        return sum(proposed_bids) != num_tricks

    @property
    # TODO: deal more or less cards depending  on round
    def cards_per_suit(self) -> Tuple[int, ...]:
        return 8, 8, 8, 8

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