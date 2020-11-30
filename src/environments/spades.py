# Created by Patrick Kao
from typing import Tuple, List

from environments.bidding_base import BiddingBase
from util import Suit


class Spades(BiddingBase):
    """
    Assumes that players 0 and 1 are teamed, and players 1 and 2 are teamed
    """
    name = "Spades"

    def __init__(self):
        super().__init__()

    @property
    def cards_per_suit(self) -> Tuple[int, ...]:
        return 8, 8, 8, 8

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
        # each team has players team*2, team*2+1
        for team in range(self.num_players // 2):
            p1 = team * 2
            p2 = team * 2 + 1
            team_tricks = self._tricks_won[p1] + self._tricks_won[p2]
            team_bid = self._player_bids[p1] + self._player_bids[p2]
            team_score = 0
            if team_tricks >= team_bid:
                team_score = 10 * team_bid + (team_tricks - team_bid)

            scores[p1] += team_score
            scores[p2] += team_score

        return scores
