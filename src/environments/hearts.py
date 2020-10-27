from typing import List, Tuple

from environments.trick_taking_game import TrickTakingGame
from util import Card, Suit


class SimpleHearts(TrickTakingGame):
    """
    Environment for the simplified game of Hearts, a 4-player no-trump trick taking game.

    The variation plays with 8 cards of each suit, for a total of 32.
    The player receives a negative point for each heart card they take, as well as -7 points for taking SPADES 5.
    If a player takes all hearts as well as SPADES 5, they "shoot the moon", which means that instead of getting
    -15 points, they receive 0 and everyone else takes -15 points.

    The rules for passing cards do not apply in this variant, for simplicity.
    """
    name = "Simple Hearts"

    def _get_trump_suit(self) -> int:
        return -1

    def _end_game_bonuses(self) -> List[int]:
        scores = self.scores
        loser = min(range(self.num_players), key=lambda i: scores[i])
        if scores[loser] == -17:
            return [17 if i == loser else -17 for i in range(self.num_players)]
        return [0 for _ in range(self.num_players)]

    def _end_trick(self) -> Tuple[List[int], int]:
        winning_player, winning_card = self._get_trick_winner()
        rewards = [0 for _ in range(self.num_players)]
        for i in range(self.num_cards, self.num_cards + self.num_players):
            card = self.index_to_card(self._state[i])
            if card.suit == Suit.HEARTS:
                rewards[winning_player] -= 1
            if card == Card(Suit.SPADES, 5):
                rewards[winning_player] -= 9
        return rewards, winning_player

    @property
    def cards_per_suit(self) -> Tuple[int, ...]:
        return 8, 8, 8, 8
