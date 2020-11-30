from typing import List, Tuple

from environments.twentyfive import TwentyFive


class HatTrick(TwentyFive):
    '''
    Environment for modified version of twenty-five, a 4-player trump trick taking game.

    This variation plays with 8 cards of each suit, for a total of 32. Players receive 5 points
    for every trick, which can be won by playing the highest of the suit that was led or a trump
    card.
    A trump card can be played at any time and the hierarchy for the trump suit is 5, J,
    Highest Card of Hearts,
    followed by the remaining cards in the trump suit in the traditional numerical order. Winning
    three
    tricks leads to an additional bonus of 25 points, but crossing three tricks results in a 50
    point penalty.

    The game consists of 5 rounds and is won by the player with the most points.

    '''

    name = 'Hat Trick'

    def __init__(self):
        super().__init__()
        self._tricks_played = 0

    def _end_trick(self) -> Tuple[List[int], int]:
        self._tricks_played += 1
        return super()._end_trick()

    def _game_has_ended(self) -> bool:
        return self.tricks_played == 5

    def _end_game_bonuses(self) -> List[int]:
        """
        Computes additional reward assigned to each player at the end of a game.
        May be overwritten by child classes.
        :return: vector of bonus rewards for each player
        """
        rewards = [0 for _ in range(self.num_players)]
        for player in range(self.num_players):
            if self.scores[player] == 10:
                rewards[player] += 25  # 25pt bonus
            elif self.scores[player] >= 40:  # aka crossing three tricks
                rewards[player] -= 50

        return rewards

    @property
    def tricks_played(self):
        return self._tricks_played
