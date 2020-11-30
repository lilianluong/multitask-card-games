# Created by Patrick Kao
import unittest
from typing import Tuple

from agents.random_agent import RandomAgent
from environments.twentyfive import TwentyFive
from game import Game
from util import Suit


class Mini25(TwentyFive):
    @property
    def cards_per_suit(self) -> Tuple[int, ...]:
        return 2, 2, 2, 2


class TwentyFiveTest(unittest.TestCase):
    def test_game(self):
        game = Game(TwentyFive, [RandomAgent] * 4)
        result = game.run()
        print(result)
        self.assertTrue(result is not None)

    def test_simple_game(self):
        game = Mini25()
        state = [0, 1, 2, 3, 0, 1, 2, 3, ]  # cards
        state.extend([-1 for _ in range(4)])  # in play
        state.extend([0 for _ in range(4)])  # score
        state.extend([Suit.SPADES, -1, 0])  # trump plus leading + players

        game.reset(state)

        plays = [0, 1, 2, 3, 5, 6, 7, 4]
        for turn in range(8):
            next_player = game.next_player
            play = plays[turn]
            observations, rewards, done, info = game.step((next_player, play))
            if turn != 7 and turn != 3:
                self.assertTrue(rewards == tuple([0] * 4))
            elif turn == 3:
                self.assertEqual(rewards, (-1.5, 5, -1.5, -1.5)) # normalized
            else:
                self.assertEqual(rewards, (-1.5, -1.5, -1.5, 5))

    # TODO: write test for bigger game to test odd trumps
