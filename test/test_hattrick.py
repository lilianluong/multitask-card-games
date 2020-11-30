# Created by Patrick Kao
import unittest

from agents.random_agent import RandomAgent
from environments.hattrick import HatTrick
from game import Game


class HatTrickTest(unittest.TestCase):
    def test_game(self):
        game = Game(HatTrick, [RandomAgent] * 4)
        result = game.run()
        print(result)
        self.assertTrue(result is not None)
