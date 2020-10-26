# Created by Patrick Kao
from agents.random_agent import RandomAgent
from environments.hearts import SimpleHearts
from game import Game


def test_hearts():
    game = Game(SimpleHearts, [RandomAgent]*4)
    result = game.run()
    print(result)


if __name__ == "__main__":
    test_hearts()
