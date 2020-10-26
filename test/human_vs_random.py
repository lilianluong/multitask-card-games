# Created by Patrick Kao
from agents.human import Human
from agents.random_agent import RandomAgent
from environments.flask_game import FlaskGame
from environments.hearts import SimpleHearts


def test_hearts():
    game = FlaskGame.getInstance(SimpleHearts, [RandomAgent] * 3 + [Human])
    result = game.run()
    print(result)


if __name__ == "__main__":
    test_hearts()
