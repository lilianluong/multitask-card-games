from environments.hearts import SimpleHearts
from game import Game
from environments.flask_game import FlaskGame
from flask import Flask

def test_hearts():
    game = Game(SimpleHearts)
    result = game.run()
    print(result)


if __name__ == "__main__":
    test_hearts()
