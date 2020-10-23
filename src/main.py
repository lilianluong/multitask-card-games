from environments.hearts import SimpleHearts
from game import Game
from environments.flask_game import FlaskGame
from flask import Flask

def test_hearts():
    game = Game(SimpleHearts)
    result = game.run()
    print(result)


def test_flask_hearts():
    game = FlaskGame(SimpleHearts)
    game.run(debug=True)


if __name__ == "__main__":
    test_hearts()
