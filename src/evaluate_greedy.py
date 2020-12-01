from agents.greedy_agent import GreedyAgent, GreedyTwentyFiveAgent, GreedyHeartsAgent
from agents.random_agent import RandomAgent
from environments.hearts import SimpleHearts
from environments.twentyfive import TwentyFive
from game import Game


def test_greedy():
	game = Game(SimpleHearts, [GreedyHeartsAgent] + [RandomAgent]*3, [{'win_trick': True}, {}, {}, {}])
	result = game.run()
	print(result)

if __name__ == "__main__":
	test_greedy()