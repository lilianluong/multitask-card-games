# Created by Patrick Kao
import numpy as np
import torch

from agents.random_agent import RandomAgent
from environments.hearts import SimpleHearts
from environments.test_hearts import TestSimpleHearts
from game import Game


def evaluate_random(agent_type, model, num_trials=25):
    with torch.no_grad():
        scores = []
        num_invalid = 0
        for _ in range(num_trials):
            game = Game(TestSimpleHearts, [agent_type] + [RandomAgent] * 3,
                        [{"model": model}, {}, {}, {}],
                        {"epsilon": 0, "verbose": False})
            score = game.run()
            scores.append(score)
            infos = game.get_info()
            for info in infos:
                if 0 in info and info[0] == "invalid":
                    num_invalid += 1

        record = []
        for score in scores:
            record.append(True if np.argmax(score) == 0 else False)
        winrate = record.count(True) / len(record)
        avg_score = np.asarray(scores)[:, 0].mean()

        # calculate invalid
        constant_game = TestSimpleHearts()

        percent_invalid = num_invalid / (
                    constant_game.num_cards / constant_game.num_players * num_trials)
        return winrate, avg_score, percent_invalid, scores
