# Created by Patrick Kao
import numpy as np
import torch

from agents.random_agent import RandomAgent
from environments.hearts import SimpleHearts
from game import Game


def evaluate_random(agent_type, model, num_trials=25):
    with torch.no_grad():
        scores = []
        for _ in range(num_trials):
            game = Game(SimpleHearts, [agent_type] + [RandomAgent] * 3, [{"model": model}, {}, {}, {}],
                        {"epsilon": 0, "verbose": False})
            score = game.run()
            scores.append(score)

        record = []
        for score in scores:
            record.append(True if np.argmax(score) == 0 else False)
        winrate = record.count(True) / len(record)
        avg_score = np.asarray(scores)[:, 0].mean()

        return winrate, avg_score, scores
