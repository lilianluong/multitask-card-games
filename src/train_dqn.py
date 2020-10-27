# Created by Patrick Kao
import torch

from agents.dqn_agent import DQNLearner, DQNAgent
from agents.random_agent import RandomAgent
from environments.hearts import SimpleHearts
from game import Game


def train(save_path):
    learner = DQNLearner()
    trained_model = learner.train([SimpleHearts])
    torch.save(trained_model.state_dict(), save_path)


def eval_filename(load_path):
    model = torch.load(load_path)
    eval(model)


def eval(model):
    # TODO: test this method
    results = []
    with torch.no_grad:
        for _ in range(10):
            game = Game(SimpleHearts, [DQNAgent] + [RandomAgent] * 3,
                        agent_params=[{"model": model}, {}, {}, {}])
            result = game.run()
            results.append(result)
    return results


if __name__ == "__main__":
    train("saved_model.pt")
