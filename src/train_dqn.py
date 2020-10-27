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
    evaluate_random(DQNAgent, model, num_trials=500)


if __name__ == "__main__":
    train("saved_model.pt")
