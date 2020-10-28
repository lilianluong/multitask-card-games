import torch

from agents.model_based_learner import ModelBasedLearner, ModelBasedAgent
from agents.random_agent import RandomAgent
from environments.test_hearts import TestSimpleHearts
from evaluators import evaluate_random
from game import Game


def train(save_path):
    # old_model_state_dict = torch.load("5000.pt")
    learner = ModelBasedLearner()#resume_state=old_model_state_dict)
    trained_model = learner.train([TestSimpleHearts])
    torch.save(trained_model.state_dict(), save_path)


def eval_filename(load_path):
    model = torch.load(load_path)
    evaluate_random(ModelBasedAgent, model, num_trials=500)


if __name__ == "__main__":
    train("saved_model.pt")
