# Created by Patrick Kao
import torch

from agents.dqn_agent import DQNLearner, DQNAgent, DQN, calculate_action_observation_size
from agents.random_agent import RandomAgent
from environments.hearts import SimpleHearts
from environments.test_hearts import TestSimpleHearts
from evaluators import evaluate_random
from game import Game


def train(save_path):
    # old_model_state_dict = torch.load("saved_model.pt")
    # learner = DQNLearner(resume_state=old_model_state_dict)
    learner = DQNLearner()
    trained_model = learner.train([TestSimpleHearts])
    torch.save(trained_model.state_dict(), save_path)


def eval_filename(load_path):
    model_state_dict = torch.load(load_path)
    action_size, observation_size = calculate_action_observation_size(TestSimpleHearts)
    model = DQN(observation_size, action_size).to("cuda:0")
    model.load_state_dict(model_state_dict)
    winrate, avg_score, percent_invalid, scores = evaluate_random(DQNAgent, model, num_trials=1000)
    print(f"winrate: {winrate}\npercent invalid: {percent_invalid}")


if __name__ == "__main__":
    # train("saved_model.pt")
    eval_filename("/home/dolphonie/Desktop/MIT/6.883/saved_models/agent_49048.pt")