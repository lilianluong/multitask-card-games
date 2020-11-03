import torch

from agents.model_based_learner import ModelBasedLearner, ModelBasedAgent, MonteCarloAgent
from agents.random_agent import RandomAgent
from environments.test_hearts import TestSimpleHearts
from environments.trick_taking_game import TrickTakingGame
from environments.hearts import SimpleHearts
from evaluators import evaluate_random


MODEL_PARAMS = {
    "Trick Taking Game": [104, 24, 4],
    "Test Simple Hearts": [104, 24, 4],
    "Simple Hearts": [136, 32, 4]
}


def train(tasks, load_model_names, save_model_names):
    # Set up learner
    if load_model_names:
        resume = {"transition": {}, "reward": {}}
        for task in tasks:
            transition_state = torch.load("models/transition_model_temp_{}.pt".format(load_model_names[task.name]))
            reward_state = torch.load("models/reward_model_temp_{}.pt".format(load_model_names[task.name]))
            resume["transition"][task.name] = {"state": transition_state, "task": task}
            resume["reward"][task.name] = {"state": reward_state, "task": task}
    else:
        resume = None
    learner = ModelBasedLearner(agent=ModelBasedAgent, model_names=save_model_names, resume_model=resume)

    # # Evaluate
    # evaluate = evaluate_random(tasks,
    #                            ModelBasedAgent,
    #                            {task.name: learner.get_models(task) for task in tasks},
    #                            num_trials=100)
    # print(evaluate)

    learner.train(tasks)


if __name__ == "__main__":
    train([TestSimpleHearts, TrickTakingGame],
          None,
          {"Test Simple Hearts": "multitask_single_tsh", "Trick Taking Game": "multitask_single_test_ttg"}
          )
