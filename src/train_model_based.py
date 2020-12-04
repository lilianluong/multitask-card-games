import torch

from agents.expert_iteration_agent import ExpertIterationAgent
from agents.expert_iteration_learner import ExpertIterationLearner
from agents.model_based_agent import MonteCarloAgent
from agents.model_based_learner import ModelBasedLearner, ModelBasedAgent
from agents.random_agent import RandomAgent
from environments.test_hearts import TestSimpleHearts
from environments.trick_taking_game import TrickTakingGame
from environments.hearts import SimpleHearts
from environments.twentyfive import TwentyFive
from environments.hattrick import HatTrick
from environments.basic_trick import BasicTrick
from evaluators import evaluate_random


MODEL_PARAMS = {
    "Trick Taking Game": [104, 24, 4],
    "Test Simple Hearts": [104, 24, 4],
    "Simple Hearts": [136, 32, 4], 
    "Test TwentyFive": [136, 32, 4],
    "Test HatTrick": [136, 32, 4], 
    "Test Basic Trick": [136, 32, 4]
}


def train(tasks, load_model_names, save_model_names, multitask, learner_name):
    # Set up learner
    if load_model_names:
        resume = {"transition": {}, "reward": {}, "apprentice": {}}
        for task in tasks:
            transition_state = torch.load("models/transition_model_{}.pt".format(load_model_names[task.name]))
            reward_state = torch.load("models/reward_model_{}.pt".format(load_model_names[task.name]))
            apprentice_state = torch.load("models/apprentice_model_{}.pt".format(load_model_names[task.name]))
            resume["transition"][task.name] = {"state": transition_state, "task": task}
            resume["reward"][task.name] = {"state": reward_state, "task": task}
            resume["apprentice"][task.name] = {"state": apprentice_state, "task": task}
    else:
        resume = None
    learner = ExpertIterationLearner(agent=ExpertIterationAgent, model_names=save_model_names,
                                     resume_model=resume, multitask=multitask,
                                     learner_name=learner_name)

    # # Evaluate
    # evaluate = evaluate_random(tasks,
    #                            ModelBasedAgent,
    #                            {task.name: learner.get_models(task) for task in tasks},
    #                            num_trials=100)
    # print(evaluate)

    learner.train(tasks)


if __name__ == "__main__":
    for i in range(5):
        train([TestSimpleHearts, TrickTakingGame],
              {"Test Simple Hearts": f"multitask_tsh_{i}",
               "Trick Taking Game": f"multitask_ttg_{i}"},
              {"Test Simple Hearts": f"multitask_cont_tsh_{i}",
               "Trick Taking Game": f"multitask_cont_ttg_{i}"},
              multitask=True,
              learner_name=f"exit-cont-{i}")
        train([TestSimpleHearts, TrickTakingGame],
              None,
              {"Test Simple Hearts": f"singletask_tsh_{i}", "Trick Taking Game": f"singletask_ttg_{i}"},
              multitask=False,
              learner_name=f"exit-{i}")
