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

def train(tasks, load_model_names, save_model_names, multitask, learner_name):
    # Set up learner
    if load_model_names:
        resume = {"transition": {}, "reward": {}, "apprentice": {}}
        for task in tasks:
            transition_state = torch.load("models/transition_model_temp_{}.pt".format(load_model_names[task.name]))
            reward_state = torch.load("models/reward_model_temp_{}.pt".format(load_model_names[task.name]))
            apprentice_state = torch.load("models/apprentice_model_temp_{}.pt".format(load_model_names[task.name]))
            resume["transition"][task.name] = {"state": transition_state, "task": task}
            resume["reward"][task.name] = {"state": reward_state, "task": task}
            resume["apprentice"][task.name] = {"state": apprentice_state, "task": task}
    else:
        resume = None
    learner = ModelBasedLearner(agent=ModelBasedAgent, model_names=save_model_names,
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
        # train([TestSimpleHearts, TrickTakingGame, BasicTrick, TwentyFive],
        #       None,
        #       {"Test Simple Hearts": f"multitask_tsh_{i}", "Trick Taking Game": f"multitask_ttg_{i}",
        #        "Basic Trick": f"multitask_basic_{i}", "Twenty-Five": f"multitask_25_{i}"},
        #       multitask=True,
        #       learner_name=f"new-multitask-{i}")
        train([TestSimpleHearts, TrickTakingGame, BasicTrick, TwentyFive],
              None,
              {"Test Simple Hearts": f"singletask_tsh_{i}", "Trick Taking Game": f"singletask_ttg_{i}",
               "Basic Trick": f"singletask_basic_{i}", "Twenty-Five": f"singletask_25_{i}"},
              multitask=False,
              learner_name=f"new-singletask-{i}")
