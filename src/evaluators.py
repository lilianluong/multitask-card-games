# Created by Patrick Kao
import numpy as np
import torch

from agents.random_agent import RandomAgent
from game import Game


def evaluate_random(tasks, agent_type, models, num_trials=50, compare_agent=None):
    """
    Evaluate an agent against 3 random agents on a list of tasks
    :param tasks: list of task classes to evaluate on
    :param agent_type: class of the agent to evaluate
    :param models: dict mapping task names to list of models (transition_model, reward_model, ...)
    :param num_trials: int, number of trial games to run per task
    :param compare_agent: optional other agent to compare agent_type to, default will use a random agent
    :return: (win_rate, avg_score, percent_invalid, scores)
        win_rate: percentage of time agent_type scores at least as well as compare_agent would on the same initial deal
        avg_score: average score that agent_type beats compare_agent by on the same initial deal
        percent_invalid: percentage of time agent_type plays an invalid card
        scores: list of score vectors
    """
    with torch.no_grad():
        scores = []
        random_scores = []
        num_invalid = 0
        total_cards_played = 0
        for task in tasks:
            task_models = models[task.name]
            for trial_num in range(num_trials):
                # print(trial_num)
                # Evaluate agent
                game = Game(task,
                            [agent_type] + [RandomAgent] * 3,
                            [{"transition_model": task_models[0], "reward_model": task_models[1]}, {}, {}, {}],
                            # [{"model": model}, {}, {}, {}],
                            {"epsilon": 0, "verbose": False})
                score, state = game.run()
                scores.append(score)

                # Evaluate current agent on same starting state
                if compare_agent and not isinstance(compare_agent, RandomAgent):
                    game = Game(task,
                                [compare_agent] + [RandomAgent] * 3,
                                [{"transition_model": task_models[0], "reward_model": task_models[1]}, {}, {}, {}],
                                # [{"model": model}, {}, {}, {}],
                                {"epsilon": 0, "verbose": False})
                else:
                    game = Game(task, [RandomAgent] * 4, [{}] * 4, {"epsilon": 0, "verbose": False})
                random_score, _ = game.run(state)
                random_scores.append(random_score)

                infos = game.get_info()
                for info in infos:
                    if 0 in info and info[0] == "invalid":
                        num_invalid += 1
            constant_game = task()
            total_cards_played += constant_game.num_cards / constant_game.num_players * num_trials

        record = []
        for i, score in enumerate(scores):
            record.append(True if score[0] >= random_scores[i][0] else False)
            # record.append(True if np.argmax(score) == 0 else False)
        winrate = record.count(True) / len(record)
        avg_score_margin = (np.asarray(scores)[:, 0] - np.asarray(random_scores)[:, 0]).mean()

        # calculate invalid
        percent_invalid = num_invalid / total_cards_played

        return winrate, avg_score_margin, percent_invalid, scores
