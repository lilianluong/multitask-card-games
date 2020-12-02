import math
import random
import time
from collections import defaultdict, Counter
from typing import Tuple, List, Union

import numpy as np
import torch

from agents.belief_agent import BeliefBasedAgent
from agents.models.model_based_models import ApprenticeModel
from agents.models.multitask_models import MultitaskApprenticeModel
from environments.trick_taking_game import TrickTakingGame
from util import Card

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

action_tensor_cache = {}


def mcts(executor, num_workers, belief, game, transition_model, reward_model, task_name,
         timeout: float = 0.5,
         horizon: int = 3,
         inverse_discount=1.2) -> int:
    """
    Given models and state, outputs action
    :param executor:
    :param game:
    :param timeout:
    :param horizon:
    :param inverse_discount:
    :return:
    """
    mcts_helper = _MCTSRunner(game, transition_model, reward_model, task_name, timeout, horizon,
                              inverse_discount)
    thread_results = executor.map(mcts_helper, [belief.detach()] * num_workers)
    # thread_results = [mcts_helper(belief) for _ in range(num_workers)]
    thread_scores, thread_plays = list(map(list, zip(*thread_results)))
    # combine scores lists
    scores_counter = Counter()
    for d in thread_scores:
        scores_counter.update(d)
    scores = dict(scores_counter)
    # combine plays lists
    plays_counter = Counter()
    for d in thread_plays:
        plays_counter.update(d)
    plays = dict(plays_counter)
    # compute best move
    card_index = _get_final_action(belief, game, scores, plays)
    return card_index


def _get_final_action(belief, game, scores, plays):
    action_list = range(game.num_cards - 1)
    valid_actions = [action for action in action_list if
                     game.valid_play_from_belief(belief, action)]
    action_values = {}
    for a in (valid_actions if len(valid_actions) else action_list):
        if (a,) in plays and (a,) in scores and plays[(a,)]:
            action_values[a] = scores[(a,)] / plays[(a,)]
        else:
            action_values[a] = -float('inf')

    # get key associated with biggest val
    return max(action_values, key=action_values.get)


class _MCTSRunner:
    """
    Helper class for mcts()
    """

    def __init__(self, game, transition_model, reward_model, task_name,
                 timeout: float = 0.5,
                 horizon: int = 4,
                 inverse_discount=1.2):
        self._game = game
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._task_name = task_name
        self._timeout = timeout
        self._horizon = horizon
        self._inverse_discount = inverse_discount

    def __call__(self, belief):
        return self._mcts_helper(belief)

    def get_transition_reward(self, current, selected_action, reward_cache, nodes, actions):
        new_node = current + (selected_action,)
        if new_node in reward_cache:
            reward = reward_cache[new_node]
        else:
            if current in nodes:
                belief = nodes[current]
            else:
                a = current[-1]
                ba = torch.cat([nodes[current[:-1]], actions[a:a + 1]], dim=1)
                belief = self._transition_model.forward(ba, self._task_name)
                nodes[current] = belief
            belief_action = torch.cat([belief, actions[selected_action:selected_action + 1]], dim=1)
            reward = self._reward_model.forward(belief_action, self._task_name)
            reward_cache[new_node] = reward
        return reward

    @staticmethod
    def ucb(score, plays, parent_plays, lowest_score, c=1.4):
        exploitation = score / plays if plays else 0
        exploitation /= abs(lowest_score) / 5  # normalization
        exploration = c * math.sqrt(math.log(parent_plays) / plays) if plays else float('inf')
        return exploitation + exploration

    def _mcts_helper(self, belief):
        # Monte Carlo
        t0 = time.time()
        timeout = self._timeout
        horizon = self._horizon
        inverse_discount = self._inverse_discount
        start_belief = belief  # torch.FloatTensor([belief]).to(device)
        actions = torch.eye(self._game.num_cards).float().to(device)
        num_actions = self._game.num_cards
        list_actions = list(range(num_actions))
        nodes = {tuple(): start_belief}
        plays = defaultdict(int)
        reward_cache = {}
        scores = defaultdict(float)
        lowest_score = 1
        while time.time() - t0 < timeout:
            current = tuple()
            plays[current] += 1
            total_reward = 0
            first_selection = True

            # Selection
            while len(current) < horizon and current + (0,) in plays:
                action_values = [_MCTSRunner.ucb(scores[current + (a,)],
                                                 plays[current + (a,)],
                                                 plays[current],
                                                 lowest_score)
                                 for a in list_actions]

                # on first selection, only choose from valid moves
                if first_selection:
                    first_selection = False
                    valid_list = [action for action in range(num_actions - 1) if
                                  self._game.valid_play_from_belief(belief, action)]
                    if len(valid_list) > 0:
                        selected_action = max(valid_list, key=lambda a: action_values[a])
                    else:
                        selected_action = max(list_actions, key=lambda a: action_values[a])
                else:
                    selected_action = max(list_actions, key=lambda a: action_values[a])

                reward = self.get_transition_reward(current, selected_action, reward_cache, nodes,
                                                    actions)
                total_reward = inverse_discount * total_reward + reward
                current = current + (selected_action,)
                plays[current] += 1

            # Expansion
            if len(current) < horizon and current + (0,) not in plays:
                plays[current + (0,)] = 0
                selected_action = random.randint(0, num_actions - 1)
                reward = self.get_transition_reward(current, selected_action, reward_cache, nodes,
                                                    actions)
                total_reward = inverse_discount * total_reward + reward
                current = current + (selected_action,)
                plays[current] += 1
            final_current = current

            # Simulation
            while len(current) < horizon:
                selected_action = random.randint(0, num_actions - 1)
                reward = self.get_transition_reward(current, selected_action, reward_cache, nodes,
                                                    actions)
                total_reward = inverse_discount * total_reward + reward
                current = current + (selected_action,)

            # Backpropagation
            for i in range(horizon + 1):
                scores[final_current[:i]] += total_reward.item()
            lowest_score = min(lowest_score, total_reward)

        # detach tensors
        return scores, plays


class ExpertIterationAgent(BeliefBasedAgent):
    def __init__(self, game: TrickTakingGame,
                 player_number: int,
                 apprentice_model: Union[ApprenticeModel, MultitaskApprenticeModel]):
        super().__init__(game, player_number)
        self._task_name = game.name
        self._apprentice_model = apprentice_model
        self._current_observation = None

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        super().observe(action, observation, reward)
        self._current_observation = observation

    def act(self, epsilon: float = 0) -> Card:
        if np.random.rand() <= epsilon:
            return self._game.index_to_card(random.randint(0, self._game.num_cards - 1))
            # valid_cards = self._get_hand(self._current_observation, valid_only=True)
            # return random.sample(valid_cards, 1)[0]

        action_values = self._apprentice_model.forward(torch.FloatTensor([self._belief]).to(device),
                                                       self._task_name)
        best_action = torch.argmax(action_values).item()
        return self._game.index_to_card(best_action)
