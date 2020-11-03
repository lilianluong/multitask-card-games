import math
import random
import time
from collections import deque, defaultdict
from typing import Tuple, List

import numpy as np
import torch

from agents.belief_agent import BeliefBasedAgent
from agents.model_based_models import RewardModel, TransitionModel
from environments.trick_taking_game import TrickTakingGame
from util import Card

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

action_tensor_cache = {}


class ModelBasedAgent(BeliefBasedAgent):
    def __init__(self, game: TrickTakingGame,
                 player_number: int,
                 transition_model: TransitionModel,
                 reward_model: RewardModel):
        super().__init__(game, player_number)
        self._task_name = game.name
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._current_observation = None
        if self._game.num_cards not in action_tensor_cache:
            action_tensor_cache[self._game.num_cards] = torch.eye(self._game.num_cards).to(device)

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        super().observe(action, observation, reward)
        self._current_observation = observation

    def act(self, epsilon: float = 0) -> Card:
        if np.random.rand() <= epsilon:
            return self._game.index_to_card(random.randint(0, self._game.num_cards - 1))
            # valid_cards = self._get_hand(self._current_observation, valid_only=True)
            # return random.sample(valid_cards, 1)[0]

        # search
        horizon = 1
        inverse_discount = 1.1
        actions = self._game.num_cards
        nodes = deque()
        nodes.append((torch.FloatTensor([self._belief]).to(device), None, 0, 0))  # belief, first_action, reward, steps
        best_first_action = 0
        best_score = -float('inf')
        while len(nodes):
            belief, first_action, reward, steps = nodes.popleft()
            if steps == horizon: break
            x = torch.cat([belief.repeat(actions, 1), action_tensor_cache[actions]], dim=1)
            action_values = self._reward_model.forward(x, self._task_name)
            next_beliefs = None
            if steps < horizon - 1:
                next_beliefs = self._transition_model.forward(x, self._task_name)
            for i in range(actions):
                new_reward = inverse_discount * reward + action_values[i].item()
                if steps < horizon - 1:
                    nodes.append((next_beliefs[i:i+1],
                                  first_action if first_action else i,
                                  new_reward,
                                  steps + 1))
                elif steps == horizon - 1:
                    if new_reward > best_score:
                        best_score = new_reward
                        best_first_action = i
        return self._game.index_to_card(best_first_action)


class MonteCarloAgent(ModelBasedAgent):
    def __init__(self, game: TrickTakingGame,
                 player_number: int,
                 transition_model: TransitionModel,
                 reward_model: RewardModel,
                 timeout: float = 0.5,
                 horizon: int = 4,
                 inverse_discount = 1.2):
        super().__init__(game, player_number, transition_model, reward_model)
        self._timeout = timeout
        self._horizon = horizon
        self._inverse_discount = inverse_discount

    @staticmethod
    def ucb(score, plays, parent_plays, lowest_score, c=1.4):
        exploitation = score / plays if plays else 0
        exploitation /= abs(lowest_score) / 5  # normalization
        exploration = c * math.sqrt(math.log(parent_plays) / plays) if plays else float('inf')
        return exploitation + exploration

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

    def act(self, epsilon: float = 0) -> Card:
        if np.random.rand() <= epsilon:
            return self._game.index_to_card(random.randint(0, self._game.num_cards - 1))
            # valid_cards = self._get_hand(self._current_observation, valid_only=True)
            # return random.sample(valid_cards, 1)[0]

        # Monte Carlo
        t0 = time.time()
        timeout = self._timeout
        horizon = self._horizon
        inverse_discount = self._inverse_discount
        start_belief = torch.FloatTensor([self._belief]).to(device)
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

            # Selection
            while len(current) < horizon and current + (0,) in plays:
                action_values = [MonteCarloAgent.ucb(scores[current + (a,)],
                                                            plays[current + (a,)],
                                                            plays[current],
                                                            lowest_score)
                                 for a in list_actions]
                selected_action = max(list_actions, key=lambda a: action_values[a])
                reward = self.get_transition_reward(current, selected_action, reward_cache, nodes, actions)
                total_reward = inverse_discount * total_reward + reward
                current = current + (selected_action,)
                plays[current] += 1

            # Expansion
            if len(current) < horizon and current + (0,) not in plays:
                plays[current + (0,)] = 0
                selected_action = random.randint(0, num_actions - 1)
                reward = self.get_transition_reward(current, selected_action, reward_cache, nodes, actions)
                total_reward = inverse_discount * total_reward + reward
                current = current + (selected_action,)
                plays[current] += 1
            final_current = current

            # Simulation
            while len(current) < horizon:
                selected_action = random.randint(0, num_actions - 1)
                reward = self.get_transition_reward(current, selected_action, reward_cache, nodes, actions)
                total_reward = inverse_discount * total_reward + reward
                current = current + (selected_action,)

            # Backpropagation
            for i in range(horizon + 1):
                scores[final_current[:i]] += total_reward
            lowest_score = min(lowest_score, total_reward)

        card_index = max(list_actions, key=lambda a: scores[(a,)] / plays[(a,)] if plays[(a,)] else -float('inf'))
        return self._game.index_to_card(card_index)
