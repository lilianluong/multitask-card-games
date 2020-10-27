"""
Model-based Agent-Learner pair.
"""

import random
from typing import List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from agents.base import Agent, Learner
from agents.belief_agent import BeliefBasedAgent
from environments.trick_taking_game import TrickTakingGame
from game import Game
from util import Card

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransitionModel(nn.Module):
    """
    Multilayered perceptron to approximate T: (b, a) -> b'
    """

    def __init__(self, belief_size: int, num_actions: int, num_players: int):
        """
        :param belief_size: number of values in a belief
        :param num_actions: number of possible actions, to be 1-hot encoded and attached to belief
        :param num_players: number of players in game, used to partition belief for sigmoid and loss
        """
        super().__init__()
        h1 = 220
        h2 = 200
        h3 = 180
        self._num_players = num_players
        self.model = nn.Sequential(
            nn.Linear(belief_size + num_actions, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, belief_size)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of the model
        :param x: a shape (batch_size, belief_size + num_actions) torch Float tensor, beliefs concatenated with actions
        :return: a shape (batch_size, belief_size) torch Float tensor, the predicted next belief
        """
        fc_out = self.model(x)
        fc_out[:, :-self._num_players] = nn.Sigmoid()(fc_out[:, :-self._num_players])
        return fc_out

    def loss(self, pred: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculate the loss of a batch of predictions against the true labels
        :param pred: (batch_size, belief_size) predicted next beliefs
        :param y: (batch_size, belief_size) actual next beliefs
        :return: mean loss as a torch Float scalar
        """
        bce_loss = nn.BCELoss()(pred[:, :-self._num_players], y[:, :-self._num_players])
        mse_loss = nn.MSELoss()(pred[:, -self._num_players:], y[:, -self._num_players:])
        # TODO: Regularization?
        return bce_loss + mse_loss


class RewardModel(nn.Module):
    """
    Multilayered perceptron to approximate T: (b, a) -> r
    """

    def __init__(self, belief_size: int, num_actions: int):
        """
        :param belief_size: number of values in a belief
        :param num_actions: number of possible actions, to be 1-hot encoded and attached to belief
        """
        super().__init__()
        h1 = 200
        h2 = 100
        h3 = 50
        self.model = nn.Sequential(
            nn.Linear(belief_size + num_actions, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h3),
            nn.ReLU(inplace=True),
            nn.Linear(h3, 1)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of the model
        :param x: a shape (batch_size, belief_size + num_actions) torch Float tensor, beliefs concatenated with actions
        :return: a shape (batch_size, 1) torch Float tensor, the predicted reward
        """
        return self.model(x)

    def loss(self, pred: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculate the loss of a batch of predictions against the true labels
        :param pred: (batch_size, 1) predicted rewards
        :param y: (batch_size, 1) actual rewards
        :return: mean loss as a torch Float scalar
        """
        mse_loss = nn.MSELoss()(pred, y)
        # TODO: Regularization?
        return mse_loss


class ModelBasedAgent(BeliefBasedAgent):
    def __init__(self, game: TrickTakingGame,
                 player_number: int,
                 transition_model: TransitionModel,
                 reward_model: RewardModel):
        super().__init__(game, player_number)
        self._transition_model = transition_model
        self._reward_model = reward_model
        self._current_observation = None

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        super().observe(action, observation, reward)
        self._current_observation = observation

    def act(self, epsilon: float = 0) -> Card:
        if np.random.rand() <= epsilon:
            valid_cards = self._get_hand(self._current_observation, valid_only=True)
            return random.sample(valid_cards, 1)[0]

        # TODO: Select an action
        raise NotImplementedError


class ModelBasedLearner(Learner):
    def __init__(self, multitask: bool = False):
        """
        :param multitask: whether to use multitask learning or not
        """
        if multitask:
            raise NotImplementedError
        self._transition_models = {}
        self._reward_models = {}

        # Hyperparameters
        self._num_epochs = 100000
        self._games_per_epoch = 3

    def train(self, tasks: List[TrickTakingGame.__class__]):
        for task in tasks:
            self._train_single_task(task)

    def _train_single_task(self, task: TrickTakingGame.__class__):
        sample_task = task()
        # TODO: initialize models
        # self._transition_models[task.name] = TransitionModel()
        for epoch_num in range(self._num_epochs):
            experiences = self._agent_evaluation(task)
            self._train_world_models(task, experiences)
            self._train_agent_policy(task)

    def _agent_evaluation(self, task: TrickTakingGame.__class__) -> List[Tuple[List[int], int, int, List[int]]]:
        """
        Collect (b, a, r, b') experiences from playing self._games_per_epoch games against itself
        :param task: the class of the game to play
        :return: list of (b, a, r, b') experiences
        """
        game = Game(task, [ModelBasedAgent] * 4)  # TODO: add models as parameters
        # TODO: port over Patrick's code from DQN

    def _train_world_models(self, task: TrickTakingGame.__class__,
                            experiences: List[Tuple[List[int], int, int, List[int]]]):
        """
        Train the transition and reward models on the experiences
        :param task: the class of the game to train models for
        :param experiences: list of (b, a, r, b') experiences as returned by _agent_evaluation
        :return: None
        """
        raise NotImplementedError  # TODO: Implement!

    def _train_agent_policy(self, task: TrickTakingGame.__class__):
        """
        Train a policy for the agent. TODO: What should this involve?
        :param task: the class of the game to train for
        :return: None
        """
        pass

    def initialize_agent(self, game: TrickTakingGame, player_number: int) -> Agent:
        return ModelBasedAgent(game, player_number, self._transition_models[game.name], self._reward_models[game.name])
