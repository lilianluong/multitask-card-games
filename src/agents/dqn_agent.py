import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from agents.base import Learner
from agents.belief_agent import BeliefBasedAgent
from environments.hearts import SimpleHearts
from environments.trick_taking_game import TrickTakingGame
from game import Game

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, observation_size, action_size, H1=50, H2=50):
        """

        :param observation_size: Size of belief as defined in belief_agent.py
        :param action_size: Model has 1 output for every single possible card in the deck.
        :param H1: size of hidden layer 1
        :param H2: size of hidden layer 2
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_size, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, action_size)

    def forward(self, observation):
        '''
        Maps observation to action values.
        '''
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(BeliefBasedAgent):
    """
    DQN agent

    A set of cards is represented by a one-hot vector with length equal to number of cards in the
    game. The vector has a one if the card is present in the set and 0 otherwise
    """

    def __init__(self, game: TrickTakingGame, player_number: int, model: DQN):
        super().__init__(game, player_number)
        self._current_observation = None
        self.model = model

    def act(self, epsilon: float = 0):
        if np.random.rand() <= epsilon:
            valid_cards = self._get_hand(self._current_observation, valid_only=True)
            return random.sample(valid_cards, 1)[0]

        # reformat observation into following format: hand +
        action_values = self.model.forward(torch.FloatTensor(self._belief).to(device))
        chosen_card = torch.argmax(action_values).item()
        return self._game.index_to_card(chosen_card)

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        super().observe(action, observation, reward)
        self._current_observation = observation


class DQNLearner(Learner):

    def __init__(self):

        # calculate parameter sizes
        constant_game = SimpleHearts()
        cards_per_suit = constant_game.cards_per_suit[0]
        num_cards = constant_game.num_cards
        self.action_size = num_cards
        self.observation_size = num_cards * 4 + cards_per_suit
        self.epsilon_greedy = 0.05  # percent time to be epsilon greedy
        self.memory = deque(maxlen=100)  # modification to dqn to preserve recent only
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # min exploration
        self.epsilon_decay = 0.995  # to decrease exploration rate over time
        self.learning_rate = 5E-4

        # training hyperparams
        self.num_epochs = 100000
        self.games_per_epoch = 3
        self.batch_size = 20
        self.num_batches = 2

        # Init agents and trainers
        self.model = DQN(self.observation_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_every = 3
        self.step = 0

    def train(self, tasks: List[TrickTakingGame.__class__]) -> nn.Module:
        for task in tasks:
            for epoch in range(self.num_epochs):
                # collect experiences
                print(f"Starting epoch {epoch}/{self.num_epochs}")
                for game_num in range(self.games_per_epoch):
                    # print(
                    #     f"Running game {game_num}/{self.games_per_epoch} in epoch {epoch}/"
                    #     f"{self.num_epochs}")
                    game = Game(task, [DQNAgent] * 4, [{"model": self.model} for _ in range(4)],
                                {"epsilon": self.epsilon, "verbose": False})
                    result = game.run()
                    barbs = game.get_barbs()
                    self.memorize(barbs)
                # update policy
                for _ in range(self.num_batches):
                    self.replay(self.batch_size)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

        return self.model

    def memorize(self, barbs):
        """
        Keep memory as array with size num samples x 4 (b, a, r, b) x varies
        :param barbs:
        :return:
        """
        self.memory.extend(barbs)

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        criterion = torch.nn.MSELoss()  # tutorial uses torch.nn.SmoothL1Loss

        # TODO: what about terminal behavior?
        # Batch data correctly
        batch = np.asarray(batch)
        belief = np.vstack(batch[:, 0])
        action = np.vstack(batch[:, 1])
        reward = np.vstack(batch[:, 2])
        next_belief = np.vstack(batch[:, 3])

        belief, next_belief = torch.from_numpy(belief).type(torch.FloatTensor).to(device), \
                              torch.from_numpy(next_belief).type(torch.FloatTensor).to(device)
        # Linear expects dims batch size x feature size (feat size is observation size here)
        target = torch.from_numpy(reward).to(device) + self.gamma * torch.argmax(
            self.model.forward(next_belief), dim=1, keepdim=True)
        pred = self.model.forward(belief)
        pred = torch.gather(pred, 1, torch.from_numpy(action).to(device)).to(
            device)  # convert prediction
        loss = criterion(pred, target).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
