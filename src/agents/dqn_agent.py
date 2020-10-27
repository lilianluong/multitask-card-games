import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from agents.base import Learner, Agent
from environments.trick_taking_game import TrickTakingGame
from game import Game

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, observation_size, action_size, H1=16, H2=16):
        super(DQN).__init__()
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


class DQNAgent(Agent):
    def __init__(self, game: TrickTakingGame, player_number: int, model: DQN):
        super(DQNAgent).__init__(game, player_number)
        self._current_observation = None
        self.model = model
        self.epsilon = 0.05


def act(self):
    if np.random.rand() <= self.epsilon:
        valid_cards = self._get_hand(self._current_observation, valid_only=True)
        return random.sample(valid_cards, 1)[0]

    action_values = self.model.predict(torch.FloatTensor(self._current_observation))
    return np.argmax(action_values[0])


def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
    self._current_observation = observation


class DQNLearner(Learner):

    def __init__(self):

        self.observation_size = 33  # 24 cards + 4 players + 4 scores + 1 current player index
        self.action_size = 24
        self.memory = deque(maxlen=100)  # modification to dqn to preserve recent only
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # min exploration
        self.epsilon_decay = 0.995  # to decrease exploration rate over time
        self.learning_rate = 5E-4
        self.model = DQN(self.observation_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_every = 3
        self.step = 0

    def train(self, tasks: List[TrickTakingGame.__class__], episodes: int):
        for task in tasks:
            player_setting = []  # TODO
            game = Game(task, player_setting)
            agent = self.initialize_agent(game, 1)
            scores = []
            for i in range(episodes):
                score = 0
                observation = task.reset()
                done = False
                while not done:
                    action = agent.act(observation)
                    next_observation, reward, done, _ = task.step(action)
                    agent.observe(action, next_observation, reward)
                    agent.memorize(observation, action, reward, next_observation, done)
                    score += reward
                    observation = next_observation
                print("Score: ", score)
                scores.append(score)
                agent.replay(32)  # TODO: Figure out if 32 makes sense, incorporate update_every

    def initialize_agent(self, game: TrickTakingGame, player_number: int) -> Agent:
        return DQNAgent(game, player_number)

    def memorize(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        criterion = torch.nn.CrossEntropyLoss()

        for observation, action, reward, next_observation, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_observation)[0])
            pred = self.model.predict(observation).gather(1, action)
            loss = criterion(pred, target).to_device()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
