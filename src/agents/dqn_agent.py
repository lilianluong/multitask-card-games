import itertools
import multiprocessing
import random
from collections import deque
from concurrent import futures
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agents.base import Learner
from agents.belief_agent import BeliefBasedAgent
from environments.test_hearts import TestSimpleHearts
from environments.trick_taking_game import TrickTakingGame
from evaluators import evaluate_random
from game import Game

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, observation_size, action_size, H1=100, H2=80, H3=60, H4=30):
        """

        :param observation_size: Size of belief as defined in belief_agent.py
        :param action_size: Model has 1 output for every single possible card in the deck.
        :param H1: size of hidden layer 1
        :param H2: size of hidden layer 2
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_size, H1)
        self.fc2 = torch.nn.Linear(H1, H2)
        self.fc3 = torch.nn.Linear(H2, H3)
        self.fc4 = torch.nn.Linear(H3, H4)
        self.fc5 = torch.nn.Linear(H4, action_size)

    def forward(self, observation):
        '''
        Maps observation to action values.
        '''
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


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
            return self._game.index_to_card(random.randint(0, self._game.num_cards - 1))

        # reformat observation into following format: hand +
        action_values = self.model.forward(torch.FloatTensor(self._belief).to(device))
        chosen_card = torch.argmax(action_values).item()
        return self._game.index_to_card(chosen_card)

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        super().observe(action, observation, reward)
        self._current_observation = observation


class GameRunner:
    def __init__(self, task, agent_params, game_params):
        self.task = task
        self.agent_params = agent_params
        self.game_params = game_params

    def __call__(self, game_num):
        # print(f"Running game {game_num}")
        game = Game(self.task, [DQNAgent] * 4, self.agent_params, self.game_params)
        result = game.run()
        barbs = game.get_barbs()
        return barbs


def calculate_action_observation_size(game):
    # calculate parameter sizes
    constant_game = game()
    cards_per_suit = constant_game.cards_per_suit[0]
    num_cards = constant_game.num_cards
    return num_cards, num_cards*2

class DQNLearner(Learner):

    def __init__(self, resume_state=None):
        self.action_size, self.observation_size = calculate_action_observation_size(TestSimpleHearts)
        """ + len(
            constant_game.cards_per_suit) + constant_game.num_players"""
        self.memory = deque(maxlen=4000)  # modification to dqn to preserve recent only
        self.gamma = 0.1  # discount rate
        self.epsilon = 1.0  # exploration rate, percent time to be epsilon greedy
        self.epsilon_min = 0.1  # min exploration
        self.epsilon_decay = 0.995  # to decrease exploration rate over time
        self.learning_rate = 5E-4

        # training hyperparams
        self.num_epochs = 5000
        self.games_per_epoch = 100
        self.batch_size = 1000
        self.num_batches = 5

        # Init agents and trainers
        self.model = DQN(self.observation_size, self.action_size).to(device)
        if resume_state is not None:
            self.model.load_state_dict(resume_state)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.evaluate_every = 50  # number of epochs to evaluate between
        self.step = 0
        self.save_every = 1001  # save model every x iterations
        self.save_base_path = "saved_models/agent"

        self.writer = SummaryWriter(f"runs/dqn {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        # TODO: add network graph to tensborboard
        torch.multiprocessing.set_start_method('spawn')  # allow CUDA in multiprocessing

    def train(self, tasks: List[TrickTakingGame.__class__]) -> nn.Module:
        num_cpus = multiprocessing.cpu_count()
        num_threads = int(num_cpus / 2)  # can use more or less CPUs
        executor = futures.ProcessPoolExecutor(max_workers=num_threads)
        for task in tasks:
            for epoch in range(self.num_epochs):
                # collect experiences
                print(f"Starting epoch {epoch}/{self.num_epochs}")
                specific_game_func = GameRunner(task,
                                                [{"model": self.model} for _ in
                                                 range(4)], {"epsilon": self.epsilon,
                                                             "verbose": False})
                barb_futures = executor.map(specific_game_func, range(self.games_per_epoch),
                                            chunksize=2)
                # wait for completion
                barbs = list(barb_futures)
                barbs = list(itertools.chain.from_iterable(barbs))
                self.memorize(barbs)
                # update policy
                losses = []
                for _ in range(self.num_batches):
                    loss = self.replay(self.batch_size).item()
                    losses.append(loss)

                self.writer.add_scalar("avg_training_loss", np.mean(losses), epoch)

                # evaluate
                if (epoch + 1) % self.evaluate_every == 0:
                    winrate, avg_score, invalid_percent, scores = evaluate_random(DQNAgent,
                                                                                  self.model,
                                                                                  num_trials=25)
                    self.writer.add_scalar("eval_winrate", winrate, epoch)
                    self.writer.add_scalar("eval_score", avg_score, epoch)
                    self.writer.add_scalar("invalid_percentage", invalid_percent, epoch)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                    self.writer.add_scalar("epsilon", self.epsilon, epoch)

                if (epoch+1) % self.save_every == 0:
                    # save model
                    torch.save(self.model.state_dict(), f"{self.save_base_path}_{epoch}.pt")

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
        target = torch.from_numpy(reward).to(device) + self.gamma * torch.max(
            self.model.forward(next_belief), dim=1, keepdim=True)[0]
        pred = self.model.forward(belief)
        pred = torch.gather(pred, 1, torch.from_numpy(action).to(device)).to(
            device)  # convert prediction
        loss = criterion(pred, target).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
