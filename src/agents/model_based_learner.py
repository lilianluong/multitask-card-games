"""
Model-based Agent-Learner pair.
"""

import random
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from agents.base import Agent, Learner
from agents.belief_agent import BeliefBasedAgent
from environments.trick_taking_game import TrickTakingGame
from evaluators import evaluate_random
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
        # h1 = 280
        # h2 = 240
        # h3 = 200
        # h4 = 160
        h1 = 600
        h2 = 300
        h3 = 120
        d = belief_size + num_actions
        input_size = d * (d + 1)
        self._num_players = num_players
        self.model = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h3),
            # nn.ReLU(inplace=True),
            # nn.Linear(h3, h4),
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

    def polynomial(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Return x raised to the second order polynomial basis
        """
        n, d = x.shape
        x1 = torch.unsqueeze(torch.cat([torch.ones((n, 1)).to(device), x], dim=1), 1)
        x = torch.unsqueeze(x, 2) * x1
        return x.reshape(n, d * (d + 1))


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
        h1 = 120
        h2 = 50
        h3 = 20
        # h4 = 20
        d = belief_size + num_actions
        input_size = d * (d + 1)
        self.model = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h3),
            nn.ReLU(inplace=True),
            # nn.Linear(h3, h4),
            # nn.ReLU(inplace=True),
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

    def polynomial(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Return x
        """
        n, d = x.shape
        x1 = torch.unsqueeze(torch.cat([torch.ones((n, 1)).to(device), x], dim=1), 1)
        x = torch.unsqueeze(x, 2) * x1
        return x.reshape(n, d * (d + 1))


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
            return self._game.index_to_card(random.randint(0, self._game.num_cards - 1))
            # valid_cards = self._get_hand(self._current_observation, valid_only=True)
            # return random.sample(valid_cards, 1)[0]

        # TODO: Select an action
        inputs = []
        for a in range(self._game.num_cards):
            inputs.append(self._belief + [0 for _ in range(self._game.num_cards)])
            inputs[-1][len(self._belief) + a] = 1
        action_values = self._reward_model.forward(self._reward_model.polynomial(torch.FloatTensor(inputs).to(device)))
        chosen_action = torch.argmax(action_values).item()
        return self._game.index_to_card(chosen_action)


class ModelBasedLearner(Learner):
    def __init__(self, multitask: bool = False, resume_model: Dict = None):
        """
        :param multitask: whether to use multitask learning or not
        """
        if multitask:
            raise NotImplementedError
        self._transition_models = {}
        self._transition_optimizers = {}
        self._reward_models = {}
        self._reward_optimizers = {}

        if resume_model is not None:
            for key, item in resume_model["transition"].items():
                self._transition_models[key] = TransitionModel(*item["params"]).to(device)
                self._transition_models[key].load_state_dict(item["state"])
            for key, item in resume_model["reward"].items():
                self._reward_models[key] = RewardModel(*item["params"]).to(device)
                self._reward_models[key].load_state_dict(item["state"])

        # Hyperparameters
        self._num_epochs = 5000
        self._games_per_epoch = 20
        self._batch_size = 112

        self.epsilon = 1.0  # exploration rate, percent time to be epsilon greedy
        self.epsilon_min = 0.1  # min exploration
        self.epsilon_decay = 0.999  # to decrease exploration rate over time

        self.writer = SummaryWriter(f"runs/dqn-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        self.evaluate_every = 50

    def train(self, tasks: List[TrickTakingGame.__class__]):
        for task in tasks:
            self._train_single_task(task)

    def _train_single_task(self, task: TrickTakingGame.__class__):
        sample_task = task()
        belief_size = 4 * sample_task.num_cards + sample_task.num_players + len(sample_task.cards_per_suit)
        if task.name not in self._transition_models:
            self._transition_models[task.name] = TransitionModel(belief_size,
                                                                 sample_task.num_cards,
                                                                 sample_task.num_players).to(device)
        if task.name not in self._reward_models:
            self._reward_models[task.name] = RewardModel(belief_size, sample_task.num_cards).to(device)
        self._transition_optimizers[task.name] = optim.Adam(self._transition_models[task.name].parameters())
        self._reward_optimizers[task.name] = optim.Adam(self._reward_models[task.name].parameters(), lr=1e-4)

        for epoch in range(self._num_epochs):
            if epoch % 10 == 0:
                print(f"Starting epoch {epoch}/{self._num_epochs}")
            experiences = self._agent_evaluation(task)
            transition_loss, reward_loss = self._train_world_models(task, experiences)
            self._train_agent_policy(task)

            self.writer.add_scalar("avg_training_transition_loss", np.mean(transition_loss), epoch)
            self.writer.add_scalar("avg_training_reward_loss", np.mean(reward_loss), epoch)

            if epoch % self.evaluate_every == 0:
                winrate, avg_score, invalid_percent, scores = evaluate_random(ModelBasedAgent,
                                                                              [self._transition_models[task.name],
                                                                               self._reward_models[task.name]],
                                                                              num_trials=25)
                self.writer.add_scalar("eval_winrate", winrate, epoch)
                self.writer.add_scalar("eval_score", avg_score, epoch)
                self.writer.add_scalar("invalid_percentage", invalid_percent, epoch)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.writer.add_scalar("epsilon", self.epsilon, epoch)

            if (epoch + 1) % 200 == 0:
                torch.save(self._transition_models[task.name].state_dict(), "models/transition_model_temp4.pt")
                torch.save(self._reward_models[task.name].state_dict(), "models/reward_model_temp4.pt")

        torch.save(self._transition_models[task.name].state_dict(), "models/transition_model3.pt")
        torch.save(self._reward_models[task.name].state_dict(), "models/reward_model3.pt")

    def _agent_evaluation(self, task: TrickTakingGame.__class__) -> List[Tuple[List[int], int, int, List[int]]]:
        """
        Collect (b, a, r, b') experiences from playing self._games_per_epoch games against itself
        :param task: the class of the game to play
        :return: list of (b, a, r, b') experiences
        """
        barbs = []
        for game_num in range(self._games_per_epoch):
            game = Game(task, [ModelBasedAgent] * 4, [{"transition_model": self._transition_models[task.name],
                                                       "reward_model": self._reward_models[task.name]}
                                                      for _ in range(task().num_players)],
                        {"epsilon": self.epsilon, "verbose": False})
            game.run()  # result = game.run()
            barbs.extend(game.get_barbs())
        return barbs

    def _train_world_models(self, task: TrickTakingGame.__class__,
                            experiences: List[Tuple[List[int], int, int, List[int]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the transition and reward models on the experiences
        :param task: the class of the game to train models for
        :param experiences: list of (b, a, r, b') experiences as returned by _agent_evaluation
        :return: (transition loss mean, reward loss mean)
        """
        transition_losses, reward_losses = [], []

        # Construct belief_action input matrices
        experience_array = np.asarray(experiences, dtype=object)
        sample_task = task()
        belief_size = 4 * sample_task.num_cards + sample_task.num_players + len(sample_task.cards_per_suit)
        belief_actions = np.pad(np.vstack(experience_array[:, 0]), (0, task().num_cards), 'constant')
        actions = experience_array[:, 1].astype(np.int)
        indices = np.arange(len(experiences))
        belief_actions[indices, actions + belief_size] = 1

        rewards = np.vstack(experience_array[:, 2])
        next_beliefs = np.vstack(experience_array[:, 3])

        # Shuffle data
        np.random.shuffle(indices)
        belief_actions = torch.from_numpy(belief_actions[indices]).float().to(device)
        rewards = torch.from_numpy(rewards[indices]).float().to(device)
        next_beliefs = torch.from_numpy(next_beliefs[indices]).float().to(device)

        for model_dict, optim_dict, targets, losses in (
                (self._transition_models, self._transition_optimizers, next_beliefs, transition_losses),
                (self._reward_models, self._reward_optimizers, rewards, reward_losses)
        ):
            model = model_dict[task.name]
            optimizer = optim_dict[task.name]
            for i in range(0, len(experiences), self._batch_size):
                x = belief_actions[i: i + self._batch_size]
                x = model.polynomial(x)
                pred = model.forward(x)
                y = targets[i: i + self._batch_size]
                loss = model.loss(pred, y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return np.mean(transition_losses), np.mean(reward_losses)

    def _train_agent_policy(self, task: TrickTakingGame.__class__):
        """
        Train a policy for the agent. TODO: What should this involve?
        :param task: the class of the game to train for
        :return: None
        """
        pass  # currently using naive one-step lookahead instead of learned policy via expert iteration

    def initialize_agent(self, game: TrickTakingGame, player_number: int) -> Agent:
        return ModelBasedAgent(game, player_number, self._transition_models[game.name], self._reward_models[game.name])


if __name__ == "__main__":
    from environments.test_hearts import TestSimpleHearts
    from agents.random_agent import RandomAgent
    transition_state = torch.load("../models/transition_model_temp3.pt")
    reward_state = torch.load("../models/reward_model_temp3.pt")
    resume = {"transition": {"Test Simple Hearts": {"state": transition_state, "params": [104, 24, 4]}}, "reward": {"Test Simple Hearts": {"state": reward_state, "params": [104, 24]}}}
    learner = ModelBasedLearner(resume_model = resume)
    def get_game():
        game = Game(TestSimpleHearts, [ModelBasedAgent, RandomAgent, RandomAgent, RandomAgent], [
            {"transition_model": learner._transition_models["Test Simple Hearts"],
             "reward_model": learner._reward_models["Test Simple Hearts"]}, {}, {}, {}], {"epsilon": 0, "verbose": False})
        game.run()
        return game.get_info()