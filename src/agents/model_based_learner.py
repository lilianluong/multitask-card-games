import itertools
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agents.base import Learner
from agents.belief_agent import BeliefBasedAgent
from agents.model_based_agent import ModelBasedAgent
from agents.models.model_based_models import RewardModel, TransitionModel
from agents.models.multitask_models import MultitaskRewardModel, MultitaskTransitionModel
from environments.trick_taking_game import TrickTakingGame
from evaluators import evaluate_random
from game import Game

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelGameRunner:
    def __init__(self, agent_type, task, agent_params, game_params):
        self.agent_type = agent_type
        self.task = task
        self.agent_params = agent_params
        self.game_params = game_params

    def __call__(self, game_num):
        # print(f"Running game {game_num}")
        game = Game(self.task, [self.agent_type] * 4, self.agent_params, self.game_params)
        result = game.run()
        barbs = game.get_barbs()
        return barbs


class ModelBasedLearner(Learner):
    def __init__(self, agent: ModelBasedAgent.__class__ = ModelBasedAgent, multitask: bool = False,
                 resume_model: Dict[str, Dict[str, Dict[str, Any]]] = None,
                 model_names: Dict[str, str] = None, learner_name: str = "MBL"):
        """
        :param agent: either ModelBasedAgent or a subclass to use
        :param multitask: whether to use multitask learning or not
        :param resume_model: maps "transition" or "reward" to dictionaries:
                                map task names to dictionaries:
                                    "state": model state dict
                                    "params": list of parameters to pass into model constructor
        :param model_names: maps task names to names to save model as
        :param learner_name: name of the trial for tensorboard records
        """
        super().__init__(use_thread=False)  # TODO: Support no threading
        self._agent_type = agent
        self._model_names = model_names
        if multitask:
            self._transition_model = MultitaskTransitionModel().to(device)
            self._reward_model = MultitaskRewardModel().to(device)
        else:
            self._transition_model = TransitionModel().to(device)
            self._reward_model = RewardModel().to(device)
        self._transition_optimizer = None
        self._reward_optimizer = None

        # Load existing
        if resume_model is not None:
            for key, item in resume_model["transition"].items():
                self._transition_model.make_model(item["task"])
                self._transition_model.models[key].load_state_dict(item["state"])
            for key, item in resume_model["reward"].items():
                self._reward_model.make_model(item["task"])
                self._reward_model.models[key].load_state_dict(item["state"])

        # Hyperparameters
        self._num_epochs = 5000
        self._games_per_epoch = 20
        self._batch_size = 112

        self.epsilon = 1.0  # exploration rate, percent time to be epsilon greedy
        self.epsilon_min = 0.1  # min exploration
        self.epsilon_decay = 0.999  # to decrease exploration rate over time

        self._reward_lr = 1e-4
        self._transition_lr = 1e-4

        self.writer = SummaryWriter(
            f"runs/{learner_name}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        self.evaluate_every = 50

    def train(self, tasks: List[TrickTakingGame.__class__]):
        for task in tasks:
            self._setup_single_task(task)
        self._transition_optimizer = optim.Adam(self._transition_model.get_parameters(),
                                                lr=self._transition_lr)
        self._reward_optimizer = optim.Adam(self._reward_model.get_parameters(), lr=self._reward_lr)

        for epoch in range(self._num_epochs):
            transition_losses, reward_losses = [], []
            for task in tasks:
                transition_loss, reward_loss = self._train_single_task(task, epoch)
                transition_losses.append(transition_loss)
                reward_losses.append(reward_loss)

            self.writer.add_scalar("avg_training_transition_loss", np.mean(transition_losses),
                                   epoch)
            self.writer.add_scalar("avg_training_reward_loss", np.mean(reward_losses), epoch)

            if epoch % self.evaluate_every == 0:
                winrate, matchrate, avg_score, invalid, scores = evaluate_random(tasks,
                                                                                 self._agent_type,
                                                                                 [
                                                                                     self._transition_model,
                                                                                     self._reward_model],
                                                                                 num_trials=50,
                                                                                 compare_agent=None)  # MonteCarloAgent)
                print("Done EVAL")
                self.writer.add_scalar("eval_winrate", winrate, epoch)
                self.writer.add_scalar("eval_matchrate", matchrate, epoch)
                self.writer.add_scalar("eval_score_margin", avg_score, epoch)
                self.writer.add_scalar("invalid_percentage", invalid, epoch)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.writer.add_scalar("epsilon", self.epsilon, epoch)

        for task in tasks:
            torch.save(self._transition_model.models[task.name].state_dict(),
                       "models/transition_model_{}.pt".format(self._model_names[task.name]))
            torch.save(self._reward_model.models[task.name].state_dict(),
                       "models/reward_model_{}.pt".format(self._model_names[task.name]))

    def _setup_single_task(self, task: TrickTakingGame.__class__):
        """
        Setup models and such for a task.
        :param task: class of task to setup
        :return: None
        """
        if task.name not in self._transition_model.models:
            self._transition_model.make_model(task)
        if task.name not in self._reward_model.models:
            self._reward_model.make_model(task)

    def _train_single_task(self, task: TrickTakingGame.__class__, epoch: int) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Train a task for a single epoch, following the single-task learning framework
        :param task: class of task to train
        :param epoch: the index of the current epoch
        :return: (average transition loss, average reward loss)
        """
        if epoch % 10 == 0:
            print(f"Starting epoch {epoch}/{self._num_epochs} for {task.name}")

        experiences = self._agent_evaluation(task)
        transition_loss, reward_loss = self._train_world_models(task, experiences)

        if (epoch + 1) % 200 == 0:
            torch.save(self._transition_model.models[task.name].state_dict(),
                       "models/transition_model_temp_{}.pt".format(self._model_names[task.name]))
            torch.save(self._reward_model.models[task.name].state_dict(),
                       "models/reward_model_temp_{}.pt".format(self._model_names[task.name]))

        return np.mean(transition_loss), np.mean(reward_loss)

    def _agent_evaluation(self, task: TrickTakingGame.__class__) -> List[
        Tuple[List[int], int, int, List[int]]]:
        """
        Collect (b, a, r, b') experiences from playing self._games_per_epoch games against itself
        :param task: the class of the game to play
        :return: list of (b, a, r, b') experiences
        """
        specific_game_func = ModelGameRunner(self._agent_type, task,
                                             [{"transition_model": self._transition_model,
                                               "reward_model": self._reward_model}
                                              for _ in range(task().num_players)],
                                             {"epsilon": self.epsilon, "verbose": False})

        # barbs= self.executor.map(specific_game_func, range(self._games_per_epoch))
        barbs = [specific_game_func(i) for i in range(self._games_per_epoch)]
        # wait for completion
        barbs = list(itertools.chain.from_iterable(barbs))
        return barbs

    def _train_world_models(self, task: TrickTakingGame.__class__,
                            experiences: List[Tuple[List[int], int, int, List[int]]]) -> Tuple[
        np.ndarray, np.ndarray]:
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
        belief_size = BeliefBasedAgent(sample_task, 0).get_belief_size()
        belief_actions = np.pad(np.vstack(experience_array[:, 0]), (0, task().num_cards),
                                'constant')
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

        # Train
        for model, optimizer, targets, losses in (
                (self._transition_model, self._transition_optimizer, next_beliefs,
                 transition_losses),
                (self._reward_model, self._reward_optimizer, rewards, reward_losses),
        ):
            for i in range(0, len(experiences), self._batch_size):
                x = belief_actions[i: i + self._batch_size]
                pred = model.forward(x, task.name)
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
        pass  # currently involves no learned policy
