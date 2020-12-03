from typing import List

import torch
from torch import nn

from agents.belief_agent import BeliefBasedAgent
from environments.trick_taking_game import TrickTakingGame
from util import polynomial_transform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultitaskTransitionModel(nn.Module):
    """
    Multilayered perceptron to approximate T: (b, a) -> b'
    """
    def __init__(self, layer_sizes: List[int] = None, shared_layers: int = 2, polynomial: bool = True):
        """
        :param layer_sizes: sizes of the hidden layers in the network (there will be len(layer_sizes) + 1 Linear layers)
        :param shared_layers: number of layers to maintain as a shared backbone
        :param polynomial: whether or not to use the polynomial basis
        """
        super().__init__()

        if layer_sizes is None:
            # Default layer sizes
            # layer_sizes = [400, 250, 150]
            layer_sizes = [1200, 600, 220]
        self._layer_sizes = layer_sizes
        self._num_shared_layers = shared_layers
        assert 0 < shared_layers <= len(self._layer_sizes)

        self._polynomial = polynomial
        self._input_size = None
        self._belief_size = None
        self._num_players = None
        self._parameters_returned = False
        self.models = {}
        self.backbone = None

    def setup(self, task_instance: TrickTakingGame):
        """
        Set parameters for task, and create shared backbone for model
        :param task_instance: instance of task to sample parameters from
        :return: None
        """
        num_actions = task_instance.num_cards
        num_players = task_instance.num_players
        belief_size = BeliefBasedAgent(task_instance, 0).get_belief_size()
        d = belief_size + num_actions
        self._input_size = d * (d + 1) if self._polynomial else d
        self._belief_size = belief_size
        self._num_players = num_players

        # Create shared backbone
        layers = []
        layer_input = self._input_size
        for layer_output in self._layer_sizes[:self._num_shared_layers]:
            layers.append(nn.Linear(layer_input, layer_output))
            layer_input = layer_output
        self.backbone = nn.Sequential(*layers).to(device)

    def get_parameters(self) -> List:
        """
        :return: list of the parameters of all the models for use by an optimizer
        """
        self._parameters_returned = True
        params = list(self.backbone.parameters())
        for model in self.models.values():
            params.extend(list(model.parameters()))
        return params

    def make_model(self, task: TrickTakingGame.__class__):
        """
        Creates a model for a task.
        :param task: class of the task for which a model should be created
        :returns: None
        """
        assert not self._parameters_returned, "Optimizer has already been initialized, this model would not train"
        game_instance = task()
        if self._input_size:
            assert BeliefBasedAgent(game_instance, 0).get_belief_size() == self._belief_size
            assert game_instance.num_players == self._num_players
        else:
            self.setup(game_instance)
        layers = []
        input_size = ([self._input_size] + self._layer_sizes)[self._num_shared_layers]
        for layer_size in self._layer_sizes[self._num_shared_layers:]:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU(inplace=True))
            input_size = layer_size
        layers.append(nn.Linear(input_size, self._belief_size))
        self.models[task.name] = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.FloatTensor, task: str) -> torch.FloatTensor:
        """
        Forward pass of the model
        :param x: a shape (batch_size, belief_size + num_actions) torch Float tensor, beliefs concatenated with actions
        :param task: the name of the task of which the model should be used
        :return: a shape (batch_size, belief_size) torch Float tensor, the predicted next belief
        """
        if self._polynomial:
            x = polynomial_transform(x)
        fc_out = self.models[task](self.backbone(x))
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
        return bce_loss + mse_loss


class MultitaskRewardModel(nn.Module):
    """
    Multilayered perceptron to approximate T: (b, a) -> r
    """

    def __init__(self, layer_sizes: List[int] = None, shared_layers: int = 2, polynomial: bool = True):
        """
        :param layer_sizes: sizes of the hidden layers in the network (there will be len(layer_sizes) + 1 Linear layers)
        :param shared_layers: number of layers to maintain as a shared backbone
        :param polynomial: whether or not to use the polynomial basis
        """
        super().__init__()

        if layer_sizes is None:
            # Default layer sizes
            # layer_sizes = [100, 50, 20]
            layer_sizes = [200, 40]
        self._layer_sizes = layer_sizes
        self._num_shared_layers = shared_layers
        assert 0 < shared_layers <= len(self._layer_sizes)

        self._polynomial = polynomial
        self._input_size = None
        self._belief_size = None
        self._parameters_returned = False
        self.models = {}
        self.backbone = None

    def setup(self, task_instance: TrickTakingGame):
        """
        Set parameters for task
        :param task_instance: instance of task to sample parameters from
        :return: None
        """
        num_actions = task_instance.num_cards
        belief_size = BeliefBasedAgent(task_instance, 0).get_belief_size()
        d = belief_size + num_actions
        self._input_size = d * (d + 1) if self._polynomial else d
        self._belief_size = belief_size

        # Create shared backbone
        layers = []
        layer_input = self._input_size
        for layer_output in self._layer_sizes[:self._num_shared_layers]:
            layers.append(nn.Linear(layer_input, layer_output))
            layer_input = layer_output
        self.backbone = nn.Sequential(*layers).to(device)

    def get_parameters(self) -> List:
        """
        :return: list of the parameters of all the models for use by an optimizer
        """
        self._parameters_returned = True
        params = list(self.backbone.parameters())
        for model in self.models.values():
            params.extend(list(model.parameters()))
        return params

    def make_model(self, task: TrickTakingGame.__class__):
        """
        Creates a model for a task.
        :param task: class of the task for which a model should be created
        :returns: None
        """
        assert not self._parameters_returned, "Optimizer has already been initialized, this model would not train"
        game_instance = task()
        if self._input_size:
            assert BeliefBasedAgent(game_instance, 0).get_belief_size() == self._belief_size
        else:
            self.setup(game_instance)
        layers = []
        input_size = ([self._input_size] + self._layer_sizes)[self._num_shared_layers]
        for layer_size in self._layer_sizes[self._num_shared_layers:]:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU(inplace=True))
            input_size = layer_size
        layers.append(nn.Linear(input_size, 1))
        self.models[task.name] = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.FloatTensor, task: str) -> torch.FloatTensor:
        """
        Forward pass of the model
        :param x: a shape (batch_size, belief_size + num_actions) torch Float tensor, beliefs concatenated with actions
        :param task: the name of the task of which the model should be used
        :return: a shape (batch_size, 1) torch Float tensor, the predicted reward
        """
        if self._polynomial:
            x = polynomial_transform(x)
        return self.models[task](self.backbone(x))

    def loss(self, pred: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculate the loss of a batch of predictions against the true labels
        :param pred: (batch_size, 1) predicted rewards
        :param y: (batch_size, 1) actual rewards
        :return: mean loss as a torch Float scalar
        """
        mse_loss = nn.MSELoss()(pred, y)
        return mse_loss


class MultitaskApprenticeModel(nn.Module):
    """
    Multilayered perceptron to approximate PI: b -> a according to the expert MCTS policy
    """

    def __init__(self, layer_sizes: List[int] = None, shared_layers: int = 1, polynomial: bool = True):
        """
        :param layer_sizes: sizes of the hidden layers in the network (there will be len(layer_sizes) + 1 Linear layers)
        :param shared_layers: number of layers to maintain as a shared backbone
        :param polynomial: whether or not to use the polynomial basis
        """
        super().__init__()

        if layer_sizes is None:
            # Default layer sizes
            # layer_sizes = [140, 80, 50]
            layer_sizes = [600, 300, 110]
        self._layer_sizes = layer_sizes
        self._num_shared_layers = shared_layers
        assert 0 < shared_layers <= len(self._layer_sizes)

        self._polynomial = polynomial
        self._input_size = None
        self._belief_size = None
        self._parameters_returned = False
        self.models = {}
        self.backbone = None

    def setup(self, task_instance: TrickTakingGame):
        """
        Set parameters for task
        :param task_instance: instance of task to sample parameters from
        :return: None
        """
        num_actions = task_instance.num_cards
        belief_size = BeliefBasedAgent(task_instance, 0).get_belief_size()
        d = belief_size
        self._input_size = d * (d + 1) if self._polynomial else d
        self._belief_size = belief_size

        # Create shared backbone
        layers = []
        layer_input = self._input_size
        for layer_output in self._layer_sizes[:self._num_shared_layers]:
            layers.append(nn.Linear(layer_input, layer_output))
            layer_input = layer_output
        self.backbone = nn.Sequential(*layers).to(device)

    def get_parameters(self) -> List:
        """
        :return: list of the parameters of all the models for use by an optimizer
        """
        self._parameters_returned = True
        params = list(self.backbone.parameters())
        for model in self.models.values():
            params.extend(list(model.parameters()))
        return params

    def make_model(self, task: TrickTakingGame.__class__):
        """
        Creates a model for a task.
        :param task: class of the task for which a model should be created
        :returns: None
        """
        assert not self._parameters_returned, "Optimizer has already been initialized, this model would not train"
        game_instance = task()
        if self._input_size:
            assert BeliefBasedAgent(game_instance, 0).get_belief_size() == self._belief_size
        else:
            self.setup(game_instance)
        layers = []
        input_size = ([self._input_size] + self._layer_sizes)[self._num_shared_layers]
        for layer_size in self._layer_sizes[self._num_shared_layers:]:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU(inplace=True))
            input_size = layer_size
        layers.append(nn.Linear(input_size, game_instance.num_cards))
        self.models[task.name] = nn.Sequential(*layers).to(device)

    def forward(self, x: torch.FloatTensor, task: str) -> torch.FloatTensor:
        """
        Forward pass of the model
        :param x: a shape (batch_size, belief_size) torch Float tensor, beliefs
        :param task: the name of the task of which the model should be used
        :return: a shape (batch_size, num_actions) torch Float tensor, the predicted action scores
        """
        if self._polynomial:
            x = polynomial_transform(x)
        return self.models[task](self.backbone(x))

    def loss(self, pred: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        """
        Calculate the loss of a batch of predictions against the true labels
        :param pred: (batch_size, num_actions) predicted action scores
        :param y: (batch_size, 1) expert action selections
        :return: cross entropy loss as a torch Float scalar
        """
        loss = nn.CrossEntropyLoss()(pred, y)
        return loss
