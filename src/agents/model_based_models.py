import torch
from torch import nn

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
        # h1, h2, h3 = 1200, 600, 220
        h1, h2, h3 = 800, 400, 220
        d = belief_size + num_actions
        input_size = d * (d + 1)
        self._num_players = num_players
        self.model = nn.Sequential(
            nn.Linear(input_size, h1),
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

    def polynomial(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Return x raised to the second order polynomial basis
        """
        polynomial_basis = True
        if polynomial_basis:
            n, d = x.shape
            x1 = torch.unsqueeze(torch.cat([torch.ones((n, 1)).to(device), x], dim=1), 1)
            x = torch.unsqueeze(x, 2) * x1
            return x.reshape(n, d * (d + 1))
        return True


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
        # h1, h2 = 200, 40
        h1, h2 = 100, 20
        d = belief_size + num_actions
        input_size = d * (d + 1)
        self.model = nn.Sequential(
            nn.Linear(input_size, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1)
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
        Return x to a polynomial order
        """
        polynomial_reward = True
        if polynomial_reward:
            n, d = x.shape
            x1 = torch.unsqueeze(torch.cat([torch.ones((n, 1)).to(device), x], dim=1), 1)
            x = torch.unsqueeze(x, 2) * x1
            return x.reshape(n, d * (d + 1))
        else:
            return x


# class PolicyNetwork(nn.Module):
#     """
#     Multilayered perceptron policy network to imitate the expert: (b,) ->
#     """
#
#     def __init__(self, belief_size: int, num_actions: int):
#         """
#         :param belief_size: number of values in a belief
#         :param num_actions: number of possible actions, to be 1-hot encoded and attached to belief
#         """
#         super().__init__()
#         h1 = 160
#         h2 = 60
#         d = belief_size + num_actions
#         input_size = d * (d + 1)
#         self.model = nn.Sequential(
#             nn.Linear(input_size, h1),
#             nn.ReLU(inplace=True),
#             nn.Linear(h1, h2),
#             nn.ReLU(inplace=True),
#             nn.Linear(h2, 1)
#         )
#
#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         Forward pass of the model
#         :param x: a shape (batch_size, belief_size + num_actions) torch Float tensor, beliefs concatenated with actions
#         :return: a shape (batch_size, 1) torch Float tensor, the predicted reward
#         """
#         return self.model(x)
#
#     def loss(self, pred: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         Calculate the loss of a batch of predictions against the true labels
#         :param pred: (batch_size, 1) predicted rewards
#         :param y: (batch_size, 1) actual rewards
#         :return: mean loss as a torch Float scalar
#         """
#         mse_loss = nn.MSELoss()(pred, y)
#         # TODO: Regularization?
#         return mse_loss
#
#     def polynomial(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         """
#         Return x to a polynomial order
#         """
#         polynomial_reward = True
#         if polynomial_reward:
#             n, d = x.shape
#             x1 = torch.unsqueeze(torch.cat([torch.ones((n, 1)).to(device), x], dim=1), 1)
#             x = torch.unsqueeze(x, 2) * x1
#             return x.reshape(n, d * (d + 1))
#         else:
#             return x