import torch

from enum import Enum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Suit(Enum):
    """Card suits."""
    NO_TRUMP = -1
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Card:
    """An immutable card."""

    def __init__(self, suit: Suit, value: int):
        self._suit = suit
        self._value = value

    @property
    def suit(self) -> Suit:
        return self._suit

    @property
    def value(self) -> int:
        return self._value

    def __eq__(self, other):
        return self.suit == other.suit and self.value == other.value

    def __hash__(self):
        return hash((self.suit, self.value))

    def __repr__(self):
        return "Card<{}, {}>".format(self.suit, self.value)

    def __str__(self):
        return "{} {}".format(self.suit, self.value)


class OutOfTurnException(Exception):
    pass


def polynomial_transform(x: torch.FloatTensor) -> torch.FloatTensor:
    """
    Return x raised to the second order polynomial basis, used for transforming an NN input.
    :param x: tensor to transform
    :returns: transformed tensor
    """
    n, d = x.shape
    x1 = torch.unsqueeze(torch.cat([torch.ones((n, 1)).to(device), x], dim=1), 1)
    x = torch.unsqueeze(x, 2) * x1
    return x.reshape(n, d * (d + 1))
