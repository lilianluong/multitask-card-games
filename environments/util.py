from enum import Enum


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
        return self.suit, self.value

    def __repr__(self):
        return "Card<{}, {}>".format(self.suit, self.value)

    def __str__(self):
        return "{} {}".format(self.suit, self.value)


class OutOfTurnException(Exception):
    pass
