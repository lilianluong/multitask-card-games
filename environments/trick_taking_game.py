import random
from typing import Dict, List, Tuple

import numpy as np

from environments.util import Card, Suit


class TrickTakingGame:
    """
    Base gym-inspired environment for a trick taking card game.

    TrickTakingGame maintains a _state that encodes the position of every card and game information such
    as cards in play, the score of each player, and whose turn it is.

    All players interact with the same interface.
    An action is defined by (id, i) where id is the player's index and i is the index of the card they intend to play.
      - if id != next player to move, no transition happens and zero reward is returned
      - if card i is not held by player id, then a random card is played and a large negative reward is added
      - otherwise, player id plays card i
        - if the trick finishes, the score changes and the next player is set
        - if the trick is ongoing, card i is marked as being played and the turn moves to the next player

    Each reset() and step(action) returns a tuple of observations specific to each player, which comprises of only
    the information visible to that player. The reward is similarly a tuple of integers for each player.

    Trick taking games should be implemented as classes that inherit TrickTakingGame and implement its abstract methods.
    They may also want to override some methods that define properties (e.g. cards_per_suit) or rules for each game.

    The state representation is defined in reset(), while the observations are defined in _get_observations().
    """

    def __init__(self):
        self._state = None

    def reset(self):
        """
        Reset the state for a new game, and return initial observations.

        State representation:
            [index of player holding card i or -1 if discarded| 0 <= i < self.num_cards] +
            [index of card in play or -1 if no card yet played by player j | 0 <= j < self.num_players] +
            [score of player j | 0 <= j < self.num_players] +
            [index of player to move next]

        :return: Tuple[List[int, ...], ...] of observations
        """
        card_distribution = self._deal()
        self._state = (
            card_distribution +
            [-1 for _ in range(self.num_players)] +
            [0 for _ in range(self.num_players)] +
            [self._get_first_player(card_distribution)]
        )
        assert len(self._state) == self.num_cards + 2 * self.num_players + 1, "state was reset to the wrong size"
        return self._get_observations()

    def step(self, action):
        """
        Execute action according to the rules defined in the docstring of TrickTakingGame.

        :param action: Tuple[int, int], (id, i) representing player id playing card i
        :return: Tuple of the following:
          - observation, Tuple[List[int, ...], ...] of observations
          - reward, Tuple[int, ...] of rewards
          - done, bool that is True if the game has ended, otherwise False
          - info, Dict of diagnostic information, currently empty
        """
        raise NotImplementedError

    def render(self, mode="world", view: int = -1):
        """
        Render the state of the game.

        :param mode: str
        :param view: int, render whole state if -1, otherwise render observation of agent view
        :return: None
        """
        raise NotImplementedError

    def _deal(self) -> List[int, ...]:
        """
        Deal cards evenly to each player, and return the positions of the cards as included in the state
        :return: List[int, ...], where the i^th element is the index of the player who holds card i
        """
        assert self.num_cards % self.num_players == 0, "cards cannot be evenly divided among the players"
        cards = []
        for i in range(self.num_players):
            cards += [i for _ in range(self.num_cards // self.num_players)]
        random.shuffle(cards)
        assert len(cards) == self.num_cards, "wrong number of cards dealt"
        return cards

    def _get_hands(self) -> Dict[int, List[int, ...]]:
        """
        Return the cards possessed by each player.
        :return: Dict, mapping i to a list of the sorted card indices in player i's hand
        """
        hands = {i: [] for i in range(self.num_players)}
        for card_index in range(self.num_cards):
            holding_player = self._state[card_index]
            if holding_player != -1:
                hands[holding_player].append(card_index)
        return hands

    def _get_observations(self) -> Tuple[List[int, ...], ...]:
        """
        Extract visible information for each player to create vector of observations.

        An observation is structured as follows:
            [0 if card i not in hand, 1 if in hand, 2 if discarded | 0 <= i < self.num_cards] +
            [index of card in play or -1 if no card yet played by player j | 0 <= j < self.num_players] +
            [score of player j | 0 <= j < self.num_players] +
            [index of player to move next]

        :return: Tuple, where the i^th element is a fixed-length list, the observation for player i
        """
        cards, public_info = self._state[:self.num_cards], self._state[self.num_cards:]
        observations = tuple([-1 if x == -1 else 0 for x in cards] for _ in range(self.num_players))

        for card_index, card_position in enumerate(cards):
            if card_position != -1:
                observations[card_position][card_index] = 1

        for i in range(self.num_players):
            observations[i].extend(public_info[:])

        return observations

    # Rule-related methods

    # noinspection PyMethodMayBeStatic
    def _get_first_player(self, card_distribution: List[int, ...]) -> int:
        """
        :param card_distribution: part of the state that shows the card positions, also the first output of _deal()
        :return: int, index of the player who gets the first turn
        """
        return 0

    @property
    def cards_per_suit(self) -> Tuple[int, ...]:
        """
        Defines the number of cards for each suit the game is played with.
        Override for children classes.
        :return: Tuple[int], where the i^th element is the number of cards for suit i
        """
        return 6, 6, 6, 6

    @property
    def num_cards(self) -> int:
        """
        :return: int, the total number of cards in the game based on cards_per_suit()
        """
        return np.prod(self.cards_per_suit).item()

    @property
    def num_players(self) -> int:
        """
        :return: int, number of players in the game
        """
        return 4

    @property
    def next_player(self) -> int:
        """
        :return: index of player who is allowed to play the next card
        """
        return self._state[-1]

    def index_to_card(self, card_index: int) -> Card:
        """
        Converts a card index to a suit and number representing the relative strength of the card.

        :param card_index: int, 0 <= card_index < self.num_cards
        :return: Card
        """
        suit, total = 0, 0
        while total + self.cards_per_suit[suit] <= card_index:
            total += self.cards_per_suit[suit]
            suit += 1
        num = card_index - total
        assert 0 <= num < self.cards_per_suit[suit], "card value is invalid"
        return Card(suit=Suit(suit), value=num)

    def card_to_index(self, card: Card) -> int:
        """
        Converts a card to a numerical index.

        :param card: Card
        :return: int index, 0 <= index < self.num_cards
        """
        target_suit = card.suit.value
        cards_before_suit = sum(self.cards_per_suit[:target_suit])
        index = cards_before_suit + card.value
        assert 0 <= index < self.num_cards, "card index is invalid"
        return index
