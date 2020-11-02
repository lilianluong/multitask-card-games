import random
from typing import Dict, List, Tuple, Union

import numpy as np

from util import Card, OutOfTurnException, Suit


class TrickTakingGame:
    """
    Base gym-inspired environment for a trick taking card game.

    TrickTakingGame maintains a _state that encodes the position of every card and game
    information such
    as cards in play, the score of each player, and whose turn it is.

    All players interact with the same interface.
    An action is defined by (id, i) where id is the player's index and i is the index of the card
    they intend to play.
        - if id != next player to move, no transition happens and zero reward is returned
        - if card i is not held by player id, then a random card is played and a large negative
        reward is added
        - otherwise, player id plays card i
            - if the trick finishes, the score changes and the next player is set
            - if the trick is ongoing, card i is marked as being played and the turn moves to the
            next player

    Each reset() and step(action) returns a tuple of observations specific to each player,
    which comprises of only
    the information visible to that player. The reward is similarly a tuple of integers for each
    player.

    Trick taking games should be implemented as classes that inherit TrickTakingGame and
    implement its abstract methods.
    They may also want to override some methods that define properties (e.g. cards_per_suit) or
    rules for each game.

    The state representation is defined in reset(), while the observations are defined in
    _get_observations().
    """
    name = "Trick Taking Game"

    def __init__(self):
        self._num_cards = sum(self.cards_per_suit)
        self._state = None

    def reset(self, state: List[int] = None) -> Tuple[List[int], ...]:
        """
        Reset the state for a new game, and return initial observations.

        State representation:
            [index of player holding card i or -1 if discarded| 0 <= i < self.num_cards] +
            [index of card in play or -1 if no card yet played by player j | 0 <= j <
            self.num_players] +
            [score of player j | 0 <= j < self.num_players] +
            [trump suit number or -1, trick leading card index or -1, index of player to move next]

        :param state: the state to force the game to start at
        :return: Tuple[List[int], ...] of observations
        """
        if state is None:
            card_distribution = self._deal()
            self._state = (
                    card_distribution +
                    [-1 for _ in range(self.num_players)] +
                    [0 for _ in range(self.num_players)] +
                    [self._get_trump_suit(), -1, self._get_first_player(card_distribution)]
            )
        else:
            self._state = state
        assert len(
            self._state) == self.num_cards + 2 * self.num_players + 3, "state was reset to the " \
                                                                       "wrong size"
        return self._get_observations()

    def step(self, action: Tuple[int, int]) -> Tuple[
        Tuple[List[int], ...], Tuple[int, ...], bool, Dict]:
        """
        Execute action according to the rules defined in the docstring of TrickTakingGame.

        :param action: Tuple[int, int], (id, i) representing player id playing card i
        :return: Tuple of the following:
            - observation, Tuple[List[int], ...] of observations
            - reward, Tuple[int, ...] of rewards
            - done, bool that is True if the game has ended, otherwise False
            - info, Dict of diagnostic information, currently empty
        """
        assert len(action) == 2, "invalid action"
        player, card_index = action
        assert card_index < self.num_cards, "Trying to pick card with index higher than allowed"
        num_cards = self.num_cards
        num_players = self.num_players

        rewards = [0 for _ in range(num_players)]

        # Check if it is this player's turn
        if player != self.next_player:
            return self._get_observations(), tuple(rewards), False, {"error": OutOfTurnException}

        # Check if the card is a valid play
        invalid_plays = {}
        if not self.is_valid_play(player, card_index):
            valid_cards = [i for i in range(num_cards) if self.is_valid_play(player, i)]
            # if self._state[card_index] == player:
            #     rewards[player] -= 10
            # else:
            rewards[player] -= 50  # Huge penalty for picking an invalid card!
            card_index = random.choice(valid_cards)  # Choose random valid move to play
            invalid_plays[player] = "invalid"
        # else:
            # rewards[player] += 20

        # Play the card
        self._state[card_index] = -1
        assert self._state[num_cards + player] == -1, "Trying to play in a trick already played in"
        self._state[num_cards + player] = card_index
        if self._state[-2] == -1:
            # Trick leader
            self._state[-2] = card_index
        # update next player
        self._state[-1] = (player + 1) % num_players

        # Check if trick completed
        played_cards = self._state[num_cards: num_cards + num_players]
        if -1 not in played_cards:
            # Handle rewards
            trick_rewards, next_leader = self._end_trick()
            total_rewards = sum(trick_rewards)
            rewards = [rewards[i] + 1.3 * trick_rewards[i] - 0.3 * total_rewards for i in range(num_players)]
            for i in range(num_players):
                offset = num_cards + num_players  # index into state correctly
                self._state[offset + i] += trick_rewards[i]  # update current scores

            # Reset trick
            for i in range(num_cards, num_cards + num_players):
                self._state[i] = -1
            self._state[-2] = -1
            self._state[-1] = next_leader

        # Check if game ended
        if sum(self._state[:num_cards]) == -num_cards:
            done = True
            # apply score bonuses
            bonus_rewards = self._end_game_bonuses()
            rewards = [rewards[i] + bonus_rewards[i] for i in range(num_players)]
            for i in range(num_players):
                offset = num_cards + num_players
                self._state[offset + i] += bonus_rewards[i]
        else:
            done = False

        return self._get_observations(), tuple(rewards), done, invalid_plays

    def render(self, mode: str = "human", view: int = -1):
        """
        Render the state of the game.

        :param mode: str
        :param view: int, render whole state if -1, otherwise render observation of agent view
        :return: None
        """
        assert mode == "human", "invalid mode"
        if view == -1 or not 0 <= view < self.num_players:
            print(self._state)
        else:
            print(self._get_observations()[view])

    def _deal(self) -> List[int]:
        """
        Deal cards evenly to each player, and return the positions of the cards as included in
        the state
        :return: List[int], where the i^th element is the index of the player who holds card i
        """
        assert self.num_cards % self.num_players == 0, "cards cannot be evenly divided among the " \
                                                       "players"
        cards = []
        for i in range(self.num_players):
            cards += [i for _ in range(self.num_cards // self.num_players)]
        random.shuffle(cards)
        assert len(cards) == self.num_cards, "wrong number of cards dealt"
        return cards

    def _get_hands(self) -> Dict[int, List[int]]:
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

    def _get_observations(self) -> Tuple[List[int], ...]:
        """
        Extract visible information for each player to create vector of observations.

        An observation is structured as follows:
            [0 if card i not in hand, 1 if in hand, -1 if discarded | 0 <= i < self.num_cards] +
            [index of card in play or -1 if no card yet played by player j | 0 <= j <
            self.num_players] +
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

    def _end_game_bonuses(self) -> List[int]:
        """
        Computes additional reward assigned to each player at the end of a game.
        May be overwritten by child classes.
        :return: vector of bonus rewards for each player
        """
        rewards = [0 for _ in range(self.num_players)]
        return rewards

    def _end_trick(self) -> Tuple[List[int], int]:
        """
        Determine the rewards of a completed trick, and choose the player to start the next trick.
        Should probably be overwritten by a child class.
        :return: Tuple, of a vector of rewards for the current trick and the index of the next
        player to start
        """
        winning_player, winning_card = self._get_trick_winner()
        rewards = [0 for _ in range(self.num_players)]
        rewards[winning_player] = 1
        return rewards, winning_player

    def _get_trick_winner(self) -> Tuple[int, Card]:
        """
        Determine the winning player and card of a completed trick
        :return: Tuple[int, Card], the index of the winning player and their played Card
        """
        trump_suit = self.trump_suit
        starting_card = self.trick_leader
        played_cards = [self.index_to_card(self._state[self.num_cards + i]) for i in
                        range(self.num_players)]

        winning_index = -1
        winning_card = starting_card
        for player_index, card in enumerate(played_cards):
            if (card.suit == winning_card.suit and card.value >= winning_card.value) or \
                    (card.suit == trump_suit and winning_card.suit != trump_suit):
                winning_index = player_index
                winning_card = card

        return winning_index, winning_card

    # noinspection PyMethodMayBeStatic
    def _get_trump_suit(self) -> int:
        """
        Determine the trump suit.
        :return: int, -1 if there is no trump else the numerical index of the suit
        """
        return random.randint(0, 3)

    # noinspection PyMethodMayBeStatic
    def _get_first_player(self, card_distribution: List[int]) -> int:
        """
        :param card_distribution: part of the state that shows the card positions, also the first
        output of _deal()
        :return: int, index of the player who gets the first turn
        """
        return 0

    def is_valid_play(self, player_index, card_index) -> bool:
        """
        Determines if a proposed card play is valid
        :param player_index: player making the move
        :param card_index: proposed card to be played
        :return: True if the card play is valid, else False
        """
        if self._state[card_index] != player_index:
            return False

        # Check if player is empty of the starting suit if different suit was played
        played_card = self.index_to_card(card_index)
        starting_card = self.trick_leader
        if starting_card is None:
            return True
        if played_card.suit != starting_card.suit:
            for i in range(self.num_cards):
                if self._state[i] == player_index and self.index_to_card(
                        i).suit == starting_card.suit:
                    return False

        return True

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
        return self._num_cards

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

    @property
    def scores(self) -> List[int]:
        """
        :return: list where the i^th integer is the score of player i
        """
        return self._state[self.num_cards + self.num_players: self.num_cards + 2 * self.num_players]

    @property
    def trump_suit(self) -> Suit:
        """
        :return: the trump suit
        """
        return Suit(self._state[-3])

    @property
    def trick_leader(self) -> Union[Card, None]:
        """
        :return: the card that was first played in the trick, or None if one hasn't yet been played
        """
        card_index = self._state[-2]
        if card_index == -1:
            return None
        else:
            return self.index_to_card(card_index)

    def current_trick(self) -> Dict[int, Card]:
        """

        :return: dictionary mapping player ids to card they have played in this trick
        """
        card_list = self._state[self.num_cards:self.num_cards + self.num_players]
        trick_dict = {i: self.index_to_card(el) for i, el in enumerate(card_list) if el != -1}
        return trick_dict

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
