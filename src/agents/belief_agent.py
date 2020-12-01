import abc
from typing import List, Set, Tuple, Union

from agents.base import Agent
from environments.trick_taking_game import TrickTakingGame
from util import Card, Suit


class BeliefBasedAgent(Agent):
    """
    Abstract base class for an AI agent that plays a trick taking game, and
    that maintains a belief of the state instead of just the current observation.
    """

    def __init__(self, game: TrickTakingGame, player_number: int):
        super().__init__(game, player_number)
        self._last_belief = None
        self._last_action = None
        self._last_reward = None
        self._running_reward = 0
        self._running_action = None
        self._belief = None
        self._player_belief = None
        self._player_action = None

    def observe(self, action: Tuple[int, int], observation: List[int], reward: int):
        """
        Handle an observation from the environment, and update any personal records/current belief/etc.

        :param action: tuple of the player who moved and the index of the card they played
        :param observation: the observation corresponding to this player as returned by the env
        :param reward: an integral reward corresponding to this player as returned by the env
        :return: None
        """
        if action[0] == self._player:
            self._last_belief = self._player_belief
            self._last_action = self._player_action
            self._last_reward = self._running_reward
            self._running_reward = 0
            self._player_belief = self._belief
            self._player_action = action
        self._running_action = action
        self._running_reward += reward if reward else 0
        self._belief = self.update_belief(observation)

    def barb(self) -> Union[None, Tuple[List[int], int, int, List[int]]]:
        """
        Return an experience if one exists.
        :return: (old belief, action, reward, new belief) experience if the last action taken was by this agent,
                 else None
        """
        if self._running_action[0] == self._player:
            try:
                return self._last_belief[:], self._last_action[1], self._last_reward, self._player_belief[:]
            except TypeError:
                return None
        return None

    @abc.abstractmethod
    def act(self, epsilon: float = 0) -> Card:
        """
        Based on the current observation/belief/known state, select a Card to play.
        :return: the card to play
        """
        pass

    def get_belief_size(self) -> int:
        """
        :return: the number of elements in the belief
        """
        return 4 * self._game.num_cards + self._game.num_players
        # return self._game.num_cards * 4 + self._game.num_players + len(self._game.cards_per_suit)

    def update_belief(self, observation: List[int]) -> List[int]:
        """
        Updates the current belief based on the observation
        Should NOT mutate the current belief, the belief is reassigned in self.observe(...)

        Invariant: self._belief should only ever be set to the output of _update_belief

        :param observation: a given observation, returned from the env
        :return: an updated belief that takes into account all information to summarize the agent's knowledge, suitable
                 for input into a NN
        """
        num_cards, num_players = self._game.num_cards, self._game.num_players

        # valid cards in hand (BINARY) +
        # trump cards in hand (BINARY) +
        # cards discarded (BINARY) +
        # cards in play (BINARY) +
        # my score (LINEAR) + everyone else's score (LINEAR)
        # = belief(4 * num_cards + num_players)

        # Valid cards
        belief = [1 if observation[i] == 1 and self._game.is_valid_play(self._player, i) else 0
                  for i in range(num_cards)]

        # Trump cards
        if observation[-3] != -1:
            trump_suit = self._game.trump_suit
            total = 0
            for suit_index, suit_cards in enumerate(self._game.cards_per_suit):
                if Suit(suit_index) == trump_suit:
                    belief.extend([1 if observation[total + i] else 0 for i in range(suit_cards)])
                else:
                    belief.extend([0 for _ in range(suit_cards)])
                total += suit_cards
        else:
            belief.extend([0 for _ in range(num_cards)])
        assert len(belief) == 2 * num_cards, len(belief)

        # Cards discarded
        belief.extend([1 if observation[i] == -1 else 0 for i in range(num_cards)])

        # Cards in play
        belief.extend([0 for _ in range(num_cards)])
        for card_index in observation[num_cards: num_cards + num_players]:
            if card_index != -1:
                belief[card_index - num_cards] = 1

        # Own score
        belief.append(observation[num_cards + num_players + self._player])
        # Other scores
        belief.extend([observation[num_cards + num_players + i] for i in range(num_players) if i != self._player])

        # # cards in hand (BINARY) + cards discarded (BINARY)
        # # + cards in play (BINARY) [maybe 1-HOT instead? but more variables]
        # # + trick leader (1-HOT or all zeros) + trump suit (1-HOT)
        # # + score (LINEAR)
        # belief = [0 for _ in range(num_cards * 4 + len(self._game.cards_per_suit))]
        #
        # # Cards in hand / discarded
        # for card_index in range(num_cards):
        #     if observation[card_index] == 1:
        #         belief[card_index] = 1
        #     elif observation[card_index] == -1:
        #         belief[num_cards + card_index] = 1
        # # Cards in play
        # for card_index in observation[num_cards: num_cards + num_players]:
        #     if card_index != -1:
        #         belief[2 * num_cards + card_index] = 1
        # # Trick leader
        # if observation[-2] != -1:
        #     belief[3 * num_cards + observation[-2]] = 1
        # # Trump suit
        # if observation[-3] != -1:
        #     belief[4 * num_cards + observation[-3]] = 1
        # # Scores
        # belief.extend(observation[num_cards + num_players: num_cards + 2 * num_players])
        return belief

    def _get_hand(self, observation: List[int], valid_only: bool = False) -> Set[Card]:
        """
        Get the hand of an agent based on an observation.
        :param observation: observation corresponding to this player as returned by the env
        :param valid_only: True if only valid card plays should be returned, False if entire hand should be returned
        :return: the set of cards in the player's hand
        """
        cards = observation[:self._game.num_cards]
        return set(self._game.index_to_card(i) for i, in_hand in enumerate(cards)
                   if in_hand and (not valid_only or self._game.is_valid_play(self._player, i)))