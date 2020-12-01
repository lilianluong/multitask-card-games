#Yeah probably just a greedy win trick agent and a greedy lose trick agent should be good
import abc
from typing import List, Set, Tuple, Union
from agents.random_agent import RandomAgent
from environments.trick_taking_game import TrickTakingGame
from util import Card

class GreedyAgent(RandomAgent):

	def __init__(self, game: TrickTakingGame, player_number: int, win_trick: bool=True):
		super().__init__(game, player_number)
		self._win_trick = win_trick

	def geq(self, card1: Card, card2: Card):
		return (card1.suit.value > card2.suit.value) or (card1.suit.value == card2.suit.value and card1.value > card2.value)

	def win_trick(self, valid_cards):
		card = None
		for valid_card in valid_cards:
			if card is None:
				card = valid_card
			elif self.geq(valid_card, card):
				card = valid_card
		return card

	def lose_trick(self, valid_cards):
		card = None
		for valid_card in valid_cards:
			if card is None:
				card = valid_card
			elif not self.geq(valid_card, card):
				card = valid_card

		return card

	def act(self, epsilon: float = 0) -> Card:
		valid_cards = self._get_hand(self._current_observation, valid_only=True)
		if self._win_trick:
			return self.win_trick(valid_cards)
		return self.lose_trick(valid_cards)

class GreedyTwentyFiveAgent(GreedyAgent): #for twentyfive + variants 

	def geq(self, card1: Card, card2: Card):
		if card1.suit.value == self._game.get_trump_suit():
			if card2.suit.value != self._game.get_trump_suit():
				return True
			elif (card1.value == 5) or (card1.value == 11 and card2.value != 5) or (card1 == Card(Suit.HEARTS,
                                      self._game._num_cards - 1) and card2.value not in {5, 11}):
				return True
		elif card1.suit.value > card2.suit.value or (card1.suit.value == card2.suit.value and card1.value > card2.value):
			return True
		return False

class GreedyHeartsAgent(GreedyAgent): #for hearts + variants

	def win_trick(self, valid_cards):
		starting_card = self._game.trick_leader
		if starting_card is None:
			return max(valid_cards, key=lambda x:x.value)
		else:
			card = None
			for valid_card in valid_cards:
				if valid_card.suit == starting_card.suit and (card is None or self.geq(valid_card, card)):
					card = valid_card
			if card is None:
				return max(valid_cards, key=lambda x:x.value)
			return card

