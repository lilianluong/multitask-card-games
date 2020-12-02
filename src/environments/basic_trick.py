import random
from typing import Dict, List, Tuple, Union

from environments.trick_taking_game import TrickTakingGame
from util import Card, Suit, OutOfTurnException


class BasicTrick(TrickTakingGame):

	'''
	Environment for simplest value-based card game. 

	This game is played with 8 cards of each suit, for a total of 32. Players receive 5 points 
	for every round, which can be won by playing the highest value, regardless of suit.
	A trump card can be played at any time and the hierarchy for the trump suit is 5, J, Highest Card of Hearts, 
	followed by the remaining cards in the trump suit in the traditional numerical order. 

	The game is won by the player with the highest score once all cards have been played (AKA 5 rounds). 

	'''

	name = 'Basic Trick'


	def step(self, action: Tuple[int, int]) -> Tuple[
	Tuple[List[int], ...], Tuple[int, ...], bool, Dict]:
		'''
		Overridden to add card redistribution under "reset trick". 
		'''
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
			rewards[player] -= 50	# Huge penalty for picking an invalid card!
			card_index = random.choice(valid_cards)  # Choose random valid move to play
			invalid_plays[player] = "invalid"
		else:
			pass
		    # possible to reward player for making good choice here

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
			rewards = [rewards[i] + trick_rewards[i] for i in range(num_players)]
			for i in range(num_players):
				offset = num_cards + num_players  # index into state correctly
				self._state[offset + i] += trick_rewards[i]  # update current scores

		    # Reset trick
			for i in range(num_cards, num_cards + num_players):
				self._state[i] = -1
			self._state[-2] = -1
			self._state[-1] = next_leader

		# Check if game ended
		if self._game_has_ended():
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

	def _deal(self) -> List[int]:
		'''
		Overridden to deal five cards to each player. 
		'''
		cards = []
		for i in range(self.num_players):
			cards += [i for _ in range(5)] #revisit, is 5 the best number for a reduced deck?
		cards += [-1]*(self.num_cards - len(cards))
		random.shuffle(cards)
		return cards

	def _end_trick(self) -> Tuple[List[int], int]:

		winning_player, winning_card = self._get_trick_winner()
		rewards = [0 for _ in range(self.num_players)]
		rewards[winning_player] = 5
		return rewards, winning_player

	def _get_trick_winner(self) -> Tuple[int, Card]:

		starting_card = self.trick_leader
		played_cards = [self.index_to_card(self._state[self.num_cards + i]) for i in
						range(self.num_players)]

		winning_index = -1
		winning_card = None
		for player_index, card in enumerate(played_cards):
			if winning_card is None:
				winning_card = card
				winning_index = player_index
			elif card.value > winning_card.value:
				winning_index = player_index
				winning_card = card
			elif card.value == winning_card.value:
				if card.suit.value > winning_card.suit.value:
					winning_index = player_index
					winning_card = card
			else:
				pass

		return winning_index, winning_card

	def is_valid_play(self, player_index, card_index) -> bool:

		if self._state[card_index] != player_index:
			return False
		return True

	@property
	def cards_per_suit(self) -> Tuple[int, ...]:
		return 6, 6, 6, 6