from typing import List, Tuple

from environments.trick_taking_game import TrickTakingGame
from util import Card, Suit


class TwentyFive(TrickTakingGame):

	'''
	Environment for twenty-five, a 4-player trump trick taking game. 
	'''

	name = 'Twenty-Five'


	def step(self, action: Tuple[int, int]) -> Tuple[
        Tuple[List[int], ...], Tuple[int, ...], bool, Dict]:
        
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
            if self._state[card_index] == player:
                rewards[player] -= 10
            else:
                rewards[player] -= 100  # Huge penalty for picking an invalid card!
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
            card_distribution = self._deal() #for now, assuming that we'll have to deal every time -- 20 cards needed and 32 - 20 < 20
            self.state += (card_distribution + [0]*(len(self.state) - len(card_distribution)))


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
		Deal five cards to each player and return positions of the cards as included in the state. 
		'''

		cards = []
		for i in range(self.num_players):
			cards += [i for _ in range(5)] #revisit, is 5 the best number for a reduced deck?
		random.shuffle(cards)
		return cards

    def _end_trick(self) -> Tuple[List[int], int]:
        winning_player, winning_card = self._get_trick_winner()
        rewards = [0 for _ in range(self.num_players)]
        rewards[winning_player] = 5
        return rewards, winning_player

    def _get_trick_winner(self) -> Tuple[int, Card]:
        trump_suit = self.trump_suit
        starting_card = self.trick_leader
        played_cards = [self.index_to_card(self._state[self.num_cards + i]) for i in
                        range(self.num_players)]

        winning_index = -1
        winning_card = starting_card
        trump_played = False
        for player_index, card in enumerate(played_cards):
        	if trump_played and card.suit = trump_suit:
        		#special trump cases
        		if (card.value = 5) or (card.value = 11 and winning_card.value != 5) or \
        			(card == Card(Suit.HEARTS, 1) and winning_card.value not in {5, 11}): #Another Ace Flag
        			winning_index = player_index
                	winning_card = card

                elif card.value >= winning_card.value:
                	winning_index = player_index
                	winning_card = card

            elif not trump_played and ((card.suit == winning_card.suit and card.value >= winning_card.value) or \
                    (card.suit == trump_suit and winning_card.suit != trump_suit) or (card == Card(Suit.HEARTS, 1))): #Another Ace Flag
                winning_index = player_index
                winning_card = card
                trump_played = True if card.suit == trump_suit or card == Card(Suit.HEARTS, 1) #Another Ace Flag

        return winning_index, winning_card

    def _game_has_ended(self) -> bool:
        scores = self.scores()
        return max(scores) == 25

    def is_valid_play(self, player_index, card_index) -> bool:
        if self._state[card_index] != player_index:
            return False

        # Check if player is empty of the starting suit if different suit was played
        played_card = self.index_to_card(card_index)
        starting_card = self.trick_leader
        if starting_card is None:
            return True
        if played_card.suit != starting_card.suit and played_card.suit != self.trump_suit:
            for i in range(self.num_cards):
            	card_in_hand = self.index_to_card(i)
                if self._state[i] == player_index and \
                	(card_in_hand.suit == starting_card.suit or \
                	(card_in_hand.suit == self.trump_suit and card_in_hand.value not in {5, 11}) or \
                	card_in_hand != Card(Suit.HEARTS, 1)): #Check if Ace is represented as 14 or 1...
                    return False
        return True

	@property
    def cards_per_suit(self) -> Tuple[int, ...]:
        return 8, 8, 8, 8


