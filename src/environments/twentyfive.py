import random
from typing import Dict, List, Tuple

from environments.trick_taking_game import TrickTakingGame
from util import Card, Suit, OutOfTurnException


class TwentyFive(TrickTakingGame):
    '''
    Environment for twenty-five, a 4-player trump trick taking game.

    This variation plays with 8 cards of each suit, for a total of 32. Players receive 5 points
    for every trick, which can be won by playing the highest of the suit that was led or a trump
    card.
    A trump card can be played at any time and the hierarchy for the trump suit is 5, J,
    Highest Card of Hearts,
    followed by the remaining cards in the trump suit in the traditional numerical order.

    The game is won when a player earns 25 points -- AKA collects 5 tricks.

    Variations on normal game:
    - No ante(win at 25 points)
    - No robbing the pack
    - No jinking (nonstandard anyway)
    - No reneging
    - No strange ranking based on suit
    '''

    name = 'Twenty-Five'

    # TODO: randomize trump suit at beginning of each game
    def get_trump_suit(self) -> Suit:
        return Suit.SPADES

    def _deal(self) -> List[int]:
        '''
        Overridden to deal five cards to each player.
        '''
        cards = []
        for i in range(self.num_players):
            cards += [i for _ in range(5)]
        cards += [-1] * (self.num_cards - len(cards))
        random.shuffle(cards)
        return cards

    def _end_trick(self) -> Tuple[List[int], int]:
        '''
        Updated reward to match twenty-five scoring rules.
        '''
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
            if trump_played and card.suit == trump_suit:
                # special trump cases
                if (card.value == 5) or (card.value == 11 and winning_card.value != 5) or \
                        (card == Card(Suit.HEARTS,
                                      self._num_cards - 1) and winning_card.value not in {5, 11}):
                    winning_index = player_index
                    winning_card = card

                elif card.value >= winning_card.value:
                    winning_index = player_index
                    winning_card = card

            elif not trump_played and (
                    (card.suit == winning_card.suit and card.value >= winning_card.value) or \
                    (card.suit == trump_suit and winning_card.suit != trump_suit) or (
                            card == Card(Suit.HEARTS, self._num_cards - 1))):
                winning_index = player_index
                winning_card = card
                if card.suit == trump_suit or card == Card(Suit.HEARTS, self._num_cards - 1):
                    trump_played = True
        return winning_index, winning_card

    def is_valid_play(self, player_index, card_index) -> bool:

        if self._state[card_index] != player_index:
            return False

        # Check if player is empty of the starting suit if different suit was played
        played_card = self.index_to_card(card_index)
        starting_card = self.trick_leader

        if starting_card is None:
            return True

        # always allow trump suit
        if played_card.suit != starting_card.suit and played_card.suit != self.trump_suit:
            for i in range(self.num_cards):
                card_in_hand = self.index_to_card(i)
                if self._state[i] == player_index and \
                        (card_in_hand.suit == starting_card.suit):
                    # TODO: allow reneging
                    # or (card_in_hand.suit == self.trump_suit and card_in_hand.value not in {5, 11}) or \
                    # card_in_hand == Card(Suit.HEARTS, self._num_cards-1)):
                    return False
        return True

    @property
    def cards_per_suit(self) -> Tuple[int, ...]:
        return 8, 8, 8, 8
