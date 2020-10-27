# Created by Patrick Kao
import queue
import threading
from collections import defaultdict
from typing import List, Dict, Any

from flask import Flask, render_template, request

from agents.base import Agent
from agents.human import Human
from environments.trick_taking_game import TrickTakingGame
from game import Game
from util import Card, Suit


class FlaskGame(Game):
    """
    Singleton class that can display contents of game on flask server
    """
    __instance = None

    @staticmethod
    def getInstance(game: TrickTakingGame.__class__ = None,
                    player_setting: List[Agent.__class__] = None):
        """ Static access method. """
        if FlaskGame.__instance is None:
            FlaskGame.__instance = FlaskGame(game, player_setting)
        return FlaskGame.__instance

    def __init__(self, game: TrickTakingGame.__class__,
                 player_setting: List[Agent.__class__] = None,
                 agent_params: List[Dict[str, Any]] = None):
        if FlaskGame.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            super().__init__(game, player_setting, agent_params)
            self.input_queue = queue.Queue()
            self.render_queue = queue.Queue()
            self.first = True

    def _choose_action_human(self, player: int) -> Card:
        """
        Block until flask server returns next card to choose
        :param player:
        :return:
        """
        return self.input_queue.get()

    def render(self, mode: str = "human", view: int = -1):
        """
        Display website and if player selects card, add player chosen card to synchronous queue
        :return:
        """
        if hasattr(request, 'form'):
            data = request.form  # contains player input in dict form fields: card=rank, type=suit
            # add selection to queue
            if "card" in data:
                card = Card(Suit[data['type'].upper()], int(data["card"]))
                self.input_queue.put(card)
                # wait for game update
                self.render_queue.get()

        trick_cards = self._game.current_trick()
        player_current_cards = defaultdict(dict)
        for player, card in trick_cards.items():
            player_current_cards[player]["type"] = card.suit.name.lower()
            player_current_cards[player]["card"] = card.value
        
        log_message = "Please select a card"

        # assumes only 1 human player and takes first one
        human_players = self._agent_list[isinstance(self._agent_list, Human)]
        # assert len(human_players)==1, f"need only 1 human player, have {len(human_players)}"
        human_index = \
            [idx for idx, element in enumerate(self._agent_list) if isinstance(element, Human)][0]
        player_state = defaultdict(list)  # needs to be dict of suit:ranks in suit
        hand = self._game._get_hands()[human_index]
        for card_ind in hand:
            card = self._game.index_to_card(card_ind)
            suit_str = card.suit.name.lower()
            player_state[suit_str].append(card.value)

        players_score = self._game.scores[human_index]
        return render_template("main.html",
                               player_current_cards=player_current_cards,
                               log_message=log_message,
                               user_player=player_state,
                               players_score=players_score)

    def _choose_action(self, player: int) -> Card:
        # update render queue
        if isinstance(self._agent_list[player], Human):
            if self.first:
                self.first = False
            else:
                self.render_queue.put(0)  # any value
        return super()._choose_action(player)


app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config['DEBUG'] = False
threading.Thread(target=app.run).start()


@app.route('/play', methods=['POST', 'GET'])
def handle_website():
    return FlaskGame.getInstance().render(request.method)
