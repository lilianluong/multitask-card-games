# Created by Patrick Kao
import queue
import threading
from collections import defaultdict

from flask import Flask, render_template, request

from environments.hearts import SimpleHearts
from environments.trick_taking_game import TrickTakingGame
from game import Game
from util import Card, Suit, Player


class FlaskGame(Game):
    def __init__(self, game: TrickTakingGame.__class__):
        super().__init__(game)
        self.input_queue = queue.Queue()

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
                card = Card(Suit(data['type']), int(data["card"]))
                self.input_queue.put(card)

        player_current_cards = dict()
        player_current_cards['0'] = {}
        player_current_cards['1'] = {}
        player_current_cards['2'] = {}
        player_current_cards['3'] = {}

        log_message = "Please select a card"

        # assumes only 1 human player and takes first one
        human_players = self._player_types[self._player_types == Player.HUMAN]
        # assert len(human_players)==1, f"need only 1 human player, have {len(human_players)}"
        human_index = [idx for idx, element in enumerate(self._player_types) if element == Player.HUMAN][0]
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


app = Flask(__name__, template_folder="../../templates", static_folder="../../static")
app.config['DEBUG'] = False
game = FlaskGame(SimpleHearts)


@app.route('/play', methods=['POST', 'GET'])
def handle_website():
    return game.render()


if __name__ == "__main__":
    # test for class
    threading.Thread(target=app.run).start()
    game.run()
