from typing import List, Tuple

from environments.trick_taking_game import TrickTakingGame
from util import Card, Player


class Game:
    """
    A class that acts as an interface for the game.

    Initialize a Game with a TrickTakingGame (or child class). Upon calling run(), the game
    will query the player for an action or run a policy.
    """

    def __init__(self, game: TrickTakingGame.__class__, player_setting: List[Player] = None):
        """
        Initialize a new game.

        :param game: either TrickTakingGame, or a class that inherits from it, the game to be played
        :param player_setting: List[Player], defining the type of player in order of their seated position
        """
        self._game = game()

        if player_setting is None:
            player_setting = [Player.HUMAN] * self._game.num_players
        assert len(player_setting) == self._game.num_players, "number of players doesn't fit the game requirements"
        self._player_types = player_setting[:]

        self._observations = None

    def run(self) -> List[int]:
        """
        Start and play the game. Can only be called once per instance of Game.
        :return: final score of the game
        """
        assert self._observations is None, "game has already been played"
        self._observations = self._game.reset()
        done = False

        # Play the game
        while not done:
            next_player = self._game.next_player
            selected_card = self._choose_action(next_player)
            card_index = self._game.card_to_index(selected_card)
            observations, rewards, done, info = self._game.step((next_player, card_index))
            self._print_report(next_player, selected_card, observations, rewards)
            self._observations = observations

        # Game has finished
        self._game_ended()

        return self._game.scores

    def _print_report(self,
                      player_who_went: int,
                      card_played: Card,
                      observations: Tuple[List[int], ...],
                      rewards: Tuple[int, ...],
                      ):
        """
        Print out any information for the user about what just happened in a human-readable way.
        Do not reveal hidden information.

        TODO: Implement!

        :param player_who_went: index of the player who took the last move
        :param card_played: the card that was played
        :param observations: a vector where the i^th element is the observation of player i
        :param rewards: a vector where the i^th element is the reward given to playre i
            NOTE: rewards are distinct from game scores
        :return: None
        """
        raise NotImplementedError

    def _game_ended(self):
        """
        Do things related to the end of the game, if necessary, e.g. printing things

        TODO: Implement!

        :return: None
        """
        raise NotImplementedError

    def _choose_action(self, player: int) -> Card:
        """
        Retrieve the selected action of a player.

        :param player: index of the player who should be taking a turn
        :return: the Card that they are playing
        """
        if self._player_types[player] == Player.HUMAN:
            return self._choose_action_human(player)
        raise NotImplementedError("This player type hasn't been defined in Game")

    def _choose_action_human(self, player: int) -> Card:
        """
        Display the necessary information for a human player and then prompt them to take their turn.
        TODO: Implement!
        :param player: index of the player who should be taking a turn
        :return: the Card the human player selects to play
        """
        raise NotImplementedError
