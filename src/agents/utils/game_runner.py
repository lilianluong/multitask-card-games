# Created by Patrick Kao
from game import Game


class GameRunner:
    def __init__(self, task, agent_type, agent_params, game_params):
        self.agent_type = agent_type
        self.task = task
        self.agent_params = agent_params
        self.game_params = game_params

    def __call__(self, game_num):
        # print(f"Running game {game_num}")
        game = Game(self.task, [self.agent_type] * 4, self.agent_params, self.game_params)
        result = game.run()
        barbs = game.get_barbs()
        return barbs
