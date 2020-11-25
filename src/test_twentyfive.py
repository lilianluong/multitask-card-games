from environments.twentyfive import TwentyFive
import random

env = TwentyFive()
env.reset()
env._state = [1, -1, 1, -1, 2, 3, -1, -1, 1, 0, -1, 0, 0, 1, -1, 2, 1, 2, 3, 3, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, 3, 3, -1, 10, 1, 2, 3, 25, 0, 0, 0, 2, -1, 0]
print(env.step((0, 11)))
# env._state = {[]}
# player = 0
#while not env._game_has_ended():
# 	for i in range(4):
# 		print(i)
# 		action = random.randint(0, 8)
# 		print(env.step((player, action)))
# 		player = env.next_player
# 		print("Next player", player)


