import time
import random
import numpy as np
import os
os.getcwd()
import sys
sys.path.append('my/path/to/module/folder')

from gym_sokoban.envs.room_utils import *
from gym_sokoban.envs.sokoban_env import *
from gym_sokoban.envs.sokoban_env_variations import SokobanEnv1

# ================================================================
# Environment and Global Parameters
RANDOM_SEED = 0
#env = SokobanEnv1(max_steps=1000)
env = SokobanEnv(dim_room=(8, 8), num_boxes=1)
# for reproducibility (since env is getting rendered randomly)
env.seed(RANDOM_SEED)               # always render the same environment
np.random.seed(RANDOM_SEED)         # always sample the same random number
random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)  # always take the same random action
env.reset()


# ================================================================
NROWS, NCOLS = env.dim_room[0], env.dim_room[1]
ALPHA = .1  # learning rate
GAMMA = .9  # discount factor
ACTION_LOOKUP = env.get_action_lookup()

# Relevant variables
V = np.zeros((10, 10))
pi = np.full([NROWS, NCOLS], np.inf, dtype=int)
# pi = np.full([NROWS, NCOLS], len(ACTION_LOOKUP)*" ", dtype="<U" + str(len(ACTION_LOOKUP)))


# ================================================================
def _run():
    global counter

    actionsTaken = list()

    for timestep in range(2):  # number of iterations
        # env.render('format')#'raw', scale=2)
        # time.sleep(1)  # to make the src moves more traceable
        current_state = env.player_position
        a = env.action_space.sample()

        #print(f"room_state=\n{env.room_state}")
        #print(f"currentState = {current_state}")
        #print(f"t={timestep} action taken = {a} = {ACTION_LOOKUP[a]}")
        # print(f"t={timestep}  a={ACTION_LOOKUP[a]}  state={currentState}")
        actionsTaken.append(a)

        print(env.room_state)

        room_structure = env.room_state.copy()
        room_structure[room_structure == 5] = 1
        room_structure[room_structure == 4] = 1

        print("DFS starts")
        #depth_first_search(env.room_state, room_structure, env.box_mapping)

        # take a step
        observation, reward, done, info = env.step(a)

        # episode has terminated
        if done:
            print("DONE: Episode finished after {} timesteps".format(timestep + 1))
            break

        if env._check_if_all_boxes_on_target():
            print("ALL BOXES ON TARGET: Episode finished after {} timesteps".format(timestep + 1))
            break

        env.render('format')
        #env.render('tiny_rgb_array', scale=scale_tiny)

    ActionsTaken = [ACTION_LOOKUP[a] for a in actionsTaken]
    print(f"actionsTaken={actionsTaken}")
    print(f"ActionsTaken={ActionsTaken}")

    if input():
        env.close()


# ================================================================
# Run the program
if __name__ == "__main__":
    _run()
