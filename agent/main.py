import pyglet
import time
import random
import numpy as np
from gym_sokoban.envs.sokoban_env import *

# ================================================================
# Environment and Global Parameters
# ================================================================
RANDOM_SEED = 0
env = SokobanEnv()

# for reproducibility (since env is getting rendered randomly)
env.seed(RANDOM_SEED)               # always render the same environment
np.random.seed(RANDOM_SEED)         # always sample the same random number
random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)  # always take the same rnadom action
env.reset()

NROWS, NCOLS = env.dim_room[0], env.dim_room[1]
ALPHA = .1  # learning rate
GAMMA = .9  # discount factor
ACTION_LOOKUP = env.get_action_lookup()


# ================================================================
# Algorithm relevant variables
# ================================================================
V = np.zeros((10, 10))
pi = np.full([NROWS, NCOLS], "012345678")


# ================================================================

def _demo():
    # let the agent reinforce
    # ----------------------------------------
    for timestep in range(3):  # number of iterations
        env.render()
        time.sleep(2.5)  # to make the agents moves more traceable

        print(env.room_state)
        currentState = env.get_player_position()

        vsAfterActions = np.zeros(len(ACTION_LOOKUP))

        # take a step : ToDo: for now random, change to rl algo step
        for index, action in ACTION_LOOKUP.items():
            # get next state after action
            nextState = env.state_after_action(index)
            aobservation, areward, adone, ainfo = env.step(index)

            vsAfterActions = np.append(vsAfterActions,
                                       areward + GAMMA * V[nextState[0]][nextState[1]])

            print(f"nextState={nextState} after action={action} : reward:{areward}  {V[nextState[0]][nextState[1]]}")

        a = env.action_space.sample()
        print(f"t={timestep}  a={ACTION_LOOKUP[a]}  state={currentState}")

        # take a step
        observation, reward, done, info = env.step(a)

        # episode has terminated
        if done:
            print("DONE: Episode finished after {} timesteps".format(timestep + 1))
            break

        if env._check_if_all_boxes_on_target():
            print("ALL BOXES ON TAGRTE: Episode finished after {} timesteps".format(timestep + 1))
            break

    if input():
        env.close()


if __name__ == "__main__":
    _demo()
