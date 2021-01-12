import pyglet
import time
import random
import numpy as np
from gym_sokoban.envs.room_utils import *
from gym_sokoban.envs.sokoban_env import *
from gym_sokoban.envs.sokoban_env_variations import SokobanEnv1, SokobanEnv2

# ================================================================
# Environment and Global Parameters
# ================================================================
RANDOM_SEED = 0
env = SokobanEnv1()

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
pi = np.full([NROWS, NCOLS], " ", dtype="<U1")
# pi = np.full([NROWS, NCOLS], len(ACTION_LOOKUP)*" ", dtype="<U" + str(len(ACTION_LOOKUP)))


# ================================================================

def _demo():
    # let the agent reinforce
    # ----------------------------------------
    for timestep in range(1):  # number of iterations
        #env.render()
        time.sleep(0.3)  # to make the agents moves more traceable

        print(env.room_state)

        # The current state of the agent
        current_state = env.player_position
        print(current_state)

        # Depth-First-Search (DFS): searches through all possible states in the room
        dfs = depth_first_search(current_state,
                                 )

        # vsAfterActions = np.zeros(len(ACTION_LOOKUP))
        # VMaxAfterAction = 0
        # asMax = np.array([])
        #
        # # take a step : ToDo: for now random, change to rl algo step
        # for index, action in ACTION_LOOKUP.items():
        #
        #     # get next state after taking an action
        #     nextState = env.state_after_action(index)
        #
        #     aobservation, areward, adone, ainfo = env.step(index)
        #
        #     bellmannEq = areward + GAMMA * V[nextState[0]][nextState[1]]
        #
        #     if bellmannEq >= VMaxAfterAction:
        #         VMaxAfterAction = bellmannEq
        #         asMax = np.append(asMax, index)
        #
        #     print(f"nextState={nextState} after action={action} : reward:{areward}  {V[nextState[0]][nextState[1]]}  Value={bellmannEq}")
        #
        # # update current state value
        # vMax = VMaxAfterAction
        # V[currentState[0]][currentState[1]] = vMax
        #
        # # find the action which led to the maximal value
        # # if multiple: choose one randomly
        # aMax = np.random.choice(asMax)
        # pi[currentState[0]][currentState[1]] = str(int(aMax)) # int(aMax) * " " + str(int(aMax)) + (len(ACTION_LOOKUP) - int(aMax) - 1) * " "
        #
        # print(f"vsMax={np.round(vMax,3)}\n{np.round(V, 3)}\n{pi}")



        print(env.action_space.sample())
        a = env.action_space.sample()
        # print(f"t={timestep}  a={ACTION_LOOKUP[a]}  state={currentState}")

        # take a step
        observation, reward, done, info = env.step(a)

        # episode has terminated
        if done:
            print("DONE: Episode finished after {} timesteps".format(timestep + 1))
            break

        if env._check_if_all_boxes_on_target():
            print("ALL BOXES ON TARGET: Episode finished after {} timesteps".format(timestep + 1))
            break

    if input():
        env.close()

# -------------------------------------------------------------------------------------------------
# Run the program
if __name__ == "__main__":
    _demo()
