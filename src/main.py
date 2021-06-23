import sys
sys.path.append('my/path/to/module/folder')

from gym_sokoban.envs.room_utils import *
from gym_sokoban.envs.sokoban_env import *


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
    totalReward = 0

    for timestep in range(5):
        env.render('colored')

        # Sample an action.
        a = env.action_space.sample()
        actionsTaken.append(env.get_action_lookup_chars(a))

        # take a step
        observation, reward, done, info = env.step(a)
        totalReward += reward

        print(env.get_action_lookup_chars(a) + "  " + str(reward))

        if done:
            print("DONE: Episode finished after {} timesteps".format(timestep + 1))
            break

        if env._check_if_all_boxes_on_target():
            print("ALL BOXES ON TARGET: Episode finished after {} timesteps".format(timestep + 1))
            break

    print(f"actionsTaken={actionsTaken}")
    print(f"total reward={totalReward}")

    env.close()

# ================================================================
# Run the program
if __name__ == "__main__":
    _run()
