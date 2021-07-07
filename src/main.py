import sys
sys.path.append('my/path/to/module/folder')
import matplotlib.pyplot as plt

from algorithms import Trainer, SokobanNN
from algorithms.mcts import execute_episode
from src.replay_memory import ReplayMemory

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
ACTION_LOOKUP = env.get_action_lookup()


# ================================================================
def _run():

    n_actions = len(ACTION_LOOKUP)
    n_obs = 50

    # TODO: fill args
    trainer = Trainer(Policy=SokobanNN(Env=env, args=...))
    network = trainer.step_model

    mem = ReplayMemory(size=...,
                       column_types=
                       { "ob": np.long,
                         "pi": np.float32,
                         "return": np.float32 },
                       column_shapes=
                       { "ob": [],
                         "pi": [n_actions],
                        "return": [] })

    def test_agent(iteration):
        """
        Test the Neural Network on the same environment. Log how it performed.

        Arguments:
             iteration: Number of training iterations\episodes so far.
        """
        test_env = SokobanEnv(dim_room=(8, 8), num_boxes=1) # TODO: check if this is always the same environemnt to test the agent on
        total_rew = 0
        observation, reward, done, _ = test_env.reset()
        step_idx = 0

        while not done:
            log(test_env, iteration, step_idx, total_rew)
            p, _ = network.step(np.array([observation]))
            action = np.argmax(p)
            observation, reward, done, _ = test_env.step(int(action))
            step_idx += 1
            total_rew += reward

        log(test_env, iteration, step_idx, total_rew)

    value_losses = []
    policy_losses = []

    for numEpisodes in range(10):

        # After each 50th iteration print the losses for the current agent.
        if numEpisodes % 50 == 0:
            test_agent(numEpisodes)
            plt.plot(value_losses, label="value losses")
            plt.plot(policy_losses, label="policy losses")
            plt.legend()
            plt.show()

    # TODO: Check if this works in principal
    # Execute an episode with a specific number of simulations per step.
    observations, pis, returns, total_reward, done_state = execute_episode(
                                                            agentNetw=network,
                                                            numSimulations=32,
                                                            Env=SokobanEnv)

    mem.add_all({"ob": observations,
                 "pi": pis,
                 "return": returns})

    batch = mem.get_minibatch()

    value_loss, policy_loss = trainer.train(obs=batch["ob"],
                                            search_pis=batch["pi"],
                                            returns=batch["return"])
    value_losses.append(value_loss)
    policy_losses.append(policy_loss)

    # global counter
    #
    # actionsTaken = list()
    # totalReward = 0
    #
    # for timestep in range(5):
    #     env.render('colored')
    #
    #     # Sample an action.
    #     a = env.action_space.sample()
    #     actionsTaken.append(env.get_action_lookup_chars(a))
    #
    #     # take a step
    #     observation, reward, done, info = env.step(a)
    #     totalReward += reward
    #
    #     print(env.get_action_lookup_chars(a) + "  " + str(reward))
    #
    #     if done:
    #         print("DONE: Episode finished after {} timesteps".format(timestep + 1))
    #         break
    #
    #     if env._check_if_all_boxes_on_target():
    #         print("ALL BOXES ON TARGET: Episode finished after {} timesteps".format(timestep + 1))
    #         break
    #
    # print(f"actionsTaken={actionsTaken}")
    # print(f"total reward={totalReward}")
    #
    # env.close()

# ================================================================
# Run the program
if __name__ == "__main__":
    _run()
