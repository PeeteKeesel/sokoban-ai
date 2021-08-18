import argparse
import sys

from gym_sokoban.envs.mcts_sokoban_env import MctsSokobanEnv
from algorithms.mcts import execute_episode, Mcts
from gym_sokoban.envs.room_utils import *
from gym_sokoban.envs.sokoban_env import *
from time import time, sleep

sys.path.append('my/path/to/module/folder')


RANDOM_SEED = 0

# ================================================================
# def _run():
#
#     # obs, _, _, _ = env.step(0)
#     # print(obs)
#     # print(obs[1][1])
#     # print(obs[1][1].shape)
#     # print(obs.shape)
#     # env.render_colored()
#     # print(env.room_state)
#     # print(env.render('raw'))
#
#     n_actions = len(ACTION_LOOKUP)
#     n_obs = 50
#
#     # TODO: fill args
#     trainer = Trainer(Policy=SokobanNN(Env=env, args=...))
#     network = trainer.step_model
#
#     mem = ReplayMemory(size=...,
#                        column_types=
#                        { "ob": np.long,
#                          "pi": np.float32,
#                          "return": np.float32 },
#                        column_shapes=
#                        { "ob": [],
#                          "pi": [n_actions],
#                          "return": [] })
#
#     def test_agent(iteration):
#         """
#         Test the Neural Network on the same environment. Log how it performed.
#
#         Arguments:
#              iteration: Number of training iterations\episodes so far.
#         """
#         test_env = SokobanEnv(dim_room=(8, 8), num_boxes=1) # TODO: check if this is always the same environemnt to test the agent on
#         total_rew = 0
#         observation, reward, done, _ = test_env.reset()
#         step_idx = 0
#
#         while not done:
#             log(test_env, iteration, step_idx, total_rew)
#             p, _ = network.step(np.array([observation]))
#             action = np.argmax(p)
#             observation, reward, done, _ = test_env.step(int(action))
#             step_idx += 1
#             total_rew += reward
#
#         log(test_env, iteration, step_idx, total_rew)
#
#     value_losses = []
#     policy_losses = []
#
#     for numEpisodes in range(10):
#
#         # After each 50th iteration print the losses for the current agent.
#         if numEpisodes % 50 == 0:
#             test_agent(numEpisodes)
#             plt.plot(value_losses, label="value losses")
#             plt.plot(policy_losses, label="policy losses")
#             plt.legend()
#             plt.show()
#
#     # TODO: Check if this works without specifically caring about the NNet
#     # Execute an episode with a specific number of simulations per step.
#     observations, pis, returns, total_reward, done_state = execute_episode(
#                                                             agentNetw=network,
#                                                             numSimulations=32,
#                                                             Env=SokobanEnv)
#
#     mem.add_all({"ob": observations,
#                  "pi": pis,
#                  "return": returns})
#
#     batch = mem.get_minibatch()
#
#     value_loss, policy_loss = trainer.train(obs=batch["ob"],
#                                             search_pis=batch["pi"],
#                                             returns=batch["return"])
#     value_losses.append(value_loss)
#     policy_losses.append(policy_loss)
#
#     # global counter
#     #
#     # actionsTaken = list()
#     # totalReward = 0
#     #
#     # for timestep in range(5):
#     #     env.render('colored')
#     #
#     #     # Sample an action.
#     #     a = env.action_space.sample()
#     #     actionsTaken.append(env.get_action_lookup_chars(a))
#     #
#     #     # take a step
#     #     observation, reward, done, info = env.step(a)
#     #     totalReward += reward
#     #
#     #     print(env.get_action_lookup_chars(a) + "  " + str(reward))
#     #
#     #     if done:
#     #         print("DONE: Episode finished after {} timesteps".format(timestep + 1))
#     #         break
#     #
#     #     if env._check_if_all_boxes_on_target():
#     #         print("ALL BOXES ON TARGET: Episode finished after {} timesteps".format(timestep + 1))
#     #         break
#     #
#     # print(f"actionsTaken={actionsTaken}")
#     # print(f"total reward={totalReward}")
#     #
#     # env.close()

def make_reproducible(env, random_seed):
    """
    Since the environment is getting rendered randomly some reproducibility is
    needed.

    Arguments:
        env         (MctsSokobanEnv) - Environment.
        random_seed (int)            - Seed for repoducibility.
    """
    env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    env.action_space.seed(random_seed)
    env.reset()


def mcts_solve(args):

    # Create the environment.
    env = gym.make("MCTS-Sokoban-v0",
                   dim_room=(6, 6),
                   max_steps=args.max_steps,
                   num_boxes=args.num_boxes)
    make_reproducible(env, args.random_seed)

    # Initialize the Monte-Carlo-Tree-Search.
    mcts = Mcts(Env=env,
                simulation_policy=args.sim_policy,
                max_rollouts=args.max_rollouts,
                max_depth=args.max_depth,
                num_parallel=args.num_parallel)
    mcts.initialize_search()

    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    first_node = mcts.root.select_and_expand()
    first_node.backpropagate(0, mcts.root)

    time_limit = arguments.time_limit * 60
    start_time = time()

    while True:
        now = time()
        if now - start_time > time_limit:
            print(f"Time limit of {args.time_limit} reached.")
            # break

        # Run the Monte-Carlo-Tree-Search for the current state and take the
        # best action after all simulations were performed.
        _, reward, done, _ = mcts.take_best_action()
        print(f"MAIN: reward={reward}")

        if done:
            print("MAIN: is done")
            break

    mcts.root.print_tree()


    # while True:
    #     now = time()


    # # Must run this once at the start, so that noise injection actually affects
    # # the first action of the episode.
    # firstNode = mcts.root.select_and_expand()
    # firstNode.backpropagate(0, mcts.root)
    #
    # # print(20*"#"+"\n" + mcts.root.print_tree() + "\n"+20*"#\n")
    # mcts.root.print_tree()
    #
    # rollouts = 1
    # while rollouts <= mcts.max_rollouts:
    #     print(f"\n\n--- Rollout {rollouts}")
    #     # the # of times the node was visited
    #     prevSimulations = mcts.root.N
    #
    #     # We want `num_simulations` simulations per action not counting
    #     # simulations from previous actions.
    #     while mcts.root.N < prevSimulations + numSimulations:
    #         print(30 * "**" + f" {mcts.root.N} < {prevSimulations} + {numSimulations} " + 30 * "**")
    #         mcts.tree_search_random(num_simulations=numSimulations)
    #
    #     print(f" {mcts.root.N} > {prevSimulations} + {numSimulations} ")
    #     print("_" * 75 + f" After {mcts.root.N - prevSimulations} simulations performed for the current node.")
    #     mcts.root.print_tree()
    #     print("_" * 100)
    #
    #     action = mcts.pick_action()
    #     print(
    #         f"    picked action {action}={Env.get_action_lookup_chars(action)} after action_traj={mcts.root.Env.print_actions_as_chars(mcts.root.action_traj)}")
    #     assert action != 0
    #     mcts.take_action(action)
    #     print(f"        reward={mcts.rewards}")
    #
    #     rollouts += 1
    #
    #     if mcts.root.Env._check_if_all_boxes_on_target():
    #         print(
    #             f"After rollout {rollouts} and traj={mcts.root.Env.print_actions_as_chars(mcts.root.action_traj)} ALL BOXES ARE ON TARGET!")
    #         break
    #
    #     # if mcts.root.game_is_done():
    #     #     print("++"*1000)
    #     #     print(f"IF MCTS.ROOT.IS_DONE() after {prevSimulations} simulations with action_traj = {mcts.root.Env.print_actions_as_chars(mcts.root.action_traj)}")
    #     #     print("++" * 1000)
    #     #     break
    #
    # print(100 * "_" + f"\n{rollouts} Rollouts performed.")
    # mcts.root.print_tree()


if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=np.int, default=0,
                        help="Seed to handle the rendered board")
    parser.add_argument("--num_boxes", type=np.int, default=1,
                        help="Number of boxes on the board")
    parser.add_argument("--max_rollouts", type=np.int, default=100,
                        help="Number of rollouts (simulations) per move")
    parser.add_argument("--max_depth", type=np.int, default=5,
                        help="Depth of each rollout")
    parser.add_argument("--max_steps", type=np.int, default=120,
                        help="Moves before game is lost")
    parser.add_argument("--num_parallel", type=np.int, default=8,
                        help="Number of leaf nodes to collect before "
                             "evaluating them in conjunction")
    parser.add_argument("--sim_policy", type=np.str, default="random",
                        help="Simulation policy")
    parser.add_argument("--time_limit", type=np.int, default=60,
                        help="Time (in minutes) per board")
    arguments = parser.parse_args()

    # Solve the game.
    mcts_solve(arguments)
