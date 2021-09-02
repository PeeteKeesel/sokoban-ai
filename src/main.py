import argparse
import sys

from algorithms.mcts import Mcts
from src.algorithms import depth_first_search as dfs
from src.algorithms import breadth_first_search as bfs
from src.algorithms import uniform_cost_search as ucs
from src.algorithms import a_star_search as astar
from src.algorithms import ida_star_search as idastar
from gym_sokoban.envs.sokoban_env import *
from time import time, sleep

sys.path.append('my/path/to/module/folder')


LEGAL_ACTIONS = np.array([1, 2, 3, 4])

RANDOM_SEED = 10
DIM_ROOM = 7
NUM_BOXES = 1

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

    Args:
        env (MctsSokobanEnv): Environment.
        random_seed (int): Seed for repoducibility.
    """
    env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    env.action_space.seed(random_seed)
    env.reset()


def create_environment(args):
    """
    Creates a Sokoban environment of a given dimension, with a given limit of
    steps and a given number of boxes to put in the world.

    Args:
        args (Namespace object): Attributes to build the Sokoban environment.
                                 Mandatories: dim_room, max_steps, num_boxes,
                                              random_seed.

    Returns:
        env (MctsSokobanEnv): The Sokoban environment.
    """
    env = gym.make("MCTS-Sokoban-v0",
                   dim_room=(args.dim_room, args.dim_room),
                   max_steps=args.max_steps,
                   num_boxes=args.num_boxes)
    make_reproducible(env, args.random_seed)

    return env


def search_algorithms_solve(args):

    # Create the environment.
    env = create_environment(args)
    env.render_colored()

    time_limit, metrics = arguments.time_limit * 60, None
    if args.search_algo in SEARCH_ALGORITHMS:
        if args.search_algo == "dfs":
            metrics, _ = dfs(env, time_limit, print_steps=True)
        elif args.search_algo == "bfs":
            metrics, _ = bfs(env, time_limit, print_steps=True)
        elif args.search_algo == "ucs":
            metrics, _ = ucs(env, time_limit, print_steps=True)
        elif args.search_algo == "astar":
            metrics, _ = astar(env, time_limit, print_steps=True)
        elif args.search_algo == "idastar":
            metrics, _ = idastar(env, time_limit, print_steps=True,
                                 heuristic=args.heuristic)
    else:
        raise Exception(f"Algorithm `{args.search_algo}` is not in the list of"
                        f" available algorithms"
                        f" [`dfs`, `bfs`, `ucs`, `astar`, `idastar`]")

    if args.render_env == "T":
        env.render()
        for a in metrics["action_traj"]:
            sleep(0.5)
            env.step(CHARS_LOOKUP_ACTIONS[a])
            env.render()
        env.sleep(60)


def mcts_solve(args):

    # Create the environment.
    env = create_environment(args)

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

    a_traj = []
    while True:
        now = time()
        if now - start_time > time_limit:
            print(f"Time limit of {args.time_limit} reached.")
            break

        # Run the Monte-Carlo-Tree-Search for the current state and take the
        # best action after all simulations were performed.
        _, reward, done, _, best_action = mcts.take_best_action()
        print(f"MAIN: reward={reward}")
        a_traj.append(best_action)

        if done:
            print(f"DONE: Solution found!\n"
                  f"      trajectory  : {mcts.Env.print_actions_as_chars(a_traj)}\n"
                  f"      total reward: {reward}")
            break
        elif len(a_traj) >= args.max_steps:
            print(f"DONE: Maximal number of steps {args.max_steps} reached!\n"
                  f"      trajectory  : {mcts.Env.print_actions_as_chars(a_traj)}\n"
                  f"      total reward: {reward}")
            break

    mcts.root.print_tree()


if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=np.int, default=RANDOM_SEED,
                        help="Seed to handle the rendered board")
    parser.add_argument("--dim_room", type=np.int, default=DIM_ROOM,
                        help="Dimension of the Sokoban board")
    parser.add_argument("--num_boxes", type=np.int, default=NUM_BOXES,
                        help="Number of boxes on the board")
    parser.add_argument("--max_rollouts", type=np.int, default=50,
                        help="Number of rollouts (simulations) per move")
    parser.add_argument("--max_depth", type=np.int, default=10,
                        help="Depth of each rollout (simulation)")
    parser.add_argument("--max_steps", type=np.int, default=120,
                        help="Moves before game is lost")
    parser.add_argument("--num_parallel", type=np.int, default=8,
                        help="Number of leaf nodes to collect before "
                             "evaluating them in conjunction")
    parser.add_argument("--sim_policy", type=np.str,
                        default=SIMULATION_POLICIES["random"],
                        help="Simulation policy. "
                             "Implemented options: "
                             "[`random`, `eps-greedy`]")
    parser.add_argument("--search_algo", type=np.str,
                        default=SEARCH_ALGORITHMS["idastar"],
                        help="Alternative search algorithm to solve the game. "
                             "Implemented options: "
                             "[`dfs`, `bfs`, `ucs`, `astar`, `idastar`]")
    parser.add_argument("--heuristic", type=np.str,
                        default=HEURISTICS["hungarian"],
                        help="Heuristic method for IDA* search algorithm. "
                             "Implemented options: "
                             "[`manhattan`, `hungarian`]")
    parser.add_argument("--time_limit", type=np.int, default=5,
                        help="Time (in minutes) per board")
    parser.add_argument("--render_env", type=np.str, default="F",
                        help="If the result should be rendered `T` or not `F`")
    arguments = parser.parse_args()

    # Solve the game.
    if arguments.search_algo:
        search_algorithms_solve(arguments)
    else:
        mcts_solve(arguments)
