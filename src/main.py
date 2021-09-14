import argparse
import sys

from absl import flags
from pathlib import Path
#from algorithms.mcts_nnet import Mcts
from algorithms.mcts_raw import Mcts
from src.algorithms import depth_first_search as dfs
from src.algorithms import breadth_first_search as bfs
from src.algorithms import uniform_cost_search as ucs
from src.algorithms import a_star_search as astar
from src.algorithms import ida_star_search as idastar
from gym_sokoban.envs.sokoban_env import *
from time import time, sleep
from utils import SIMULATION_POLICIES, SEARCH_ALGORITHMS, HEURISTICS, \
    ALGORITHM_NAME_DFS, ALGORITHM_NAME_BFS, ALGORITHM_NAME_UCS, \
    ALGORITHM_NAME_A_STAR, ALGORITHM_NAME_IDA_STAR, ALGORITHM_NAME_MCTS


sys.path.append('my/path/to/module/folder')

# Path to the results of the experiments.
PATH = "../experimental_results"

LEGAL_ACTIONS = np.array([1, 2, 3, 4])
ACTION_MAP = {
    'push up': "U",
    'push down': "D",
    'push left': "L",
    'push right': "R",
}

# The following is only used for automatic rendering.
RANDOM_SEED = [1]
DIM_ROOM = 8
NUM_BOXES = 2

# Set of levels to solve by different random seed values.
LEVELS_TO_SOLVE = list(map(str, np.arange(1, 21, 1)))
PATH_TO_TEST_BOARDS = "../boards/"
TEST_BOARDS = [PATH_TO_TEST_BOARDS + "test_board_8x8_3"]

MICROBAN_BOARDS = [PATH_TO_TEST_BOARDS + "microban_levels"]
MICROBAN_LEVELS_TO_SOLVE = list(map(str, np.arange(1, 11, 1)))

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
    env = None
    if args.file_name:

        if args.input_type == "raw":
            dim_room, n_boxes, soko_map = parse_raw_input(args.file_name)
        elif args.input_type == "human":
            print(f"micro_level = {args.micro_levels}")
            dim_room, n_boxes, soko_map, microban_level = \
                parse_human_input(args.file_name,
                                  args.micro_levels)
        else:
            return NotImplementedError(
                f"args.input_type `{args.input_type}` is not implemented. "
                f"Choose from [`human`, `raw`].")

        env = gym.make("MCTS-Sokoban-v0",
                       dim_room=dim_room,
                       max_steps=args.max_steps,
                       num_boxes=n_boxes,
                       original_map=soko_map)
    else:
        env = gym.make("MCTS-Sokoban-v0",
                       dim_room=(args.dim_room, args.dim_room),
                       max_steps=args.max_steps,
                       num_boxes=args.num_boxes)

    make_reproducible(env, args.seeds)

    return env


def search_algorithms_solve(args):

    # Create the environment.
    env = create_environment(args)
    env.render_colored()

    time_limit, results = arguments.time_limit * 60, None
    if args.search_algo in SEARCH_ALGORITHMS:
        if args.search_algo == ALGORITHM_NAME_DFS:
            results = dfs(env, time_limit, print_steps=False)
        elif args.search_algo == ALGORITHM_NAME_BFS:
            results = bfs(env, time_limit, print_steps=False)
        elif args.search_algo == ALGORITHM_NAME_UCS:
            results = ucs(env, time_limit, print_steps=False)
        elif args.search_algo == ALGORITHM_NAME_A_STAR:
            results = astar(env, time_limit, print_steps=False)
        elif args.search_algo == ALGORITHM_NAME_IDA_STAR:
            results = idastar(env, time_limit, print_steps=False,
                                 heuristic=args.heuristic)
    else:
        raise Exception(f"Algorithm `{args.search_algo}` is not in the list of"
                        f" available algorithms: "
                        f"[`{ALGORITHM_NAME_DFS}`, "
                        f"`{ALGORITHM_NAME_BFS}`, "
                        f"`{ALGORITHM_NAME_UCS}`, "
                        f"`{ALGORITHM_NAME_A_STAR}`, "
                        f"`{ALGORITHM_NAME_IDA_STAR}`]")

    if results == (None, None):
        return
    results['dim_room'] = args.dim_room
    results['num_boxes'] = args.num_boxes
    results['seed'] = args.seeds

    if args.print_results and results:
        print_search_algorithm_results(results)

    if args.render_env:
        env.render()
        for a in results['traj']:
            sleep(0.5)
            env.step(CHARS_LOOKUP_ACTIONS[a])
            env.render()
        env.sleep(60)

    if args.write_to_file:
        orig_stdout = sys.stdout
        with open(f"{PATH}/{results['algorithm']}_results.txt", 'a') as file:
            # file.write(json.dumps(results)) # in case of json
            sys.stdout = file
            print_search_algorithm_results(results)
            sys.stdout = orig_stdout
            file.close()

def mcts_solve(args):

    # Create the environment.
    env = create_environment(args)

    # Initialize the Monte-Carlo-Tree-Search.
    # mcts = Mcts(Env=env,
    #             simulation_policy=args.sim_policy,
    #             max_rollouts=args.max_rollouts,
    #             max_depth=args.max_depth,
    #             num_parallel=args.num_parallel)
    # mcts.initialize_search()
    mcts = Mcts(Env=env,
                actions=LEGAL_ACTIONS,
                max_rollouts=args.max_rollouts,
                max_depth=args.max_depth)


    # Must run this once at the start, so that noise injection actually affects
    # the first action of the episode.
    # first_node, _ = mcts.root.select_and_expand()
    # first_node.backpropagate(0, mcts.root)
    # first_node.N -= 1

    time_limit = arguments.time_limit * 60
    start_time = time()

    a_traj = []
    actions = []
    while True:
        mcts.Env.render_colored()
        now = time()
        if now - start_time > time_limit:
            print(f"Time limit of {args.time_limit} reached.")
            break

        #env.render(mode=args.render_mode)

        # Run the Monte-Carlo-Tree-Search for the current state and take the
        # best action after all simulations were performed.
        obs, reward, done, info, best_action = mcts.take_best_action()
        a_traj.append(best_action)

        print(f"\nMCTS's decision: "
              f"Move: {env.get_action_lookup_chars(best_action)}  "
              f"Action: {best_action}\n"
              f"{args.max_rollouts} simulations ({round(time()-start_time, 1)} sec)\n"
              f"action trajectory: {env.print_actions_as_chars(a_traj)}\n\n")

        if "action.name" in info:
            actions.append(ACTION_MAP[info["action.name"]])
        if done and "mcts_giveup" in info:
            env.reset()
            actions.clear()
        elif done and info["all_boxes_on_target"]:
            mcts.Env.render_colored()
            print(f"Solved: Solution found!\n"
                  f"        trajectory:   {mcts.Env.print_actions_as_chars(a_traj)}\n"
                  f"        total reward: {reward}\n"
                  f"        total time:   {round(time()-start_time, 1)} sec")
            actions.append("Solved in {:.0f} mins".format((now - start_time)/60))
            break
        elif done and info["maxsteps_used"]:
            env.reset()
            actions.clear()

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

    # env.render(mode=args.render_mode)
    # sleep(30)
    # env.close()
    print(f"total time = {time()-start_time}")


def solve_game(argus):
    """
    Try to solve Sokoban using different methods.

    Args:
        argus (Namespace): Arguments which describe necessary parameters.
    """
    if argus.search_algo:
        search_algorithms_solve(argus)
    else:
        mcts_solve(argus)


def read_input_boards_and_solve(input_args):
    if input_args.file_name:
        for file in input_args.file_name:
            input_args.file_name = Path(file)
            print(f"input_args.file_name={input_args.file_name}")
            solve_game(input_args)
    elif input_args.folder_name:
        for file in Path(input_args.folder_name).iterdir():
            input_args.file_name = file
            solve_game(input_args)
    else:
        solve_game(input_args)


if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=np.str, default="", #"#MICROBAN_BOARDS,
                        nargs="+",
                        help="Name of file(s) to read Sokoban boards from.")
    parser.add_argument("--folder_name", type=np.str, default="",
                        help="Name of a folder in which Sokoban boards are.")
    parser.add_argument("--input_type", type=np.str, default="human",
                        help="Type of representation of the Sokoban boards "
                             "in the file. Options are: "
                             "[`human`, `raw`]")
    parser.add_argument("--micro_levels", type=np.str,
                        default=MICROBAN_LEVELS_TO_SOLVE, nargs="+",
                        help="The Microban levels which should be solved.")
    parser.add_argument("--seeds", type=np.str, default=RANDOM_SEED,
                        nargs='+',
                        help="Seed(s) to handle the rendered board. "
                             "Multiple seeds are splitted by a space.")
    parser.add_argument("--render_mode", default="human",
                        help="How to observe the game. `human` will render a window.")
    parser.add_argument("--dim_room", type=np.int, default=DIM_ROOM,
                        help="Dimension of the Sokoban board")
    parser.add_argument("--num_boxes", type=np.int, default=NUM_BOXES,
                        help="Number of boxes on the board")
    parser.add_argument("--max_rollouts", type=np.int, default=8000,
                        help="Number of rollouts (simulations) per move")
    parser.add_argument("--max_depth", type=np.int, default=50,
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
                             f"[`{SIMULATION_POLICIES['random']}`, "
                             f"`{SIMULATION_POLICIES['eps-greedy']}`]")
    parser.add_argument("--search_algo", type=np.str,
                        default=SEARCH_ALGORITHMS[ALGORITHM_NAME_IDA_STAR],
                        help="Alternative search algorithm to solve the game. "
                             "Implemented options: "
                            f"[`{SEARCH_ALGORITHMS[ALGORITHM_NAME_DFS]}`, "
                            f"`{SEARCH_ALGORITHMS[ALGORITHM_NAME_BFS]}`, "
                            f"`{SEARCH_ALGORITHMS[ALGORITHM_NAME_UCS]}`, "
                            f"`{SEARCH_ALGORITHMS[ALGORITHM_NAME_A_STAR]}`, "
                            f"`{SEARCH_ALGORITHMS[ALGORITHM_NAME_IDA_STAR]}`]")
    parser.add_argument("--heuristic", type=np.str,
                        default=HEURISTICS["manhattan"],
                        help="Heuristic method for IDA* search algorithm. "
                             "Implemented options: "
                             f"[`{HEURISTICS['manhattan']}`, "
                             f"`{HEURISTICS['hungarian']}`]")
    parser.add_argument("--time_limit", type=np.int, default=300,
                        help="Time (in minutes) per board")
    parser.add_argument("--render_env", type=np.bool, default=False,
                        help="If the result environment should be rendered.")
    parser.add_argument("--print_results", type=np.bool, default=True,
                        help="If the result metrics should be printed.")
    parser.add_argument("--write_to_file", type=np.bool, default=True,
                        help="If the results should be written to a file.")
    arguments = parser.parse_args()

    # Solve the game.
    # for micro_level in arguments.micro_levels:
    #     arguments.micro_levels = "1"#micro_level # 31
    #     arguments.seeds = int(RANDOM_SEED[0])
    #     read_input_boards_and_solve(arguments)

    for seed in arguments.seeds:
        print(f"\n\nseed={seed}")
        arguments.seeds = int(seed)
        read_input_boards_and_solve(arguments)


