import time


###############################################################################
# Global variables
#   Supporting variables for other files.
#   If any global values need to be changed, e.g. the name a user needs to
#   input to use the `eps-greedy` simulation policy, then the value of the
#   dictionary SIMULATION_POLICIES needs to be changed.
#   example: The input should be `epsgreedy` instead of `eps-greedy`, then
#            change
#            SIMULATION_POLICIES = {"random": "random",
#                                   "eps-greedy": "epsgreedy"}
###############################################################################

# Names of the search algorithms. This is important to the code.
ALGORITHM_NAME_DFS = "dfs"
ALGORITHM_NAME_BFS = "bfs"
ALGORITHM_NAME_UCS = "ucs"
ALGORITHM_NAME_A_STAR = "astar"
ALGORITHM_NAME_IDA_STAR = "idastar"
ALGORITHM_NAME_MCTS = "mcts"

# Different search algorithms instead of MCTS. The keys are important to the
# code. The valuea are used as the names the user has to input.
SEARCH_ALGORITHMS = {ALGORITHM_NAME_DFS: "dfs",
                     ALGORITHM_NAME_BFS: "bfs",
                     ALGORITHM_NAME_UCS: "ucs",
                     ALGORITHM_NAME_A_STAR: "astar",
                     ALGORITHM_NAME_IDA_STAR: "idastar"}

# Different heuristics to use for IDA*.
HEURISTICS = {"manhattan": "manhattan",
              "hungarian": "hungarian"}

# Different types of simulation/rollout policies used in mcts.py
SIMULATION_POLICIES = {"random": "random",
                       "eps-greedy": "eps-greedy"}

LEVEL_FORMAT = {
    0: '#',  # wall
    1: ' ',  # empty space
    2: 'T',  # box target
    3: '*',  # box on target
    4: 'B',  # box not on target
    5: '@',  # agent
}

BG_COLORS = {
    0: "black",   # wall
    1: "white",   # empty space
    2: "red",     # box target
    3: "blue",    # box on target
    4: "orange",    # box not on target
    5: "green",   # agent
}

ACTION_LOOKUP_LONG = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

ACTION_LOOKUP_CHARS_LONG = {
    0: 'n',
    1: 'U',
    2: 'D',
    3: 'L',
    4: 'R',
    5: 'u',
    6: 'd',
    7: 'l',
    8: 'r',
}

CHARS_LOOKUP_ACTIONS_LONG = {
    'n': 0,
    'U': 1,
    'D': 2,
    'L': 3,
    'R': 4,
    'u': 5,
    'd': 6,
    'l': 7,
    'r': 8,
}

ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right'
}

ACTION_LOOKUP_CHARS = {
    1: 'U',
    2: 'D',
    3: 'L',
    4: 'R'
}

CHARS_LOOKUP_ACTIONS = {
    'U': 1,
    'D': 2,
    'L': 3,
    'R': 4
}

# Moves are mapped to coordinate changes as follows
CHANGE_COORDINATES = {
    0: (-1, 0), # 0: Move up
    1: (1, 0),  # 1: Move down
    2: (0, -1), # 2: Move left
    3: (0, 1)   # 3: Move right
}

RENDERING_MODES = ['colored', 'rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw', 'format']

BACKGROUND_COLORS = {"black": "0;{};40",
                     "red": "0;{};41",
                     "green": "0;{};42",
                     "orange": "0;{};43",
                     "blue": "0;{};44",
                     "purple": "0;{};45",
                     "dark green": "0;{};46",
                     "white": "0;{};47"}

COLORS = {"black": "30",
          "red": "31",
          "green": "32",
          "orange": "33",
          "blue": "34",
          "purple": "35",
          "olive green": "36",
          "white": "37"}

# Metrics to save for output reasons.
METRICS_SCELETON = {
    'no_of_nodes_discovered': 0,  # The total number of discovered
                                  # nodes. Including repeated ones.
    'no_of_nodes_repeated': 0,  # The number of a times nodes got
                                # discovered repeatedly.
    'nodes_explored': set(),  # The set of all discovered nodes
                              # excluding duplications.
    'environemnts': set(),  # This saves the environment of the nodes.
    'action_traj': [],  # The trajectory of action taken.
    'time': 0  # The time it took until the current node.
}

# Messages for the outcome of levels.
MESSAGE_SOLUTION_FOUND = "solution-found"
MESSAGE_TIME_LIMIT_REACHED = "time-limit-reached"
MESSAGE_MAX_STEPS_PERFORMED = "max-steps-performed"

OUTCOMES = {MESSAGE_SOLUTION_FOUND: "WIN",
            MESSAGE_TIME_LIMIT_REACHED: "FAIL",
            MESSAGE_MAX_STEPS_PERFORMED: "FAIL"}

MESSAGES = {MESSAGE_SOLUTION_FOUND: "SOLUTION FOUND",
            MESSAGE_TIME_LIMIT_REACHED: "TIME LIMIT REACHED",
            MESSAGE_MAX_STEPS_PERFORMED: "MAXIMUM NUMBER OF STEPS PERFORMED"}


###############################################################################
# Methods
#   Supporting methods for other files
###############################################################################

def log(test_env, iteration, step_idx, total_reward):
    """
    Logs one step in a testing episode.

    Arguments:
        test_env:       Test environemnt that should be rendered.
        iteration:      Number of training iterations so far.
        step_idx:       Index of the step in the episode.
        total_reward:   Total reward collected so far.
    """
    time.sleep(.3)
    print()
    print(f"Training Episodes: {iteration}")
    test_env.colored_print()
    print(f"Step:   {step_idx}")
    print(f"Return: {total_reward}")

def manhattan_distance(pos1, pos2):
    """
    Returns the Manhattan distance between two 2-dimensional points.
    Generally, in a 2d-grid: What is the minimal number of vertical and horizontal
    steps to go to come from position {@pos1} to position {@pos2}.

    Arguments:
        pos1  (2d-list) or (2d-tuple)  - Position in a 2-dimensional plane.
        pos2  (2d-list) or (2d-tuple) - Position in a 2-dimensional plane.
    Returns:
        (float)  - The Manhattan distance between pos1 and pos2.
    """
    assert len(pos1) == len(pos2) == 2
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def convert_room_state_to_output_format(mat):
    for key, value in LEVEL_FORMAT.items():
        mat[mat==str(key)] = value
    return mat

def print_room_state(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print(mat[i][j], end='')
        print()
    print()

def colored_print(text, color, background_color, end=""):
    """
    Prints text with color.
    """
    color_string = BACKGROUND_COLORS[background_color].format(COLORS[color])
    text = f"\x1b[{color_string}m{text}\x1b[0m{end}"
    if end == "":
        print(text, end=end)
    else:
        print(text)

def get_search_algorithm_results(search_algo_name, node_env, metrics, message):
    return {
        "algorithm": search_algo_name,
        "message": message,
        "outcome": OUTCOMES[message],
        "node_env": None,
        "steps": len(node_env.action_trajectory),
        "trajectory": node_env.print_actions_as_chars(node_env.action_trajectory),
        "discovered": metrics['no_of_nodes_discovered'],
        "repeated": metrics['no_of_nodes_repeated'],
        "explored": len(metrics['nodes_explored']),
        "time": round(metrics['time'], 3)
    }

def print_search_algorithm_results(results):
    print(80*"-" + "\n" +
          f"{results['algorithm']}(): {results['message']}!\n" +
          f"outcome:    {results['outcome']}\n" +
          f"steps:      {results['steps']}\n" +
          f"trajectory: {results['trajectory']}\n" +
          f"discovered: {results['discovered']}\n" +
          f"repeated:   {results['repeated']}\n" +
          f"explored:   {results['explored']}\n" +
          f"dim_room:   {(results['dim_room'], results['dim_room'])}\n" +
          f"num_boxes:  {results['num_boxes']}\n" +
          f"seed:       {results['seed']}\n" +
          f"time:       {results['time']} s")

###############################################################################
# Parse the Sokoban environment from a file.
###############################################################################

def read_sokoban_input(filename):
    """
    This reads a file containing a information about a Sokoban map and returns
    sets of tuples containing the size of the board, positions of walls, boxes,
    and goals as well as the player position.

    Args:
        filename (str): Name of the file to read the information from.

    Returns:
        (size, walls, boxes, storages, start):
            size (tuple): Dimension of the Sokoban room.

            walls (set): Set of tuples of the wall positions.

            boxes (set): Set of tuples of the box positions.

            storages (set): Set of tuples of the goal positions for the boxes.

            start (tuple): Starting position of the player.
    """
    with open(filename, 'r') as f:
        # read size
        line = f.readline()
        inputs = line.split()
        print(inputs)
        size = (int(inputs[0]), int(inputs[1]))

        # read walls
        line = f.readline()
        inputs = line.split()
        inputs.pop(0)
        walls = set()
        for i in range(0, len(inputs), 2):
            walls.add((int(inputs[i]), int(inputs[i+1])))

        # read boxes
        line = f.readline()
        inputs = line.split()
        inputs.pop(0)
        boxes = set()
        for i in range(0, len(inputs), 2):
            boxes.add((int(inputs[i]), int(inputs[i + 1])))

        # read storages
        line = f.readline()
        inputs = line.split()
        inputs.pop(0)
        storages = set()
        for i in range(0, len(inputs), 2):
            storages.add((int(inputs[i]), int(inputs[i + 1])))

        # read start position
        line = f.readline()
        inputs = line.split()
        start = (int(inputs[0]), int(inputs[1]))

    return size, walls, boxes, storages, start

# @param filename: name of file containing sokoban input
# @return (rows, cols): 2-tuple containing rows and columns of sokoban board
# @return len(boxes): number of boxes in the board
# @return map: 2d list containing the board with
#  '#' for walls, '$' for boxes, '.' for storages, '@' for player, and ' ' for empty spaces
# parse sokoban input and return the dimensions of board, number of boxes, and map
def parse(filename):
    size, walls, boxes, targets, start = read_sokoban_input(filename=filename)
    print(f"size= {size}\n"
          f"walls={walls}\n"
          f"boxes={boxes}\n"
          f"targets={targets}\n"
          f"start={start}")
    cols, rows = size
    mapping_file_to_board = [[""]*cols for i in range(rows)]
    for row in range(1, rows+1):
        for col in range(1, cols+1):
            if (row, col) in walls:
                mapping_file_to_board[row-1][col-1] = "#"
            elif (row, col) in boxes:
                mapping_file_to_board[row-1][col-1] = "$"
            elif (row, col) in targets:
                mapping_file_to_board[row-1][col-1] = "."
            elif (row, col) == start:
                mapping_file_to_board[row-1][col-1] = "@"
            else:
                mapping_file_to_board[row-1][col-1] = " "
    return (rows, cols), len(boxes), mapping_file_to_board
