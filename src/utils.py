import time

###############################################################################
# Methods                                                                     #
#   Supporting methods for other files                                        #
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


###############################################################################
# Global variables                                                            #
#   Supporting variables for other files                                      #
###############################################################################

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
