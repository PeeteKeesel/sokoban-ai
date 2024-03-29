import random
import numpy as np
import marshal


def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False):
    """
    Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
    0 = wall
    1 = empty space
    2 = box target empty
    3 = box off target
    4 = box on target
    5 = player

    Arguments:
        dim                 (tuple): the 2d-dimension of the board.
        p_change_directions (float): the probability of changing the direction of a walk.
        num_steps           (int):   number of times to generate a room topology.

    Returns:
        (Numpy 2d Array): represents the Sokoban room.
    """
    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)

    # Some times rooms with a score == 0 are the only possibility.
    # In these case, we try another model.
    for t in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

        # Room fixed represents all not movable parts of the room
        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        # Room structure represents the current state of the room including movable parts
        room_state = room.copy()
        room_state[room_state == 2] = 4

        #print(f"\nroom_structure = \n{room_structure}")
        #print(f"room_state       = \n{room_state}\n")

        room_state, score, box_mapping = reverse_playing(room_state, room_structure)
        room_state[room_state == 3] = 4

        if score > 0:
            break

    if score == 0:
        raise RuntimeWarning('Generated Model with score == 0')

    return room_structure, room_state, box_mapping

def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15):
    """
    Generate a room topology, which consits of empty floors and walls.

    Arguments:
        dim                 (tuple): the 2d-dimension of the room.
        p_change_directions (float): the probability of changing the direction of a walk.
        num_steps           (int):   number of times to generate a room topology.

    Returns:
        (Numpy 2d Array): the room with the filled integers to indicate where the room elements are.
    """
    dim_x, dim_y = dim

    # The ones in the mask represent all fields which will be set to floors
    # during the random walk. The centered one will be placed over the current
    # position of the walk.
    masks = [
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0]
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ]
    ]

    # Possible directions during the walk
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = random.sample(directions, 1)[0]

    # Starting position of random walk
    position = np.array([
        random.randint(1, dim_x - 1),
        random.randint(1, dim_y - 1)]
    )

    level = np.zeros(dim, dtype=int)

    for s in range(num_steps):

        # Change direction randomly
        if random.random() < p_change_directions:
            direction = random.sample(directions, 1)[0]

        # Update position
        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        # Apply mask
        mask = random.sample(masks, 1)[0]
        mask_start = position - 1
        level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level

def place_boxes_and_player(room, num_boxes, second_player):
    """
    Places the player and the boxes into the floors in a room.

    Arguments:
        room      (numpy array): the room structure in 2d.
        num_boxes (int):         the number of boxes to place.

    Returns:
        (Numpy 2d Arra): the room with the places player and boxes.
    """
    # Get all available positions
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
            num_possible_positions,
            num_players,
            num_boxes)
        )

    # Place player(s)
    ind = np.random.randint(num_possible_positions)
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    room[player_position] = 5

    if second_player:
        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

    # Place boxes
    for n in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = np.random.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        room[box_position] = 2

    return room


# Global variables used for reverse playing.
explored_states = set()
num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None
counter = 0


def reverse_playing(room_state, room_structure, search_depth=100):
    """
    This function plays Sokoban reverse in a way, such that the player can
    move and pull boxes.

    It ensures a solvable level with all boxes not being placed on a box target.

    Arguments:
        room_state      (numpy array): 2d-state of the room with walls, empty states, boxes on target and the agent.
        room_structure  (numpy array): represents the current state of the room including movable parts.
        search_depth    (int):         number of steps to search in depth.
    Returns:
        best_room:        TODO what is best_room?
        best_room_score:  TODO what is best_room_score?
        best_box_mapping: TODO what is best_box_mapping?
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping

    # box_mapping is used to calculate the box displacement for every box
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box

    # explored_states globally stores the best room state and score found during search
    explored_states = set()
    best_room_score = -1
    best_box_mapping = box_mapping
    depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300)

    return best_room, best_room_score, best_box_mapping


def depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300):
    """
    Searches through all possible states of the room.

    This is a recursive function, which stops if the ttl is reduced to 0 or
    over 1.000.000 states have been explored.

    Arguments:
        room_state     (numpy array): state of the room with walls, empty states, boxes on target and the agent.
        room_structure (numpy array): state of the room with walls, empty states and empty box targets.
                                      The agent is an empty state.
        box_mapping    (dict):        stores the initial and current position of each box.
                                      used to calculate the box displacement for every box.
        box_swaps      (int):         Number of times any box was moved.
        last_pull      (tuple):       position to which the box was pushed, and because of a reverse action it was
                                      pulled from there.
        ttl            (int):         time-to-live

    Returns:
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping, counter

    #counter = counter + 1
    #if counter % 50_000 == 0:
    #   print(counter)

    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 300_000:
        return

    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    if not (state_tohash in explored_states):

        # Add current state and its score to explored states
        room_score = box_swaps * box_displacement_score(box_mapping)
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = room_state
            best_room_score = room_score
            best_box_mapping = box_mapping

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            # The state and box mapping  need to be copied to ensure
            # every action start from a similar state.
            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()

            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1

            depth_first_search(room_state_next, room_structure,
                               box_mapping_next, box_swaps_next,
                               last_pull, ttl)


def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    """
    Perform reverse action. Where all actions in the range [0, 3] correspond to
    push actions and the ones greater 3 are simple move actions.

    Arguments:
        room_state     (numpy array): state of the room with walls, empty states, boxes on target and the agent.
        room_structure (numpy array): state of the room with walls, empty states and empty box targets.
        box_mapping    (dict):        stores the initial and current position of each box.
                                      Used to calculate the box displacement for every box.
        last_pull      (tuple):       position to which the box was pushed
        action         (int):         the reverse action the agent wants to do.

    Returns:
        room_state, box_mapping and last_pull after the reverse move
    """
    player_position = np.where(room_state == 5)
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change

    # Check if next position is an empty floor or an empty box target
    if room_state[next_position[0], next_position[1]] in [1, 2]:

        # Move player, independent of pull or move action.
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = 5

        # In addition try to pull a box if the action is a pull action
        if action < 4:
            #print(f"change={change}")
            possible_box_location = change[0] * -1, change[1] * -1
            possible_box_location += player_position
            #print(f"possible_box_location={possible_box_location}")

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                # Perform pull of the adjacent box
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]]

                # Update the box mapping
                for k in box_mapping.keys():
                    if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                        box_mapping[k] = (player_position[0], player_position[1])
                        last_pull = k

    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping):
    """
    Calculates the sum of all Manhattan distances, between the boxes
    and their origin box targets.

    Arguments:
        box_mapping (dict): stores the initial position of a box as the key
                            and its current position as the value.
                            Stores this for all boxes on the board.

    Returns:
        (float) sum of all Manhattan distances, between the boxes and their
                origin box targets. Meaning, how good are the boxes placed
                right now or how bad are they placed respectively.
    """
    score = 0
    
    for box_target in box_mapping.keys():
        box_location = np.array(box_mapping[box_target])
        box_target = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target))
        score += dist

    return score


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

