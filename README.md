# Teaching an AI to solve Sokoban

Sokoban is Japanese for warehouse keeper and a traditional video game. The game is a transportation
puzzle, where the player has to push all boxes in the room on the storage locations (targets). 

The following algorithms will be implemented. As a comparison for the RL approaches we implement 
basic search algorithms as 
    
- Depth-First-Search 
- Breadth-First-Search
- Best-First-Search
- A* 
- Uniform-Cost-Search
  
RL approaches will be 

- AlphaGo approach (MCTS + network)  
- DQL 


## Prior Steps
To make the environment running some prior steps may need to be done. Either all 
or a subset of the following commands need to be run to display the environment.

```bash
> brew install swig
> pip3 install PyOpenGL
> pip3 install pyglet==1.5.11
> pip3 install box2d
```

In addition, when running on _Mac OS Big Sur_ the file `cocoalibs.py` needs to be changed by 

```python
appkit = cdll.LoadLibrary(util.find_library('AppKit'))                           # remove this 
appkit = cdll.LoadLibrary('/System/Library/Frameworks/AppKit.framework/AppKit')  # add this
```

as well as `lib.py` by changing the function `find_framework(...)` by replacing 
the body with

```python
return '/System/Library/Frameworks/OpenGL.framework/OpenGL' 
```

---

## 3 Environment

The [mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban) repository is being used as an environment 
to train the agent in. 

## Action Space
```python
A = {
    0 : "no operation"
    1 : "push up" 
    2 : "push down"
    3 : "push left"
    4 : "push right"
    5 : "move up"
    6 : "move down" 
    7 : "move left" 
    8 : "move right" 
}
```

### Step 

```python
observation, reward, done, info = env.step(action)
```
with

-  `observation` (__object__): an environment-specific object representing your observation of the environment
   ```python 
   # represents the pixel data (rgb) of the game board
   ```

- `reward` (__float__): amount of reward achieved by the previous step

- `done` (__boolean__): whether its time to `reset` the environment. 
    ```python
    done == True  # either (1) all boxes on target 
                  #     or (2) max. number of steps reached 
    ``` 

-  `info` (__dict__): diagnostic info useful for debugging. 
    ```python
    info["action.name"]          # which action was taken
    info["action.moved_player"]  # did the agent move? True: yes, False: no
    info["action.moved_box"]     # did a box was being moved? True: yes, False: no
    ``` 

### `env.room_state`

The matrix `env.room_state` consists of the following elements
```python
0 : # wall        (outside the game) 
1 : # empty space (inside the game)
2 : # box target 
3 : # box on target
4 : # box not on target
5 : # agent/player
```

## Deep Model-Based RL 

## Notes 

__How to construct the trees of MCTS?__

- for each move we save the current state of the board, e.g. `(0, 0, 1, ... 4, 5)`
    - and execute the mcts steps from there 
    - __save the constructed tree for that state__ in memory !!! 
    - s.t. we have a tree for all visited board states 
    
- [ ] What is the input of the NN? The states of the last `x` moves? Only the current state? 
    - if its the states of the last `x` moves then what the the layers detect?  
    - if its the current state, should there be planes only containing subinformation like: 
    the state but only the boxes in it. the state but only the goal states in it, ...? 
