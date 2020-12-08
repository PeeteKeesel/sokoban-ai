

# Teaching an AI to solve Sokoban

Sokoban is Japanese for warehouse keeper and a traditional video game. The game is a transportation
puzzle, where the player has to push all boxes in the room on the storage locations (targets). 

![Example World](SokobanExample1)

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

# Environment

The [mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban) repository is being used as an environment 
to train the agent in. 

### Action Space
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

- `done` (__boolean__): whether its tome to `reset` the environment. 
    ```python
    done == True  # episode has terminated  
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
0 : # brick wall   (outside the game) 
1 : # black ground (inside the game)
2 : # goal position for a box 
3 : # box on a goal position
4 : # box
5 : # agent 
```

---


## ToDo's

- [ ] try out different rl algo's to get sense about the problem
    - [ ] Value Iteration
    - [ ] Policy Iteration
    - [ ] Q-Learning
- [ ] include unit-tests when starting to _really_ implement
- [ ] How to play with __one__ world to test agents behaviour
- [ ] Research on what algo's could be efficient
- [ ] Research on how to handle deadlocks
