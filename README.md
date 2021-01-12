# Teaching an AI to solve Sokoban

Sokoban is Japanese for warehouse keeper and a traditional video game. The game is a transportation
puzzle, where the player has to push all boxes in the room on the storage locations (targets). 

| Sokoban-v0 Example | Sokoban-v1 Example | Sokoban-v2 Example |
| :---: | :---: | :---: 
| ![v-0](/docs/imgs/Sokoban-v0-Example.png?raw=true) | ![v-1](/docs/imgs/Sokoban-v1-Example.png?raw=true) | ![v-2](/docs/imgs/Sokoban-v2-Example.png?raw=true) |


## 2 Prior Steps
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

## 3.1 Action Space
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

### 3.2 Step 

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

### 3.3 `env.room_state`

The matrix `env.room_state` consists of the following elements
```python
0 : # wall        (outside the game) 
1 : # empty space (inside the game)
2 : # box target 
3 : # box not on target
4 : # box on target
5 : # agent/player
```

---


## 4 ToDo's


- [ ] try to implement __Single Agent__ from _Feng et al., 2020_ ([page 6](https://arxiv.org/pdf/2008.05598v2.pdf))
    - [ ] implement Resnet for _Learning_
    - [ ] implement MCTS for _Planning_
        - [ ] tests
- [ ] try out different search algorithms 
    - [ ] Backtracking 
        - [ ] tests
    - [ ] Depth First Search (DFS)
        - [ ] tests
    - [ ] Breadth First Search (BFS)
        - [ ] tests
    - [ ] Uniform Cost Search (UCS)
        - [ ] tests
    - [ ] A*
        - [ ] tests 
- [ ] implement deadlock detection to the database to check future positions 
- [ ] train CNN to predict best possible action for a given state  
- [ ] include unit-tests 
- [x] How to play with __one__ world to test agents behaviour
- [ ] Research on what algo's could be efficient

## 5 Deep Model-Based RL 
