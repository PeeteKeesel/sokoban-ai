---

## ToDo's

#### General 
- [x] `state_after_action(self, a)` 
    - [x] test 
- [x] `successors()`
    - [x] test 
- [x] `set_children()`
    - [x] test
- [x] `get_children()`
    - [x] test
- [ ] class structure
    - [ ] tests
- [ ] for a given state, build a game tree until either max_steps is reached or the game is finished
    - [ ] test 
- [ ] clean code 
- [ ] track metrics to separate file for later plotting

#### Comparison Algorithms 
- [ ] implement different search algorithms 
    - [ ] Backtracking 
        - [ ] tests
    - [x] Depth First Search (DFS)
        - [ ] tests
    - [x] Breadth First Search (BFS)
        - [ ] tests
    - [ ] Best First Search 
        - [ ] tests 
    - [x] Uniform Cost Search (UCS)
        - [ ] tests
    - [x] A*
        - [ ] tests 
        - [x] `manhattan_distance` + test 
        - [x] `manhattan_heuristic` + test 
        - [ ] other Heuristics
        
#### RL Algorithm Ideas 
        
- [ ] __Single Agent__ from _Feng et al., 2020_ ([page 6](https://arxiv.org/pdf/2008.05598v2.pdf))
    - [ ] implement Resnets/ConvNets for _Learning_
    - [ ] implement MCTS for _Planning_
        - [ ] tests
- [ ] AlphaGo 
    - [ ] MCTS 
    - [ ] Convolutional Neural Network 
- [ ] Deep-Q-Learning
    - [ ] neural network 
    - [ ] tests  
    
#### Additional Todos 

- [ ] implement deadlock detection
- [ ] train CNN to predict best possible action for a given state  
- [x] How to play with __one__ world to test agents behaviour
- [x] Research on what algo's to implement

#### Optional Tasks 

- [ ] Deadlock detection (helps making the game tree sparser)