# Searching Algorithms

## Parts of Search

### Problem

| Type                     | Examples |
| ------------------------ | -------- |
| Single-Agent             |          |
| Two-Agent                |          |
| Constraint-Satisfication |          |

### Agents

| Type of Agents                               |                                                              |
| -------------------------------------------- | ------------------------------------------------------------ |
| Problem Solving Agents/<br />Rational Agents | has goals and tries to perform a series of actions that yield the best outcome. |
| Reflex Agents                                | Don’t think about consequences of its actions, and selects an action based on current state of the world. |

### State Spaces and Search

|                  |                                                              |                                                           |
| ---------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| State Space      | Set of all states that are possible and can be reached in an environment/system. | ![State space graph](../assets/State%20space%20graph.png) |
| State space size | Total number of states. Counted using fundamental counting principle. |                                                           |
| Start State      | State from where agent begins the search                     |                                                           |
| Goal State       | State where success is attained, which we want to reach<br />Multiple goal states can exist |                                                           |
| Goal Test        | Function that observes current state and returns whether goal state is achieved/not |                                                           |
| Search Tree      | Tree representation of search problem.                       | ![Search tree](./assets/Search%20tree.png)                |

## Search Type

### Direction

| Direction         | Use-Case   |
| ----------------- | ---------- |
| Forward chaining  | Diagonosis |
| Backward chaining | Planning   |

| Information |      |
| ----------- | ---- |
| Informed    |      |
| Uninformed  |      |

## Uninformed Search 

Performed on a search tree.

Completeness - whether the algorithm is guaranteed to find a solution, given there exists a solution to the given search problem

Optimality - whether the strategy is guaranteed to find the lowest cost path to a goal state?

Time Complexity - a measure of time for an algorithm to complete its task.

Space Complexity: the maximum storage space required at any point during the search.

### 1. Depth First Search  

- Explore the deepest node from the starting node (i.e. leftmost node)
- Not complete  
- Not optimal  
- TC - $O(b^m)$  
- SC - $O(bm)$

  where there are $b$ nodes at each level and depth is $m$

### 2. Breadth First Search

- Explore the shallowest node from the starting node (i.e. traverse left to right, level by level)
- complete
- optimal
- TC - $O(b^s)$
- SC - $O(b^s)$
where there are $b$ nodes at each level (i.e. branching factor) and the shallowest solution is at depth $s$

### 3. Iterative Deepening Search

- Perform DFS for every depth level (combination of DFS and BFS)
- TC - $O(b^d)$
- SC - $O(bd)$

### 4. Uniform Cost Search

- Explore the lowest cost node from the starting node

- complete

- optimal assuming all edges (cost) are non negative

- TC - $O(b^{C^∗/ε})$  

- SC - $O(b^{C^∗/ε})$

  where $C^*$ is optimal path cost and $\epsilon$ is minimal cost between 2 nodes

Fringe is a priority queue

### 5. Bidirectional Search  

- Performed on search graph
- Two simultaneous search - forward search from start vertex toward goal vertex and backward search from goal vertex toward start vertex
- complete if we use BFS in both searches
- optimal
- TC - $O(b^d)$
- SC - $O(bd)$


## Informed Search

In this form of search we have some notion of the direction in which we should focus our search hence improving performance

### 1. Best First Search/Greedy Search

- Explore the node with lowest heuristic value which also brings it closer to the goal
- identical to UCS but with a priority queue 
- not guaranteed to be complete or optimal

### 2. A* Search Algorithm

- Explore the node with lowest total cost value 
- Also uses priority queue
- complete
- optimal

### IDK



> Notes for Local Search need to be added

## Iterative Improvement Search

Local Search

|                                              | States |                                                              | Advantage                        | Limitation                                                   |
| -------------------------------------------- | ------ | ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| Hill Climbing<br />(Gradient descent/ascent) | 1      | Like climbing Mount Everest with thick fog and amnesia       | Fast<br />Low memory requirement | Not complete<br />Irreversible steps<br />Stuck at local minima/maxima<br />Skips ridges |
| Local Beam                                   | $k$    | Choose $k$ successors<br />Similar to natural selection      |                                  | Inefficient                                                  |
| Simulated Annealing                          | 1      | Compromise between hill climbing and local beam<br />Probabilistic |                                  |                                                              |
| Genetic Algorithm                            |        |                                                              |                                  |                                                              |

## Heuristic

A heuristic function $h(x)$ provides an estimate of the cost of the path from a given node to the closest goal state

### Characteristics

| Characteristic | Definition                                          | Comment                                       |
| -------------- | --------------------------------------------------- | --------------------------------------------- |
| admissible     | $h(x, G) \in [0, C(x, G)] \quad \forall x$          |                                               |
| consistent     | $\vert h(x_1, G) - h(x_2, G) \vert \le C(x_1, x_2)$ | Every consistent heuristic is also admissible |

 where

- $h(x)$ is heuristic cost
- $C(x)$ is true cost 

