## Searching Algorithms

### Agents

- Problem Solving Agents/Rational Agents - an entity that has goals and tries to perform a series of actions that yield the best outcome.

- Reflex agents don't think about consequences of it's actions, and selects an action based on current state of the world.

### State Spaces and Search

- State space - set of all states that are possible and can be reached in an environment/system.

- Start State - the state from where agent begins the search.

- Goal test: a function which observe the current state and returns whether the goal state is achieved or not.

- Search tree : a tree representation of search problem.

- State space size : total number of states. Counted using fundamental counting principle.

| State Space Graph  | Search Tree |
|-|-|
| ![State space graph](../assets/State%20space%20graph.png)  | ![Search tree](../assets/Search%20tree.png) |

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

### 3. Uniform Cost Search

- Explore the lowest cost node from the starting node
- complete
- optimal assuming all edges (cost) are non negative
- TC - $O(b^{C∗/ε})$  
- SC - $O(b^{C∗/ε})$  
where $C*$ is optimal path cost and ε is minimal cost between 2 nodes

### 4. Iterative Deepening Search

- Perform DFS for every depth level (combination of DFS and BFS)
- TC - $O(b^d)$
- SC - $O(bd)$

### 5. Bidirectional Search  

- Performed on search graph
- Two simultaneous search - forward search from start vertex toward goal vertex and backward search from goal vertex toward start vertex
- complete if we use BFS in both searches
- optimal
- TC - $O(b^d)$
- SC - $O(bd)$


## Informed Search

In this form of search we have some notion of the direction in which we should focus our search hence improving performance

A heuristic function $h(n)$ provides an estimate of the cost of the path from a given node to the closest goal state.

A heuristic is
1. admissible if $0 ≤ h(n) ≤ h^*(n)$ $ where $h(n)$ is heuristic cost and $h^*(n)$ is estimated cost 
2. consistent if $h(n) ≤ c(n,a,n') + h(n')$ 

Remember every consistent heuristic is also admissible

### 1. Best First Search/Greedy Search

- Explore the node with lowest heuristic value which also brings it closer to the goal
- identical to UCS but with a priority queue 
- not guaranteed to be complete or optimal

### 2. A* Search Algorithm

- Explore the node with lowest total cost value 
- Also uses priority queue
- complete
- optimal

> Notes for Local Search need to be added
